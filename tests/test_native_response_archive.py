"""四节点已知零位移/支反力档案：验证独立读取、分派及篡改检出，不求压力。"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_NUMERIC=all(importlib.util.find_spec(n) is not None for n in ('numpy','scipy','shapely'))


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    return value


def write_json(path,value):
    path.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_state(folder,state):
    path=folder/'state.json.gz'
    with gzip.open(path,'wt',encoding='utf8') as stream:json.dump(plain(state),stream,ensure_ascii=False)
    write_json(folder/'final_state_receipt.json',dict(source_state_sha256=sha(path),state_file=path.name,
        input_fingerprint=state['input_fingerprint'],accepted=True,source_identity_unchanged=True,
        response_receipt_file='response_receipt.json',response_receipt_sha256=sha(folder/'response_receipt.json')))
    return path


@unittest.skipUnless(HAS_NUMERIC,'需要可选NumPy/SciPy/GEOS')
class NativeResponseArchiveTests(unittest.TestCase):
    def fixture(self,folder):
        """接触三角固定，唯一自由顶点零载荷；u=0为已知精确弹性解。"""
        import numpy as np
        from scipy import sparse
        from bevel_gear.native_pressure_domain import NATIVE_INTEGRATION_METHOD
        triangles=np.asarray([[0,1,2],[0,3,1],[1,3,2],[2,3,0]])
        base=np.asarray([[-2.,-2.,0.],[2.,-2.,0.],[0.,2.,0.],[0.,0.,-1.]])+[.5,0.,0.]
        worlds={'pinion':base.copy(),'wheel':base.copy()};worlds['wheel'][3,2]=1.
        meshes={m:SimpleNamespace(nodes_m=v,tetrahedra=np.asarray([[0,1,2,3]]),boundary_triangles=triangles) for m,v in worlds.items()}
        frames={m+'_rotation_rows':np.eye(3) for m in worlds}
        frames.update(pinion_axis_world=[0.,1.,0.],wheel_axis_world=[0.,-1.,0.])
        pieces=[np.asarray([[-.25,-.25],[.25,-.25],[.25,.25]]),np.asarray([[-.25,-.25],[.25,.25],[-.25,.25]])]
        records=[];samples=[];W={m:np.zeros((12,1)) for m in worlds};levers={m:0. for m in worlds}
        for vertices in pieces:
            xy=vertices.mean(axis=0);point=np.r_[xy+[.5,0.],0.];mappings={};coefficients={};ells={}
            for member,sign in (('pinion',-1.),('wheel',1.)):
                world=worlds[member][:3];ab=np.linalg.solve((world[1:,:2]-world[0,:2]).T,point[:2]-world[0,:2]);shapes=np.r_[1.-ab.sum(),ab]
                normal=np.asarray([0.,0.,1. if member=='pinion' else -1.]);coefficient=np.asarray([0.,0.,sign])
                ell=-float(coefficient@np.cross(frames[member+'_axis_world'],point));levers[member]+=.5*ell
                mappings[member]=dict(nodes=np.arange(3),weights=shapes,mapped_world=point,height_m=0.,normal_world=normal,normal_cosine=normal[2])
                coefficients[member]=coefficient;ells[member]=ell
                for node,shape in enumerate(shapes):W[member][3*node:3*node+3,0]+=.5*shape*coefficient
            samples.append(dict(piece_vertices_xy_m=vertices,piece_area_plane_m2=.125,integration_weight=.5,
                target_xy_m=xy,mappings=mappings,coefficients_local=coefficients,levers_m=ells,gap_m=0.))
        record=dict(tooth_offset=0,row=0,column=0,area_plane_m2=.25,gap_m=0.,levers_m=levers,quadrature=samples)
        patch_geometry=dict(tooth_offset=0,frames=frames,source_material_reference_frames=frames,
            grid=dict(spacing_x_m=.5,spacing_y_m=.5,origin_xy_m=[0.,0.]),valid_mask=[[True]],gaps_m=[[0.]],
            plane_origin_world_m=[.5,0.,0.],plane_tangent_rows=[[1.,0.,0.],[0.,1.,0.]],plane_outward_normal_world=[0.,0.,1.],
            geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD)
        geometry=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            driving_angle_rad=0.,pinion_torque_nm=-.5,pinion_to_wheel_ratio=1.,reference_follower_angle_rad=0.,patches=[patch_geometry])
        write_json(folder/'index.json',dict(fixture='固定接触面、自由顶点零载荷的四节点弹性档案'))
        cache_sha=sha(folder/'index.json')
        system=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,geometry=geometry,
            native_cache_index_sha256=cache_sha,records=[record],matrices={m:sparse.csc_matrix(v) for m,v in W.items()})
        inputs=dict(native_cache_index_sha256=cache_sha,fixture='零位移支反力，不认证接触压力精度')
        write_json(folder/'inputs.json',inputs)
        fp=hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
        mesh_hashes={};bodies={};body_checks={}
        for member,mesh in meshes.items():
            file=folder/(member+'_mesh.json');write_json(file,vars(mesh));mesh_hashes[member]=sha(file)
            force=W[member].reshape(4,3);u=np.zeros((4,3))
            response=folder/(member+'_response.npz')
            np.savez_compressed(response,displacements_m=u,nodal_forces_n=force,ku_minus_f_n=-force,
                free_dofs=np.asarray([9,10,11]),force_plane_n=np.asarray([1.]),contact_cell_keys=np.asarray([[0,0,0]]))
            matrix=folder/(member+'_load_matrix.npz');sparse.save_npz(matrix,system['matrices'][member])
            bodies[member]=dict(response_file=response.name,response_sha256=sha(response),load_matrix_file=matrix.name,
                load_matrix_sha256=sha(matrix),load_matrix_shape=[12,1])
            body_checks[member]=dict(strain_energy_j=0.,external_work_j=0.,maximum_free_residual_n=0.,relative_free_residual=0.)
        receipt=dict(passed=True,input_fingerprint=fp,bodies=bodies);write_json(folder/'response_receipt.json',receipt)
        result=dict(converged=True,contact_representation='distributed_P0_surface_cells',pressure_filter_applied=False,
            halfspace_compliance_used=False,geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            native_cache_index_sha256=cache_sha,actual_native_mesh_sha256=mesh_hashes,source_input_fingerprint=fp,
            geometry=geometry,requested_tooth_offsets=[0],selected_tooth_offsets=[0],
            fields=[dict(tooth_offset=0,forces_n=[[1.]],pressure_pa=[[4.]],valid_mask=[[True]],contact_mask=[[True]],
                cell_area_m2=.25,elastic_deflection_m=[[0.]])],cells=[],quadrature_order='exact_affine_partition_centroids',
            active_cell_keys=[(0,0,0)],linearization_step_theta_rad=0.,loaded_follower_angle_rad=0.,total_loaded_te_rad=0.,
            body_response_checks=body_checks,fe_strain_energy_j=0.,pinion_torque_relative_residual=0.,
            geometry_linearization_checked=True,geometry_check=dict(passed=True),
            equilibrium=dict(forces_n=[1.],remaining_gaps_m=[0.],theta_rad=0.,strain_energy_j=0.,
                certificate=dict(passed=True,force_acceptance_n=1e-8,gap_acceptance_m=1e-10,complementarity_acceptance_nm=1e-9)))
        state=dict(accepted=True,source_identity_unchanged=True,native_cache_index_sha256=cache_sha,
            input_fingerprint=fp,result=result,response_receipt=receipt)
        path=write_state(folder,state)
        arguments=dict(run_root=folder,native_system=system,meshes=meshes,
            materials={m:dict(young_modulus_pa=210e9,poisson_ratio=.3) for m in meshes},
            fixed_nodes={m:[0,1,2] for m in meshes},mesh_source_sha256=mesh_hashes)
        return path,state,arguments

    def test_known_reaction_archive_rechecks_Ku_without_new_global_K(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        with tempfile.TemporaryDirectory() as folder:
            path,state,kwargs=self.fixture(Path(folder))
            with patch('scipy.sparse.linalg.splu',side_effect=AssertionError('不得分解新K')):
                result,proof=audit_native_response_archive(path,**kwargs)
            self.assertTrue(proof['passed']);self.assertTrue(proof['independent_Ku_recheck'])
            self.assertFalse(proof['factorization_performed']);self.assertFalse(proof['NURBS_gap_queried'])
            self.assertEqual(proof['maximum_average_gap_reconstruction_error_m'],0.)
            self.assertEqual(proof['condensed_energy_relative_error'],0.)

    def test_final_operator_file_and_sha_are_separate_from_original_static_cache(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);path,state,kwargs=self.fixture(folder)
            operator_file=folder/'operator_manifest.json';write_json(operator_file,dict(fixture='同已知解的最终几何 operator 档案'))
            fields=dict(current_operator_manifest_sha256=sha(operator_file),operator_fingerprint='e'*64)
            state['result'].update(fields,current_operator_manifest_file=operator_file.name)
            kwargs['native_system'].update(fields,current_operator_manifest_file=str(operator_file.resolve()))
            path=write_state(folder,state)
            _,proof=audit_native_response_archive(path,**kwargs)
            self.assertTrue(proof['passed'])
            self.assertEqual(proof['current_operator_manifest_sha256'],sha(operator_file))
            operator_file.write_text('{}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'manifest'):
                audit_native_response_archive(path,**kwargs)

    def test_production_archive_without_historical_cache_uses_real_numeric_mesh_identity(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        from bevel_gear.distributed_pressure_domain import numeric_array_identity
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);path,state,kwargs=self.fixture(folder)
            inputs=json.loads((folder/'inputs.json').read_text());inputs.pop('native_cache_index_sha256')
            write_json(folder/'inputs.json',inputs)
            fp=hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
            state.pop('native_cache_index_sha256');state['result'].pop('native_cache_index_sha256');kwargs['native_system'].pop('native_cache_index_sha256')
            state['input_fingerprint']=fp;state['result']['source_input_fingerprint']=fp
            state['response_receipt']['input_fingerprint']=fp;write_json(folder/'response_receipt.json',state['response_receipt'])
            state['result']['operator_input_fingerprint']='e'*64;kwargs['native_system']['operator_input_fingerprint']='e'*64
            state['result']['geometry']['operator_input_fingerprint']='e'*64
            state['result']['actual_native_mesh_numeric_identity']={m:dict(nodes_m=numeric_array_identity(mesh.nodes_m),
                tetrahedra=numeric_array_identity(mesh.tetrahedra,integer=True),
                boundary_triangles=numeric_array_identity(mesh.boundary_triangles,integer=True)) for m,mesh in kwargs['meshes'].items()}
            path=write_state(folder,state)
            _,proof=audit_native_response_archive(path,**kwargs)
            self.assertTrue(proof['passed']);self.assertTrue(proof['independent_Ku_recheck'])

    def test_NPZ_tamper_rejected_before_reusing_response(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        with tempfile.TemporaryDirectory() as folder:
            path,_,kwargs=self.fixture(Path(folder))
            with (Path(folder)/'pinion_response.npz').open('ab') as stream:stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'SHA'):audit_native_response_archive(path,**kwargs)

    def test_wrong_fixed_nodes_rejected_even_for_zero_displacement(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        with tempfile.TemporaryDirectory() as folder:
            path,_,kwargs=self.fixture(Path(folder));kwargs['fixed_nodes']['pinion']=[3]
            with self.assertRaisesRegex(ValueError,'固定节点'):audit_native_response_archive(path,**kwargs)

    def test_saved_mean_tamper_is_rejected_by_independent_true_FE_geometry(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        with tempfile.TemporaryDirectory() as folder:
            path,_,kwargs=self.fixture(Path(folder));kwargs['native_system']['records'][0]['gap_m']=1e-6
            with self.assertRaisesRegex(ArithmeticError,'独立重建未通过'):audit_native_response_archive(path,**kwargs)

    def test_false_saved_energy_cannot_pass_only_because_Ku_residual_is_zero(self):
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        with tempfile.TemporaryDirectory() as folder:
            path,state,kwargs=self.fixture(Path(folder))
            state['result']['body_response_checks']['pinion']['strain_energy_j']=1e-6
            write_state(Path(folder),state)
            _,proof=audit_native_response_archive(path,**kwargs)
            self.assertFalse(proof['passed'])
            self.assertGreater(proof['body_checks']['pinion']['original_energy_relative_error'],1e-8)


if __name__=='__main__':unittest.main()
