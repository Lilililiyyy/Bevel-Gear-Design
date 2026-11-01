"""原生有限FE积分协议、小几何及整体R0/全局节点身份验证；不求压力。"""
from copy import deepcopy
import importlib.util
import unittest
from bevel_gear.native_pressure_domain import (native_geometry_mode,native_result_identity,
    NATIVE_INTEGRATION_METHOD,audit_native_partition_geometry)
from bevel_gear.distributed_pressure_domain import distributed_loaded_pose, distributed_gauss_mean_gaps

HAS_NUMERIC=all(importlib.util.find_spec(name) is not None for name in ('numpy','scipy','shapely'))


class NativeIdentityTests(unittest.TestCase):
    def test_legacy_missing_fields_keeps_empty_identity(self):
        self.assertEqual(native_result_identity({'geometry':{}}),{})

    def test_native_label_alone_is_not_an_algorithm_certificate(self):
        result=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            geometry=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD))
        self.assertTrue(native_geometry_mode(result))
        with self.assertRaises(ValueError):native_result_identity(result)
        with self.assertRaises(ValueError):audit_native_partition_geometry(dict(geometry_source='FE_boundary',
            integration_method=NATIVE_INTEGRATION_METHOD),{})

    def test_conflicting_or_unknown_branches_cannot_fall_back_to_NURBS(self):
        with self.assertRaises(ValueError):native_geometry_mode(dict(geometry_source='FE_boundary',geometry=dict(geometry_source='NURBS')))
        with self.assertRaises(NotImplementedError):native_geometry_mode(dict(geometry_source='unknown',geometry={}))

    def test_native_pose_never_treats_increment_as_pure_elastic_TE(self):
        result=dict(geometry_source='FE_boundary', integration_method=NATIVE_INTEGRATION_METHOD,
            geometry=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
                driving_angle_rad=.2,pinion_to_wheel_ratio=.5,pinion_torque_nm=-64.,reference_follower_angle_rad=-.1001),
            equilibrium=dict(theta_rad=-.0002),linearization_step_theta_rad=-.0002,
            loaded_follower_angle_rad=-.1003,total_loaded_te_rad=-.0003)
        pose=distributed_loaded_pose(result)
        self.assertAlmostEqual(pose['total_loaded_te_rad'],-.0003)
        self.assertNotIn('total_elastic_te_rad',pose)
        result['total_loaded_te_rad']=-.0002
        with self.assertRaises(ValueError):distributed_loaded_pose(result)

    def test_current_operator_identity_is_atomic_and_never_a_NURBS_default(self):
        with self.assertRaises(ValueError):
            native_result_identity(dict(geometry={},current_operator_manifest_sha256='a'*64))
        with self.assertRaises(ValueError):
            native_result_identity(dict(geometry={},current_operator_manifest_file='operator/manifest.json'))


@unittest.skipUnless(HAS_NUMERIC,'需要可选NumPy/SciPy/GEOS')
class NativeGeometryTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        from scipy import sparse
        from types import SimpleNamespace
        rotations=dict(pinion=np.asarray([[np.cos(.2),-np.sin(.2),0.],[np.sin(.2),np.cos(.2),0.],[0.,0.,1.]]),
            wheel=np.asarray([[1.,0.,0.],[0.,np.cos(.3),-np.sin(.3)],[0.,np.sin(.3),np.cos(.3)]]))
        ids=np.asarray([1,3,4]);meshes={};worlds={};normals={};frames={}
        for member in ('pinion','wheel'):
            xy=np.asarray([[-2.,-2.],[2.,-2.],[0.,2.]])
            height=np.zeros(3) if member=='pinion' else .03+.02*xy[:,0]-.01*xy[:,1]
            world=np.column_stack((xy,height));nodes=np.zeros((6,3));nodes[ids]=world@rotations[member]
            meshes[member]=SimpleNamespace(nodes_m=nodes,boundary_triangles=np.asarray([ids]))
            worlds[member]=world;normal=np.cross(world[1]-world[0],world[2]-world[0]);normal/=np.linalg.norm(normal)
            normals[member]=normal if member=='pinion' else -normal
            frames[member+'_rotation_rows']=rotations[member];frames[member+'_axis_world']=[0.,0.,1.]
        vertices=[np.asarray([[-.25,-.25],[.25,-.25],[.25,.25]]),np.asarray([[-.25,-.25],[.25,.25],[-.25,.25]])]
        samples=[];W={m:np.zeros((18,1)) for m in meshes};levers={m:0. for m in meshes}
        for piece in vertices:
            xy=piece.mean(axis=0);mappings={};coefs={};ells={}
            for member,sign in (('pinion',-1.),('wheel',1.)):
                world=worlds[member];ab=np.linalg.solve((world[1:,:2]-world[0,:2]).T,xy-world[0,:2]);weights=np.r_[1.-ab.sum(),ab]
                point=weights@world;normal=normals[member];cosine=float(normal[2]);coef_world=sign*normal/cosine
                coefs[member]=rotations[member].T@coef_world;ells[member]=-float(coef_world@np.cross([0.,0.,1.],point))
                mappings[member]=dict(nodes=ids.copy(),weights=weights,mapped_world=point,height_m=point[2],
                    facet_index=0,normal_world=normal,normal_cosine=cosine,distance_m=0.)
                levers[member]+=.5*ells[member]
                for node,shape in zip(ids,weights):W[member][3*node:3*node+3,0]+=.5*shape*coefs[member]
            samples.append(dict(piece_vertices_xy_m=piece,piece_area_plane_m2=.125,integration_weight=.5,target_xy_m=xy,
                mappings=mappings,coefficients_local=coefs,levers_m=ells,gap_m=mappings['wheel']['height_m']-mappings['pinion']['height_m']))
        record=dict(tooth_offset=0,row=0,column=0,area_plane_m2=.25,gap_m=.03,levers_m=levers,quadrature=samples)
        patch=dict(tooth_offset=0,frames=frames,source_material_reference_frames=dict(pinion_rotation_rows=np.eye(3)),
            grid=dict(spacing_x_m=.5,spacing_y_m=.5,origin_xy_m=[0.,0.]),valid_mask=[[True]],
            plane_origin_world_m=[0.,0.,0.],plane_tangent_rows=[[1.,0.,0.],[0.,1.,0.]],plane_outward_normal_world=[0.,0.,1.],
            geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD)
        return dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            geometry=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,patches=[patch]),
            records=[record],matrices={m:sparse.csc_matrix(w) for m,w in W.items()}),meshes

    def test_true_FE_affine_mean_shapes_W_and_whole_R0(self):
        system,meshes=self.fixture();gaps,keys,proof=audit_native_partition_geometry(system,meshes)
        self.assertTrue(proof['passed']);self.assertAlmostEqual(gaps[0],.03,places=14)
        self.assertEqual(keys.tolist(),[[0,0,0]])
        self.assertFalse(proof['source_Rk_used_for_mechanics']);self.assertFalse(proof['NURBS_gap_queried'])
        self.assertFalse(proof['CAD_error_set_from_FE_closure']);self.assertFalse(proof['physical_pressure_verified'])

    def test_recomputed_levers_available_without_reusing_stored_values(self):
        system,meshes=self.fixture()
        _,_,proof=audit_native_partition_geometry(system,meshes,include_recomputed_values=True)
        self.assertEqual(set(proof['recomputed_lever_arms_m']),{'pinion','wheel'})
        self.assertAlmostEqual(proof['recomputed_lever_arms_m']['wheel'][0],system['records'][0]['levers_m']['wheel'])

    def test_updated_operator_cannot_be_bound_to_outer0_records(self):
        system,meshes=self.fixture();system['native_cache_index_sha256']='a'*64
        result=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            native_cache_index_sha256='a'*64,source_input_fingerprint='b'*64,
            actual_native_mesh_sha256=dict(pinion='c'*64,wheel='d'*64),geometry=system['geometry'],
            current_operator_manifest_sha256='e'*64,operator_fingerprint='f'*64)
        with self.assertRaisesRegex(ValueError,'operator'):
            audit_native_partition_geometry(system,meshes,result=result)
        system.update(current_operator_manifest_sha256='e'*64,operator_fingerprint='f'*64)
        _,_,proof=audit_native_partition_geometry(system,meshes,result=result)
        self.assertTrue(proof['passed'])
        self.assertEqual(proof['current_operator_manifest_sha256'],'e'*64)
        system['operator_fingerprint']='0'*64
        with self.assertRaisesRegex(ValueError,'operator'):
            audit_native_partition_geometry(system,meshes,result=result)

    def test_native_mean_reconstruction_cannot_enter_NURBS_sampler(self):
        system,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'FE_boundary'):
            distributed_gauss_mean_gaps(None,system)

    def test_NURBS_archive_explicitly_rejects_native_gzip(self):
        import gzip,json,tempfile
        from pathlib import Path
        from bevel_gear.distributed_pressure_domain import audit_distributed_response_archive
        system,_=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json.gz'
            with gzip.open(path,'wt',encoding='utf8') as stream:
                json.dump(dict(accepted=True,source_identity_unchanged=True,result=dict(
                    geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
                    geometry=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD)),
                    response_receipt={}),stream)
            with self.assertRaisesRegex(ValueError,'audit_native_response_archive'):
                audit_distributed_response_archive(path,run_root=folder,prepared_surface=None,meshes={},materials={})

    def test_native_archive_rejects_unaccepted_candidate_before_reading_response(self):
        import gzip,json,tempfile
        from pathlib import Path
        from bevel_gear.native_pressure_domain import audit_native_response_archive
        both={m:None for m in ('pinion','wheel')}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json.gz'
            with gzip.open(path,'wt',encoding='utf8') as stream:json.dump(dict(accepted=False,source_identity_unchanged=True),stream)
            with self.assertRaisesRegex(ValueError,'候选'):
                audit_native_response_archive(path,run_root=folder,native_system={},meshes=both,materials=both,
                    fixed_nodes=both,mesh_source_sha256=both)

    def test_piecewise_FE_roof_mean_differs_from_centre_gap(self):
        """折面g=1µm+.02|x|，原cell平均为6µm、中心仅1µm。"""
        import numpy as np
        from scipy import sparse
        from shapely.geometry import Polygon,box
        from types import SimpleNamespace
        xy=np.asarray([[-1.,-1.],[0.,-1.],[1.,-1.],[-1.,1.],[0.,1.],[1.,1.]])*1e-3
        triangles=np.asarray([[0,1,4],[0,4,3],[1,2,5],[1,5,4]])
        worlds={'pinion':np.column_stack((xy,np.zeros(len(xy)))),
            'wheel':np.column_stack((xy,1e-6+.02*np.abs(xy[:,0])))}
        meshes={m:SimpleNamespace(nodes_m=v,boundary_triangles=triangles) for m,v in worlds.items()}
        frame={m+'_rotation_rows':np.eye(3) for m in worlds}
        frame.update({m+'_axis_world':[0.,0.,1.] for m in worlds})
        area=1e-6;cell=box(-.0005,-.0005,.0005,.0005)
        samples=[];W={m:np.zeros((18,1)) for m in worlds};mean=0.;levers={m:0. for m in worlds}
        for ids in triangles:
            polygon=Polygon(xy[ids]).intersection(cell)
            if polygon.area==0:continue
            vertices=np.asarray(polygon.exterior.coords[:-1]);centre=np.asarray(polygon.centroid.coords[0]);weight=polygon.area/area
            mappings={};coefficients={};ells={};heights={}
            for member,sign in (('pinion',-1.),('wheel',1.)):
                world=worlds[member][ids];ab=np.linalg.solve((world[1:,:2]-world[0,:2]).T,centre-world[0,:2]);shapes=np.r_[1.-ab.sum(),ab]
                mapped=shapes@world;normal=np.cross(world[1]-world[0],world[2]-world[0]);normal/=np.linalg.norm(normal)
                coef=sign*normal/normal[2];ell=-float(coef@np.cross([0.,0.,1.],mapped))
                mappings[member]=dict(nodes=ids.copy(),weights=shapes,mapped_world=mapped,height_m=mapped[2],normal_world=normal,normal_cosine=normal[2])
                coefficients[member]=coef;ells[member]=ell;heights[member]=mapped[2];levers[member]+=weight*ell
                for node,shape in zip(ids,shapes):W[member][3*node:3*node+3,0]+=weight*shape*coef
            gap=heights['wheel']-heights['pinion'];mean+=weight*gap
            samples.append(dict(piece_vertices_xy_m=vertices,piece_area_plane_m2=polygon.area,integration_weight=weight,
                target_xy_m=centre,mappings=mappings,coefficients_local=coefficients,levers_m=ells,gap_m=gap))
        record=dict(tooth_offset=0,row=0,column=0,area_plane_m2=area,gap_m=mean,levers_m=levers,quadrature=samples)
        patch=dict(tooth_offset=0,frames=frame,grid=dict(spacing_x_m=.001,spacing_y_m=.001,origin_xy_m=[0.,0.]),
            valid_mask=[[True]],gaps_m=[[mean]],plane_origin_world_m=[0.,0.,0.],plane_tangent_rows=[[1.,0.,0.],[0.,1.,0.]],
            plane_outward_normal_world=[0.,0.,1.],geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD)
        system=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            geometry=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,patches=[patch]),
            records=[record],matrices={m:sparse.csc_matrix(w) for m,w in W.items()})
        gaps,_,proof=audit_native_partition_geometry(system,meshes)
        self.assertTrue(proof['passed']);self.assertAlmostEqual(gaps[0],6e-6,places=16)
        record['gap_m']=1e-6;patch['gaps_m']=[[1e-6]]
        _,_,proof=audit_native_partition_geometry(system,meshes)
        self.assertFalse(proof['passed'])
        self.assertAlmostEqual(proof['maximum_reconstruction_errors']['gap_m'],5e-6,places=16)
        # 生产回调也必须先独立核平均g0，不能只给几何专用工具接线。
        from bevel_gear.distributed_pressure_domain import audit_distributed_response
        result=dict(geometry_source='FE_boundary',integration_method=NATIVE_INTEGRATION_METHOD,
            geometry=system['geometry'],native_cache_index_sha256='a'*64,
            source_input_fingerprint='b'*64,actual_native_mesh_sha256=dict(pinion='c'*64,wheel='d'*64))
        system['native_cache_index_sha256']='a'*64
        prepared=SimpleNamespace(systems={m:SimpleNamespace(mesh=mesh) for m,mesh in meshes.items()})
        with self.assertRaisesRegex(ArithmeticError,'独立重建未通过'):
            audit_distributed_response(prepared,system,result)

    def test_partial_renormalization_or_missing_piece_is_rejected(self):
        system,meshes=self.fixture();system['records'][0]['quadrature']=system['records'][0]['quadrature'][:1]
        system['records'][0]['quadrature'][0]['integration_weight']=1.
        with self.assertRaises(ValueError):audit_native_partition_geometry(system,meshes)

    def test_real_global_FE_triangle_required_not_fake_node_triplet(self):
        system,meshes=self.fixture();system['records'][0]['quadrature'][0]['mappings']['pinion']['nodes'][0]=2
        with self.assertRaises(ValueError):audit_native_partition_geometry(system,meshes)

    def test_wrong_mechanical_R_caught_even_if_source_Rk_exists(self):
        import numpy as np
        system,meshes=self.fixture();system['geometry']['patches'][0]['frames']['pinion_rotation_rows']=np.eye(3)
        _,_,proof=audit_native_partition_geometry(system,meshes)
        self.assertFalse(proof['passed'])

    def test_nonfinite_mapping_cannot_become_zero_error(self):
        system,meshes=self.fixture();system['records'][0]['quadrature'][0]['mappings']['wheel']['weights'][0]=float('nan')
        with self.assertRaises(ValueError):audit_native_partition_geometry(system,meshes)

    def test_original_W_column_must_match_piece_geometry(self):
        system,meshes=self.fixture();system['matrices']['pinion'].data[0]*=2.
        _,_,proof=audit_native_partition_geometry(system,meshes);self.assertFalse(proof['passed'])


if __name__=='__main__':unittest.main()
