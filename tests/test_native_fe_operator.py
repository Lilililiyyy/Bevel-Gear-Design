"""新phi原生operator的零K数值、全请求覆盖和typed归档控制。"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from bevel_gear.halfspace_contact import RectangularContactGrid
from bevel_gear.native_fe_operator import (build_native_operator_at_pose,save_native_operator,load_native_operator,fingerprint,
    compact_fe_boundary,build_native_fe_contact_operator,check_native_operator_pose,advance_native_follower_frame)
from bevel_gear.projected_fe_partition import build_projected_fe_contact_system,evaluate_partition_remaining_gaps
from bevel_gear.projected_fe_partition_arrays import evaluate_array_piece_vertices
from test_projected_fe_partition import body,PLANE,rotation


def fixture():
    compact={m:{0:body(m,nx,ny,a=a,b=b,c=c,reverse=reverse)} for m,nx,ny,a,b,c,reverse in (
        ('pinion',3,4,.02,-.015,0.,False),('wheel',5,3,.01,.003,.0002,True))}
    for member,parts in compact.items():
        b=parts[0];b['global_node_ids']=np.arange(len(b['nodes_m']))*2+7;b['total_node_count']=int(b['global_node_ids'].max())+3
    grid=RectangularContactGrid(3,2,.4,.4,(-.4,-.2))
    frames={m+'_rotation_rows':np.eye(3) for m in compact}
    frames.update({m+'_axis_world':np.array([0.,1.,0.]) for m in compact});frames['native_reference_follower_angle_rad']=.04
    patch=dict(tooth_offset=0,grid=grid,valid_mask=np.ones(grid.shape,dtype=bool),source_valid_mask=np.ones(grid.shape,dtype=bool),
        plane_origin_world_m=PLANE['origin_world_m'],plane_tangent_rows=PLANE['tangent_rows'],plane_outward_normal_world=PLANE['normal_world'],
        frames=deepcopy(frames),source_material_reference_frames=deepcopy(frames))
    geometry=dict(patches=[patch],driving_angle_rad=.2,reference_follower_angle_rad=.04)
    return dict(geometry=geometry,native_cache_index_sha256='original-index-fixture'),compact,frames


def reference(base,compact,frames):
    grid=base['geometry']['patches'][0]['grid'];cells=[]
    for j,i in zip(*np.nonzero(base['geometry']['patches'][0]['source_valid_mask'])):
        x=grid.origin_xy_m[0]+i*grid.spacing_x_m;y=grid.origin_xy_m[1]+j*grid.spacing_y_m
        cells.append(dict(tooth_offset=0,row=int(j),column=int(i),bounds_xy_m=[x-grid.spacing_x_m/2,y-grid.spacing_y_m/2,x+grid.spacing_x_m/2,y+grid.spacing_y_m/2]))
    data=[dict(compact[m][0],rotation_rows=frames[m+'_rotation_rows'],axis_world=frames[m+'_axis_world']) for m in ('pinion','wheel')]
    return build_projected_fe_contact_system(*data,PLANE,cells)


class NativeOperatorTests(unittest.TestCase):
    def compare(self,actual,old):
        self.assertEqual(len(actual['records']),len(old['records']))
        for m,w in old['matrices'].items():
            difference=actual['matrices'][m]-w
            self.assertLess(float(np.max(np.abs(difference.data))) if difference.nnz else 0.,2e-13)
        for a,b in zip(actual['records'],old['records']):
            for key in ('tooth_offset','row','column'):self.assertEqual(a[key],b[key])
            self.assertAlmostEqual(a['gap_m'],b['gap_m'],places=13)
            for m in ('pinion','wheel'):self.assertAlmostEqual(a['levers_m'][m],b['levers_m'][m],places=13)
            oldq={(q['mappings']['pinion']['facet_index'],q['mappings']['wheel']['facet_index']):q for q in b['quadrature']}
            for q in a['quadrature']:
                r=oldq[(q['mappings']['pinion']['facet_index'],q['mappings']['wheel']['facet_index'])]
                self.assertAlmostEqual(q['gap_m'],r['gap_m'],places=13)
                for m in ('pinion','wheel'):
                    np.testing.assert_array_equal(q['mappings'][m]['nodes'],r['mappings'][m]['nodes'])
                    np.testing.assert_allclose(q['coefficients_local'][m],r['coefficients_local'][m],rtol=0,atol=2e-14)
                self.assertEqual(len(q['piece_vertices_xy_m']),len(r['piece_vertices_xy_m']))
                for vertex,value in zip(q['piece_vertices_xy_m'],q['piece_vertex_gap_m']):
                    nearest=np.argmin(np.linalg.norm(r['piece_vertices_xy_m']-vertex,axis=1))
                    self.assertAlmostEqual(value,r['piece_vertex_gap_m'][nearest],places=13)

    def test_multibatch_true_W_gap_lever_and_original_vertex_views(self):
        base,compact,frames=fixture();before=fingerprint(base)
        actual=build_native_operator_at_pose(base,compact,frames,.04,batch_cells=1)
        self.compare(actual,reference(base,compact,frames));self.assertEqual(fingerprint(base),before)
        a=actual['typed_patch_arrays'][0]
        self.assertEqual(len(a['piece_facet_indices']['pinion']),a['piece_count'])
        rng=np.random.default_rng(53);u={m:rng.normal(0,1e-6,(w.shape[0]//3,3)) for m,w in actual['matrices'].items()}
        old=evaluate_partition_remaining_gaps(actual,u,1e-7);new=evaluate_array_piece_vertices(a,u,1e-7)
        for x,y in zip(old['cell_checks'],new['cell_checks']):
            self.assertAlmostEqual(x['minimum_remaining_gap_m'],y['minimum_remaining_gap_m'],places=13)

    def test_new_phi_changes_wheel_only_and_preserves_requested_scope(self):
        base,compact,frames=fixture();old=build_native_operator_at_pose(base,compact,frames,.04)
        changed=deepcopy(frames);changed['wheel_rotation_rows']=rotation([0,1,0],.002)
        new=build_native_operator_at_pose(base,compact,changed,.042,batch_cells=2)
        self.compare(new,reference(base,compact,changed))
        p0=old['geometry']['patches'][0];p1=new['geometry']['patches'][0]
        np.testing.assert_array_equal(p0['first_points_world_m'],p1['first_points_world_m'])
        self.assertGreater(np.nanmax(np.abs(p0['gaps_m']-p1['gaps_m'])),1e-5)
        self.assertEqual(new['geometry']['reference_follower_angle_rad'],.042)
        np.testing.assert_array_equal(p1['source_valid_mask'],base['geometry']['patches'][0]['source_valid_mask'])
        self.assertEqual(fingerprint(p1['source_material_reference_frames']),fingerprint(base['geometry']['patches'][0]['source_material_reference_frames']))

    def test_partial_keeps_missing_pieces_and_never_normalizes_into_W(self):
        base,compact,frames=fixture();base['geometry']['patches'][0]['grid']=RectangularContactGrid(1,1,.5,.5,(.95,.95))
        base['geometry']['patches'][0]['source_valid_mask']=np.ones((1,1),bool);base['geometry']['patches'][0]['valid_mask']=np.ones((1,1),bool)
        system=build_native_operator_at_pose(base,compact,frames,.04);a=system['typed_patch_arrays'][0]
        self.assertEqual(a['partial_cell_count'],1);self.assertGreater(a['piece_count'],0)
        self.assertTrue(system['geometry']['patches'][0]['native_partial_cell_mask'].all())
        self.assertEqual(system['matrices']['pinion'].shape[1],0);self.assertFalse(a['partial_cell_renormalization_used'])
        self.assertGreater(a['cell_statuses'][0]['missing_area_plane_m2'],0.)

    def test_typed_archive_roundtrip_no_pickle_and_all_identities(self):
        base,compact,frames=fixture();system=build_native_operator_at_pose(base,compact,frames,.04,batch_cells=1)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'operator';identity=save_native_operator(system,folder,actual_mesh_sha256={'pinion':'p-mesh','wheel':'w-mesh'},source_identity={'builder':'test'})
            loaded=load_native_operator(folder);self.compare(loaded,reference(base,compact,frames))
            self.assertEqual(fingerprint(system['geometry']),fingerprint(loaded['geometry']))
            for key,value in identity.items():self.assertEqual(loaded[key],value)
            for file in folder.glob('*arrays.npz'):
                with np.load(file,allow_pickle=False) as a:self.assertTrue(all(a[k].dtype.kind!='O' for k in a.files))
            for m in ('pinion','wheel'):
                for name in ('data','indices','indptr'):np.testing.assert_array_equal(getattr(system['matrices'][m],name),getattr(loaded['matrices'][m],name))
            with self.assertRaises(FileExistsError):save_native_operator(system,folder,actual_mesh_sha256={})

    def test_modified_manifest_and_payload_rejected(self):
        base,compact,frames=fixture();system=build_native_operator_at_pose(base,compact,frames,.04)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'operator';save_native_operator(system,folder,actual_mesh_sha256={'pinion':'p','wheel':'w'})
            file=folder/'manifest.json';original=file.read_bytes();data=json.loads(original);data['whole_body_R0_frames']['pinion_axis_world']=[1,0,0]
            file.write_text(json.dumps(data),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'指纹'):load_native_operator(folder)
            file.write_bytes(original);payload=folder/'pinion_W.npz';payload.write_bytes(payload.read_bytes()+b'tamper')
            with self.assertRaisesRegex(ValueError,'SHA'):load_native_operator(folder)

    def test_streamed_operator_cannot_be_silently_archived_as_new_typed(self):
        base,_,_=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'重新求解'):save_native_operator(base,Path(tmp)/'operator',actual_mesh_sha256={})

    def test_public_mesh_input_without_cache_build_pose_and_archive(self):
        base,parts,frames=fixture();meshes={};compact={}
        parts['wheel'][0]['nodes_m'][:,2]+=.04
        for m in ('pinion','wheel'):
            b=parts[m][0];nodes=np.zeros((b['total_node_count'],3));nodes[b['global_node_ids']]=b['nodes_m']
            meshes[m]=SimpleNamespace(nodes_m=nodes,boundary_triangles=b['global_node_ids'][b['triangles']])
            compact[m]=compact_fe_boundary(meshes[m],m)
        source=base['geometry']['patches'][0]
        patches=[dict(tooth_offset=0,grid=source['grid'].as_dict(),plane=PLANE,requested_mask=source['source_valid_mask'],
            source_material_reference_frames={'caller_Rk':'explicit-reference'})]
        system=build_native_fe_contact_operator(compact,patches,frames,.04,driving_angle_rad=.2)
        self.assertNotIn('native_cache_index_sha256',system)
        self.assertEqual(len(system['operator_input_fingerprint']),64)
        self.assertEqual(system['geometry']['operator_input_fingerprint'],system['operator_input_fingerprint'])
        self.assertEqual(system['geometry']['patches'][0]['source_material_reference_frames'],patches[0]['source_material_reference_frames'])
        u={m:np.zeros_like(meshes[m].nodes_m) for m in meshes};theta=1e-8
        remaining=[r['gap_m']-theta*r['levers_m']['wheel'] for r in system['records']]
        checked=check_native_operator_pose(system,compact,u,theta,remaining)
        self.assertTrue(checked['passed'],checked.get('failure'));self.assertFalse(checked['whole_flank_nonpenetration_certified'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'operator';save_native_operator(system,path);loaded=load_native_operator(path)
            self.assertEqual(loaded['operator_input_fingerprint'],system['operator_input_fingerprint'])
            self.assertNotIn('native_cache_index_sha256',loaded)
            from bevel_gear.native_pressure_domain import audit_native_partition_geometry
            _,_,proof=audit_native_partition_geometry(loaded,meshes)
            self.assertTrue(proof['passed'])

    def test_finite_pose_bad_mean_and_hidden_vertex_penetration_rejected(self):
        base,compact,frames=fixture();system=build_native_operator_at_pose(base,compact,frames,.04)
        u={m:np.zeros((w.shape[0]//3,3)) for m,w in system['matrices'].items()}
        wrong=np.asarray([r['gap_m'] for r in system['records']])+.0001
        proof=check_native_operator_pose(system,compact,u,0.,wrong)
        self.assertFalse(proof['passed']);self.assertGreater(proof['initial_piece_vertex_check']['maximum_mean_vs_QP_error_m'],1e-5)
        u['wheel'][:,2]=-.1;remaining=np.asarray([r['gap_m'] for r in system['records']])+np.asarray(system['matrices']['wheel'].T@u['wheel'].ravel()).ravel()
        proof=check_native_operator_pose(system,compact,u,0.,remaining)
        self.assertFalse(proof['passed']);self.assertLess(proof['loaded_FE_phi_geometry_check']['minimum_remaining_gap_m'],-1e-3)

    def test_archive_paths_reject_parent_absolute_and_windows_drive(self):
        base,compact,frames=fixture();system=build_native_operator_at_pose(base,compact,frames,.04)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'operator';save_native_operator(system,path)
            file=path/'manifest.json';original=json.loads(file.read_text(encoding='utf8'))
            for name in ('../external.json','C:/external.json',str(file.resolve())):
                data=deepcopy(original);data['geometry']['file']=name;file.write_text(json.dumps(data),encoding='utf8')
                with self.assertRaisesRegex(ValueError,'路径'):load_native_operator(path)

    def test_pose_never_accepts_partial_original_requests_after_mask_shrinks(self):
        base,compact,frames=fixture();patch=base['geometry']['patches'][0]
        patch['grid']=RectangularContactGrid(1,1,.5,.5,(.95,.95));patch['source_valid_mask']=patch['valid_mask']=np.ones((1,1),bool)
        system=build_native_operator_at_pose(base,compact,frames,.04)
        self.assertFalse(system['geometry']['patches'][0]['valid_mask'].any())
        u={m:np.zeros((w.shape[0]//3,3)) for m,w in system['matrices'].items()}
        proof=check_native_operator_pose(system,compact,u,0.,[])
        self.assertTrue(proof['completed'],proof.get('failure'));self.assertFalse(proof['passed'])
        self.assertEqual(proof['initial_piece_vertex_check']['partial_cells'],1)
        self.assertEqual(proof['loaded_FE_phi_geometry_check']['partial_cells'],1)

    def test_cold_import_needs_no_numeric_cad_fe_or_cache_runtime(self):
        root=Path(__file__).resolve().parents[1]
        code="import sys; import bevel_gear.native_fe_operator; import bevel_gear.projected_fe_partition_arrays; import bevel_gear.projected_fe_body_arrays; assert not any(n in sys.modules for n in ('numpy','scipy','shapely','OCP','gmsh'))"
        result=subprocess.run([sys.executable,'-S','-B','-c',code],cwd=root,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode('utf8',errors='replace'))

    def test_bad_plane_rotation_indices_mask_and_batch_fail_explicitly(self):
        base,compact,frames=fixture();source=base['geometry']['patches'][0]
        for bad in (0,True,1.2):
            with self.assertRaises(ValueError):build_native_operator_at_pose(base,compact,frames,.04,batch_cells=bad)
        wrong=deepcopy(base);wrong['geometry']['patches'][0]['plane_tangent_rows']=[[2,0,0],[0,1,0]]
        with self.assertRaisesRegex(ValueError,'正交'):build_native_operator_at_pose(wrong,compact,frames,.04)
        wrong=deepcopy(frames);wrong['wheel_rotation_rows']=np.ones((3,3))
        with self.assertRaisesRegex(ValueError,'旋转'):build_native_operator_at_pose(base,compact,wrong,.04)
        with self.assertRaises(ValueError):compact_fe_boundary({'nodes_m':[[0,0,0]],'boundary_triangles':[[0,1,2]]},'wheel')
        with self.assertRaisesRegex(ValueError,'布尔'):build_native_fe_contact_operator(compact,[dict(tooth_offset=0,grid=source['grid'],plane=PLANE,requested_mask=np.ones(source['grid'].shape))],frames,.04)

    def test_axis_origin_translation_and_pinion_pose_are_preserved(self):
        _,_,frames=fixture();frames.update(wheel_translation_world_m=[.1,0,.2],wheel_axis_origin_world_m=[.03,0,.04])
        delta=.03;new=advance_native_follower_frame(frames,delta);r=rotation([0,1,0],delta)
        np.testing.assert_array_equal(new['pinion_rotation_rows'],frames['pinion_rotation_rows'])
        np.testing.assert_allclose(new['wheel_translation_world_m'],np.array([.03,0,.04])+r@(np.array([.1,0,.2])-np.array([.03,0,.04])),rtol=0,atol=1e-16)


if __name__=='__main__':unittest.main(verbosity=2)
