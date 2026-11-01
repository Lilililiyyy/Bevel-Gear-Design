"""生产outer接线：真实小mesh/typed几何配合显式mock QP，不认证齿轮压力。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from bevel_gear import native_fe_operator as operator
from bevel_gear.native_fe_contact import solve_native_fe_contact
from bevel_gear.native_pressure_domain import audit_native_partition_geometry, native_result_identity


class NativeContactTests(unittest.TestCase):
    def fixture(self, *, reference=0., origin=(-.00025, -.00025)):
        """两侧真实四节点体在平面接触；原QP由受控回调替代。"""
        faces=np.array([[0,1,2],[0,3,1],[1,3,2],[2,3,0]])
        plane=np.array([[-.002,-.002,0.],[.002,-.002,0.],[0.,.002,0.]])
        meshes={};compact={}
        for member,side in (('pinion',-1.),('wheel',1.)):
            nodes=np.vstack((plane,[0.,0.,side*.002]))
            meshes[member]=SimpleNamespace(nodes_m=nodes,tetrahedra=np.array([[0,1,2,3]]),
                boundary_triangles=faces,boundary_tags=np.zeros(4,dtype=int))
            compact[member]={0:dict(name=member,nodes_m=plane.copy(),triangles=np.array([[0,1,2]]),
                global_node_ids=np.array([0,1,2]),total_node_count=4)}
        frames=dict(pinion_rotation_rows=np.eye(3),wheel_rotation_rows=np.eye(3),pinion_axis_world=[0.,1.,0.],
            wheel_axis_world=[0.,1.,0.],translation_m=[0.,0.,0.])
        frames=operator.advance_native_follower_frame(frames,reference)
        template=dict(frames=frames,driving_angle_rad=.1,reference_follower_angle_rad=reference,
            pinion_torque_nm=-1.,pinion_to_wheel_ratio=.5,
            patches=[dict(tooth_offset=0,grid=dict(x_count=2,y_count=2,spacing_x_m=.0005,
                spacing_y_m=.0005,origin_xy_m=origin),plane=dict(origin_world_m=[0.,0.,0.],
                tangent_rows=[[1.,0.,0.],[0.,1.,0.]],normal_world=[0.,0.,1.]),requested_mask=np.ones((2,2),dtype=bool),
                source_material_reference_frames=deepcopy(frames))])
        prepared=SimpleNamespace(systems={m:SimpleNamespace(mesh=mesh) for m,mesh in meshes.items()},
            models={m:SimpleNamespace(request=SimpleNamespace(tooth_count=z)) for m,z in (('pinion',17),('wheel',34))})
        return prepared,template,compact,meshes

    def qp(self, calls, *, target=None, boundary=False):
        def solve(prepared,system,torque,**options):
            phi=system['geometry']['reference_follower_angle_rad']
            step=0. if target is None else (math.sin(target)-math.sin(phi))/math.cos(phi)
            calls.append(dict(reference=phi,step=step,warm=options['initial_indices']))
            count=len(system['records']);forces=np.ones(count);fields=[];cursor=0
            for part in system['geometry']['patches']:
                mask=np.asarray(part['valid_mask']);n=int(mask.sum())
                if not n:continue
                force=np.zeros(mask.shape);force[mask]=forces[cursor:cursor+n];cursor+=n
                fields.append(dict(tooth_offset=part['tooth_offset'],forces_n=force,valid_mask=mask,
                    pressure_pa=force/part['grid'].cell_area_m2,elastic_deflection_m=np.zeros(mask.shape)))
            diagnostics=dict(strain_energy_j=0.,external_work_j=0.,maximum_free_residual_n=0.,relative_free_residual=0.)
            body={m:dict(displacements_m=np.zeros_like(body.mesh.nodes_m),
                nodal_forces_n=(system['matrices'][m]@forces).reshape(-1,3),**diagnostics) for m,body in prepared.systems.items()}
            return dict(converged=True,status='显式mock QP：仅验证生产编排',
                equilibrium=dict(theta_rad=step,forces_n=forces,remaining_gaps_m=[r['gap_m']-step*r['levers_m']['wheel'] for r in system['records']],
                    strain_energy_j=0.,certificate=dict(passed=True)),elastic_te_rad=step,
                body_fields=body,body_response_checks={m:dict(diagnostics) for m in body},
                active_cell_keys=[(0,0,0)],fields=fields,selected_tooth_offsets=[f['tooth_offset'] for f in fields],
                requested_tooth_offsets=[p['tooth_offset'] for p in system['geometry']['patches']],cells=[],
                contact_representation='distributed_P0_surface_cells',pressure_filter_applied=False,halfspace_compliance_used=False,
                quadrature_order='exact_affine_partition_centroids',pinion_torque_relative_residual=0.,fe_strain_energy_j=0.,
                contact_boundary_checks=[dict(touches_window_edge=boundary,touches_invalid_candidate_neighbor=False)])
        return solve

    def test_actual_flat_typed_pose_no_cache_identity_and_actual_mesh_binding(self):
        prepared,template,compact,meshes=self.fixture();calls=[]
        with patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=self.qp(calls)):
            answer=solve_native_fe_contact(prepared,template,compact_facets=compact)
        self.assertTrue(answer['accepted'],answer['rejection_reasons']);self.assertEqual(len(calls),1)
        self.assertTrue(answer['geometry_check']['passed']);self.assertNotIn('native_cache_index_sha256',answer['result'])
        self.assertNotIn('body_fields',answer['result']);self.assertIn('distributed_body_identity',answer['result'])
        native_result_identity(answer['result'])
        _,_,proof=audit_native_partition_geometry(answer['system'],meshes,result=answer['result'])
        self.assertTrue(proof['passed']);self.assertFalse(proof['NURBS_gap_queried'])
        self.assertEqual(proof['operator_input_fingerprint'],answer['system']['operator_input_fingerprint'])
        altered=deepcopy(answer['result']);altered['actual_native_mesh_numeric_identity']['pinion']['nodes_m']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'数字身份'):
            audit_native_partition_geometry(answer['system'],meshes,result=altered)

    def test_smooth_outer_fresh_calls_total_TE_and_same_phase_warm_only(self):
        prepared,template,compact,_=self.fixture(reference=.05);calls=[]
        checker=lambda *a,**k:dict(passed=True,fixture_only=True)
        with (patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=self.qp(calls,target=.2)),
                patch.object(operator,'check_native_operator_pose',side_effect=checker)):
            answer=solve_native_fe_contact(prepared,template,compact_facets=compact,include_body_fields=True)
            second=solve_native_fe_contact(prepared,template,compact_facets=compact,include_body_fields=True)
        self.assertTrue(answer['accepted'],answer['rejection_reasons']);n=len(answer['geometry_outer_history'])
        self.assertGreater(n,1);self.assertIsNone(calls[0]['warm']);self.assertIsNone(calls[n]['warm'])
        self.assertTrue(all(c['warm']==[0] for c in calls[1:n]))
        self.assertLess(abs(answer['result']['linearization_step_theta_rad']),1e-8)
        self.assertAlmostEqual(answer['result']['loaded_follower_angle_rad'],.2,places=12)
        self.assertAlmostEqual(answer['result']['total_loaded_te_rad'],.25,places=12)
        self.assertAlmostEqual(answer['result']['cumulative_loaded_follower_shift_rad'],.15,places=12)
        self.assertNotIn('elastic_te_rad',answer['result']);self.assertFalse(answer['result']['native_unloaded_mesh_tangency_solved'])
        self.assertIn('body_fields',answer['result']);self.assertEqual(len(prepared.systems),2)

    def test_maximum_outer_and_boundary_failures_are_not_accepted(self):
        prepared,template,compact,_=self.fixture(reference=.05)
        with patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=self.qp([],target=.2)):
            value=solve_native_fe_contact(prepared,template,compact_facets=compact,maximum_outer_iterations=1)
        self.assertFalse(value['accepted']);self.assertIsNone(value['geometry_check'])
        self.assertIn('最多外迭代',value['rejection_reasons'][0])
        prepared,template,compact,_=self.fixture()
        with patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=self.qp([],boundary=True)):
            value=solve_native_fe_contact(prepared,template,compact_facets=compact)
        self.assertFalse(value['accepted']);self.assertTrue(value['geometry_check']['passed'])

    def test_partial_source_window_is_rejected_before_QP(self):
        prepared,template,compact,_=self.fixture(origin=(.01,0.))
        with patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system') as solve:
            answer=solve_native_fe_contact(prepared,template,compact_facets=compact)
        self.assertFalse(answer['accepted']);self.assertIsNone(answer['result']);solve.assert_not_called()

    def test_updated_partial_operator_cannot_inherit_previous_u_or_pressure(self):
        prepared,template,compact,_=self.fixture(reference=.05);calls=[]
        original=operator.build_native_operator_at_pose
        def build(*args,**kwargs):
            value=original(*args,**kwargs)
            value['geometry']['patches'][0]['valid_mask'][0,0]=False
            value['geometry']['patches'][0]['native_partial_cell_mask'][0,0]=True
            value['typed_patch_arrays'][0]['partial_cell_count']=1
            return value
        # 首次高层build也调用低层；仅第2个低层调用制造明确partial。
        count=[0]
        def controlled(*args,**kwargs):
            count[0]+=1
            return original(*args,**kwargs) if count[0]==1 else build(*args,**kwargs)
        with (patch.object(operator,'build_native_operator_at_pose',side_effect=controlled),
                patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=self.qp(calls,target=.2))):
            value=solve_native_fe_contact(prepared,template,compact_facets=compact)
        self.assertFalse(value['accepted']);self.assertIsNone(value['result']);self.assertEqual(len(calls),1)
        self.assertEqual(len(value['geometry_outer_history']),1)

    def test_foreign_compact_and_relaxed_limits_are_rejected(self):
        prepared,template,compact,_=self.fixture();compact['wheel'][0]['nodes_m'][0,2]+=1e-9
        with self.assertRaisesRegex(ValueError,'完整FE网格'):
            solve_native_fe_contact(prepared,template,compact_facets=compact)
        # 显式点间隙合法上限扩展到1μm；默认仍0.2μm，超过新上限须拒绝。
        for options in (dict(maximum_outer_iterations=7),dict(outer_step_tolerance_rad=2e-8),
                dict(mean_geometry_tolerance_m=2e-10),dict(maximum_piece_penetration_m=1.01e-6)):
            with self.assertRaises(ValueError):solve_native_fe_contact(prepared,template,compact_facets=compact,**options)

    def test_explicit_point_penetration_range_preserves_default_and_stops_before_solver(self):
        """只测显式参数合法范围；通过哨兵停在compact校验前，不创建K或mock压力。"""
        from inspect import signature
        self.assertEqual(signature(solve_native_fe_contact).parameters['maximum_piece_penetration_m'].default,2e-7)
        prepared,template,compact,_=self.fixture()
        class ReachedCompactValidation(Exception):pass
        with patch('bevel_gear.native_fe_contact._validate_compacts',side_effect=ReachedCompactValidation) as reached:
            for options in ({},dict(maximum_piece_penetration_m=1e-6)):
                with self.assertRaises(ReachedCompactValidation):
                    solve_native_fe_contact(prepared,template,compact_facets=compact,**options)
            self.assertEqual(reached.call_count,2)
            with self.assertRaisesRegex(ValueError,'maximum_piece_penetration_m'):
                solve_native_fe_contact(prepared,template,compact_facets=compact,maximum_piece_penetration_m=1.000001e-6)
            self.assertEqual(reached.call_count,2)

    def test_ratio_must_match_actual_prepared_teeth_before_QP(self):
        prepared,template,compact,_=self.fixture();template['pinion_to_wheel_ratio']=17/47
        with (patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system') as solve,
                self.assertRaisesRegex(ValueError,'实际z1/z2')):
            solve_native_fe_contact(prepared,template,compact_facets=compact)
        solve.assert_not_called()

    def test_optional_extraction_and_stale_load_fields_are_not_input(self):
        prepared,template,compact,_=self.fixture();calls=[]
        template['old_force_n']=[999.];template['patches'][0]['gaps_m']=[[999.]]
        with (patch('bevel_gear.native_fe_preparation.extract_native_flank_compacts',return_value=(compact,dict(passed=True))) as extract,
                patch('bevel_gear.generated_distributed_fe.solve_distributed_contact_system',side_effect=self.qp(calls))):
            answer=solve_native_fe_contact(prepared,template)
        self.assertTrue(answer['accepted']);extract.assert_called_once()
        self.assertIsNone(calls[0]['warm']);self.assertNotIn('old_force_n',answer['system']['geometry'])
        self.assertAlmostEqual(float(np.nanmax(answer['system']['geometry']['patches'][0]['gaps_m'])),0.)

    def test_import_without_numeric_or_CAD_packages(self):
        code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).resolve().parents[1]))+');import bevel_gear.native_fe_contact'
        result=subprocess.run([sys.executable,'-S','-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
