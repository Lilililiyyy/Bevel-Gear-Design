"""公开复算入口的主轴选择/纯SI区域文件：全部重计算mock，不启动大FE。"""
from dataclasses import asdict
from copy import deepcopy
import hashlib,json
import unittest
from unittest.mock import patch
import test_distributed_fe_driver as support


class DistributedFeDriverRegionTests(unittest.TestCase):
    def setUp(self):
        self.h=support.DistributedFeDriverTests('runTest');self.h.setUp();self.addCleanup(self.h.doCleanups)

    def region_file(self):
        h=self.h;region=dict(refinement_polylines_m=[[[.001,0.,0.],[.002,0.,0.]]],
            refined_size_m=.000075,refinement_core_radius_m=.0003,refinement_radius_m=.0012)
        self.document=dict(schema='bevel_refinement_regions_v1',units='m',coordinate_system='gear_body_local',
            source_driver_angles_rad=[0.],manufacturing={m:dict(request=asdict(model.request),blank=asdict(model.settings)) for m,model in h.models.items()},
            regions={m:[deepcopy(region)] for m in h.models})
        self.path=h.root/'controls/regions.json';self.path.parent.mkdir()
        self.write_region();h.case['mesh']['refinement_regions_file']='controls/regions.json';h.write_case()

    def write_region(self):
        self.path.write_text(json.dumps(self.document,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')

    def test_gap_hessian_selector_is_explicit_and_initial_direction_evidence_is_saved(self):
        h=self.h;h.case['analysis'].update(plane_alignment='gap_hessian',alignment_step_m=.00005,
            alignment_fallback_to_surface_basis=True);h.write_case()
        evidence=dict(method='actual_gap_hessian_principal_axes',directions=[dict(tooth_offset=-2,accepted=False,angle_rad=0.,reason='有限面stencil失败')])
        geometry=dict(driving_angle_rad=0.,pinion_torque_nm=-64.,grid_alignment=evidence,build_parameters=dict(plane_basis_angles_rad={0:-.2}))
        with patch('bevel_gear.contact_plane.build_principal_aligned_patches',return_value=geometry) as aligned:
            code,mocks,solve=h.run_driver(selected=[1])
        self.assertEqual(code,0);self.assertEqual(aligned.call_args.kwargs['step_m'],.00005)
        self.assertTrue(aligned.call_args.kwargs['fallback_to_surface_basis'])
        self.assertNotIn('plane_alignment',aligned.call_args.kwargs)
        self.assertEqual(mocks['build_generated_halfspace_patches'].call_count,0)
        self.assertEqual(solve.call_args.kwargs['initial_geometry'],geometry)
        state=json.loads((h.output/'states/phase_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(state['grid_alignment'],evidence)
        self.assertIn('bevel_gear.contact_plane',state['actual_imports'])

    def test_unknown_alignment_and_conflicting_explicit_angles_are_rejected(self):
        h=self.h;h.case['analysis']['plane_alignment']='pressure_fit';h.write_case()
        with self.assertRaises(ValueError):h.run_driver(selected=[1])
        self.assertFalse(h.output.exists())
        h.case['analysis'].update(plane_alignment='gap_hessian',plane_basis_angles_rad={'0':-.2});h.write_case()
        with self.assertRaises(ValueError):h.run_driver(selected=[1])

    def test_pure_si_file_original_bytes_and_expanded_controls_bind_fingerprint(self):
        h=self.h;self.region_file();raw=self.path.read_bytes();code,_,prepare=h.run_driver(selected=[1])
        self.assertEqual(code,0)
        inputs=json.loads((h.output/'distributed_fe_inputs.json').read_text(encoding='utf-8'))
        self.assertEqual(inputs['refinement_regions_document']['regions'],self.document['regions'])
        self.assertEqual(inputs['refinement_regions_source']['sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual((h.output/'refinement_regions_input.json').read_bytes(),raw)
        summary=h.summary();self.assertFalse(summary['refinement_regions_scope']['coverage_or_convergence_certified'])
        self.assertEqual(summary['refinement_regions_scope']['source_driver_angles_rad'],[0.])

    def test_cold_mesh_preserves_background_and_forwards_each_member_region(self):
        h=self.h;self.region_file();code,mocks,_=h.run_driver(selected=[1])
        self.assertEqual(code,0);kwargs=mocks['prepare_generated_ltca'].call_args.kwargs
        self.assertIsNotNone(kwargs['refinement_polylines_m'])
        self.assertEqual(kwargs['refinement_regions'],self.document['regions'])

    def test_declared_single_phase_cannot_be_used_for_other_selected_phases(self):
        h=self.h;self.region_file()
        with self.assertRaisesRegex(ValueError,'单相位'):h.run_driver()
        self.assertFalse(h.output.exists())

    def test_manufacturing_identity_mismatch_is_rejected_before_contact_solve(self):
        h=self.h;self.region_file();self.document['manufacturing']['pinion']['request']['tooth_count']=16;self.write_region()
        with self.assertRaisesRegex(ValueError,'request/blank'):h.run_driver(selected=[1])
        self.assertEqual(h.summary()['failure_stage'],'preparation');self.assertEqual(h.summary()['states'],[])

    def test_non_si_or_nonfinite_controls_fail_before_cad_preparation(self):
        h=self.h;self.region_file();original=deepcopy(self.document)
        for key,value in [('units','mm'),('coordinate_system','assembly_world')]:
            self.document=deepcopy(original);self.document[key]=value;self.write_region()
            with self.assertRaises(ValueError):h.run_driver(selected=[1])
        self.document=deepcopy(original);self.document['regions']['pinion'][0]['refinement_polylines_m'][0][0][0]=float('nan');self.write_region()
        with self.assertRaises(ValueError):h.run_driver(selected=[1])
        self.assertFalse(h.output.exists())

    def test_reused_mesh_regions_must_match_and_are_not_applied_twice(self):
        h=self.h;self.region_file();h.write_reused_meshes()
        with self.assertRaisesRegex(ValueError,'已有加密区域'):h.run_driver(reuse=True,selected=[1])
        h.output=h.root/'matching_reused'
        for member in h.models:
            path=h.mesh_dir/(member+'_mesh.json');mesh=json.loads(path.read_text(encoding='utf-8'))
            mesh['metadata']['refinement_regions']=self.document['regions'][member];path.write_text(json.dumps(mesh),encoding='utf-8')
        code,mocks,_=h.run_driver(reuse=True,selected=[1]);self.assertEqual(code,0)
        self.assertNotIn('refinement_regions',mocks['prepare_generated_ltca'].call_args.kwargs)

    def test_later_source_file_edit_cannot_replace_the_copied_original_input(self):
        h=self.h;self.region_file();original=self.path.read_bytes()
        def solve(*args,**kwargs):
            self.path.write_text('{}',encoding='utf-8');return support.accepted_result()
        code,_,_=h.run_driver(selected=[1],solve_effect=solve);self.assertEqual(code,0)
        self.assertEqual((h.output/'refinement_regions_input.json').read_bytes(),original)
        inputs=json.loads((h.output/'distributed_fe_inputs.json').read_text(encoding='utf-8'))
        self.assertEqual(inputs['refinement_regions_source']['sha256'],hashlib.sha256(original).hexdigest())


if __name__=='__main__':unittest.main()
