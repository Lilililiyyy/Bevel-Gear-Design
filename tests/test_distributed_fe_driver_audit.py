"""公开FE业务入口的来源、原字节和暖启动回归；不启动真实大型FE。"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import test_distributed_fe_driver as _support
accepted_result=_support.accepted_result


class DistributedFeDriverAuditTests(unittest.TestCase):
    def setUp(self):
        self.h=_support.DistributedFeDriverTests('runTest');self.h.setUp();self.addCleanup(self.h.doCleanups)

    def test_q4_geometry_default_and_deep_core_are_forwarded(self):
        h=self.h;h.case['mesh'].update(contact_refined_size_m=.0001,refinement_core_radius_m=.0003,refinement_radius_m=.0012)
        h.write_case();code,mocks,solve=h.run_driver(selected=[1]);self.assertEqual(code,0)
        self.assertEqual(mocks['prepare_generated_ltca'].call_args.kwargs['refinement_core_radius_m'],.0003)
        self.assertEqual(solve.call_args.kwargs['quadrature_order'],4)
        self.assertEqual(solve.call_args.kwargs['maximum_geometry_error_m'],1e-10)
        inputs=json.loads((h.output/'distributed_fe_inputs.json').read_text(encoding='utf-8'))
        self.assertEqual(inputs['effective_solver']['quadrature_order'],4)

    def test_warm_start_uses_only_preceding_accepted_state_and_failed_phase_resets_it(self):
        h=self.h;first=accepted_result();first['active_cell_keys']=[[-1,3,4],[0,5,6]]
        h.case['analysis']['driver_angles_rad']=[-.1,0.,.1,.2];h.write_case()
        code,_,solve=h.run_driver(solve_effect=[first,dict(converged=False,status='not_accepted'),accepted_result(),accepted_result()])
        self.assertEqual(code,1);calls=solve.call_args_list
        self.assertNotIn('initial_total_te_rad',calls[0].kwargs)
        self.assertEqual(calls[1].kwargs['initial_total_te_rad'],first['elastic_te_rad'])
        self.assertEqual(calls[1].kwargs['initial_active_cell_keys'],first['active_cell_keys'])
        self.assertNotIn('initial_total_te_rad',calls[2].kwargs)
        self.assertEqual(calls[3].kwargs['initial_total_te_rad'],accepted_result()['elastic_te_rad'])

    def test_explicit_cold_start_disables_cross_phase_seeds(self):
        h=self.h;original=h.driver.main
        with patch.object(h.driver,'main',side_effect=lambda argv:original(argv+['--cold-start-each-phase'])):
            code,_,solve=h.run_driver()
        self.assertEqual(code,0)
        self.assertTrue(all('initial_total_te_rad' not in c.kwargs for c in solve.call_args_list))

    def test_original_mesh_and_each_raw_state_sha_are_recorded(self):
        h=self.h;h.write_reused_meshes()
        raw={m:hashlib.sha256((h.mesh_dir/(m+'_mesh.json')).read_bytes()).hexdigest() for m in h.models}
        h.run_driver(reuse=True,selected=[1]);summary=h.summary()
        for m in h.models:self.assertEqual(summary['reused_mesh_sources'][m]['sha256'],raw[m])
        row=summary['states'][0]
        self.assertEqual(row['state_sha256'],hashlib.sha256((h.output/row['state_file']).read_bytes()).hexdigest())
        state=json.loads((h.output/row['state_file']).read_text(encoding='utf-8'))
        self.assertTrue(state['strict_import_binding_verified']);self.assertTrue(state['actual_imports'])
        self.assertFalse(summary['result_artifact_contract']['external_tooth_coverage_certified'])

    def test_other_installed_package_module_is_rejected_before_preparation(self):
        h=self.h;wrong=SimpleNamespace(__file__=str(h.root/'outside.py'))
        with patch.dict(sys.modules,{'bevel_gear._foreign_test':wrong}):
            with self.assertRaisesRegex(RuntimeError,'不属于本项目'):h.driver.main([str(h.case_file),'--output',str(h.output)])
        self.assertFalse(h.output.exists())

    def test_late_foreign_import_preserves_computed_original_state_as_rejected(self):
        h=self.h;original=accepted_result()
        def solve(*args,**kwargs):
            sys.modules['bevel_gear._foreign_test']=SimpleNamespace(__file__=str(h.root/'outside.py'))
            return original
        with patch.dict(sys.modules,{}):code,_,_=h.run_driver(selected=[1],solve_effect=solve)
        self.assertEqual(code,1);summary=h.summary();self.assertFalse(summary['states'][0]['accepted'])
        self.assertEqual(summary['failure_stage'],'import_binding_phase_1')
        state=json.loads((h.output/'states/phase_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(state['result'],original);self.assertFalse(state['strict_import_binding_verified'])

    def test_nonfinite_or_boolean_te_cannot_be_accepted_or_used_as_warm_start(self):
        h=self.h;result=accepted_result();result['elastic_te_rad']=True
        code,_,_=h.run_driver(selected=[1],result=result)
        self.assertEqual(code,1);self.assertFalse(h.summary()['states'][0]['accepted'])


if __name__=='__main__':unittest.main()
