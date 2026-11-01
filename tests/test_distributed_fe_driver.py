"""分布FE示例的业务编排回归；全部重计算用mock替代，不生成大型体网格。"""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


@dataclass
class Request:
    tooth_count:int
    control_points:tuple=(1.,2.)


@dataclass
class Blank:
    bore_diameter_m:float
    section:tuple=(.1,.2)


def accepted_result():
    return dict(converged=True,status='mock_accepted',geometry_linearization_checked=True,
        equilibrium=dict(certificate=dict(passed=True)),pinion_torque_relative_residual=.001,
        maximum_pressure_pa=1e9,elastic_te_rad=-1e-4,minimum_gauss_remaining_gap_m=0.)


class DistributedFeDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'output';self.mesh_dir=self.root/'meshes'
        project=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('distributed_driver_under_test',project/'examples/run_distributed_fe_contact.py')
        self.driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.driver)
        self.models={m:SimpleNamespace(request=Request(z),settings=Blank(bore))
            for m,z,bore in [('pinion',17,.012),('wheel',47,.032)]}
        base=dict(schema='bevel_contact_manufacturing_case_v1',
            manufacturing={m:dict(request=dict(tooth_count=model.request.tooth_count),blank=dict(bore_diameter_m=model.settings.bore_diameter_m)) for m,model in self.models.items()},
            materials={m:dict(young_modulus_pa=210e9,poisson_ratio=.3) for m in self.models},
            load=dict(ltca_torque_sign=-1,pinion_torque_n_m=64.),cad_sampling=dict(width_count=5,profile_count=5))
        self.case=dict(schema='bevel_distributed_fe_case_v1',base_case='base.json',
            analysis=dict(driver_angles_rad=[-.1,0.,.1],tooth_offsets=[-1,0,1],grid_shape=[3,3],spacing_m=[.001,.001],ray_backend='batch'),
            mesh=dict(element_size_m=.002,maximum_nodes=100,refinement_phase_intervals=2),
            solver=dict(maximum_outer_iterations=2),acceptance=dict(maximum_pinion_torque_relative_residual=.02))
        (self.root/'base.json').write_text(json.dumps(base),encoding='utf-8');self.case_file=self.root/'case.json';self.write_case()

    def write_case(self):
        self.case_file.write_text(json.dumps(self.case),encoding='utf-8')

    def mesh(self,member):
        from bevel_gear.elasticity import TetraMesh
        return TetraMesh(nodes_m=[[0.,0.,0.],[.001,0.,0.],[0.,.001,0.],[0.,0.,.001]],
            tetrahedra=[[0,1,2,3]],metadata=dict(units='m',
                actual_manufacturing_request=asdict(self.models[member].request),
                actual_blank_settings=asdict(self.models[member].settings)))

    def write_reused_meshes(self):
        self.mesh_dir.mkdir()
        for member in self.models:
            (self.mesh_dir/(member+'_mesh.json')).write_text(json.dumps(asdict(self.mesh(member))),encoding='utf-8')

    def run_driver(self,*,reuse=False,selected=None,result=None,solve_effect=None,build_error=None,identity=None):
        def build(request,blank,**sampling):
            if build_error:raise build_error
            return self.models['pinion' if request['tooth_count']==17 else 'wheel']
        def prepare(*args,**kwargs):
            meshes=kwargs['meshes'] or {m:self.mesh(m) for m in self.models}
            return SimpleNamespace(systems={m:SimpleNamespace(mesh=mesh) for m,mesh in meshes.items()},diagnostics=dict(mock_only=True))
        def tracks(*args,**kwargs):
            return dict(tracks={m:[[[0.,0.,0.],[.001,0.,0.]]] for m in self.models})
        def geometry(prepared,angle,torque,**kwargs):
            return dict(driving_angle_rad=angle,pinion_torque_nm=torque,build_parameters=kwargs)
        with ExitStack() as stack,redirect_stdout(io.StringIO()):
            mocks={}
            for name,value in [('bevel_gear.generated_solid.build_generated_gear',build),
                    ('bevel_gear.generated_tca.prepare_generated_tca',lambda *a,**k:object()),
                    ('bevel_gear.generated_nodal_ltca.generated_refinement_tracks',tracks),
                    ('bevel_gear.generated_ltca.prepare_generated_ltca',prepare),
                    ('bevel_gear.generated_halfspace.prepare_generated_halfspace',lambda *a,**k:object()),
                    ('bevel_gear.generated_halfspace.build_generated_halfspace_patches',geometry)]:
                mocks[name.rsplit('.',1)[-1]]=stack.enter_context(patch(name,side_effect=value))
            solve=stack.enter_context(patch('bevel_gear.generated_distributed_fe.solve_generated_distributed_fe_ltca',
                side_effect=solve_effect,return_value=accepted_result() if result is None else result))
            stack.enter_context(patch.object(self.driver,'_identity',side_effect=identity or (lambda:dict(kernel='stable'))))
            argv=[str(self.case_file),'--output',str(self.output)]
            if reuse:argv+=['--mesh-dir',str(self.mesh_dir)]
            if selected is not None:
                for index in selected:argv+=['--phase-index',str(index)]
            code=self.driver.main(argv)
            return code,mocks,solve

    def summary(self):
        return json.loads((self.output/'distributed_fe_summary.json').read_text(encoding='utf-8'))

    def test_cold_start_tracks_are_member_to_polylines_and_batch_is_forwarded(self):
        code,mocks,solve=self.run_driver(selected=[1]);self.assertEqual(code,0)
        call=mocks['prepare_generated_ltca'].call_args.kwargs
        self.assertIsNone(call['meshes']);self.assertEqual(set(call['refinement_polylines_m']),{'pinion','wheel'})
        for lines in call['refinement_polylines_m'].values():
            self.assertEqual(len(lines),3)
            self.assertTrue(all(len(line)==2 and all(len(point)==3 for point in line) for line in lines))
        self.assertEqual(call['element_size_m'],.002);self.assertEqual(call['width_count'],5)
        self.assertEqual(mocks['build_generated_halfspace_patches'].call_args.kwargs['ray_backend'],'batch')
        self.assertEqual(solve.call_args.kwargs['ray_backend'],'batch')
        self.assertEqual(solve.call_args.kwargs['initial_geometry']['driving_angle_rad'],0.)
        self.assertTrue(self.summary()['analysis_completed'])

    def test_reused_json_lists_match_dataclass_tuples_and_source_mesh_is_not_overwritten(self):
        self.write_reused_meshes();before={m:(self.mesh_dir/(m+'_mesh.json')).read_bytes() for m in self.models}
        code,mocks,_=self.run_driver(reuse=True,selected=[1]);self.assertEqual(code,0)
        self.assertIsNone(mocks['prepare_generated_ltca'].call_args.kwargs['refinement_polylines_m'])
        self.assertIsNotNone(mocks['prepare_generated_ltca'].call_args.kwargs['meshes'])
        for member,content in before.items():self.assertEqual((self.mesh_dir/(member+'_mesh.json')).read_bytes(),content)

    def test_actual_saved_mesh_hashes_are_bound_into_final_state_input_fingerprint(self):
        from bevel_gear.surface_workflow import _fingerprint
        self.run_driver(selected=[1]);summary=self.summary();inputs=json.loads((self.output/'distributed_fe_inputs.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['input_fingerprint'],_fingerprint(inputs))
        for member,name in summary['mesh_files'].items():
            actual=hashlib.sha256((self.output/name).read_bytes()).hexdigest()
            self.assertEqual(summary['mesh_sha256'][member],actual)
            self.assertEqual(inputs['mesh_sha256'][member],actual)
        state=json.loads((self.output/'states/phase_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(state['input_fingerprint'],summary['input_fingerprint'])

    def test_mesh_request_mismatch_is_rejected_before_any_contact_solve(self):
        self.write_reused_meshes();path=self.mesh_dir/'pinion_mesh.json';mesh=json.loads(path.read_text(encoding='utf-8'));mesh['metadata']['actual_manufacturing_request']['tooth_count']=16;path.write_text(json.dumps(mesh),encoding='utf-8')
        with self.assertRaises(ValueError):self.run_driver(reuse=True,selected=[1])
        summary=self.summary();self.assertFalse(summary['analysis_completed']);self.assertEqual(summary['states'],[])
        self.assertEqual(summary['failure_stage'],'preparation');self.assertTrue(summary['source_identity_unchanged'])

    def test_hard_phase_failure_preserves_prior_state_and_records_failure_without_fake_pressure(self):
        code,_,solve=self.run_driver(solve_effect=[accepted_result(),ArithmeticError('独立mock数值失败'),accepted_result()])
        self.assertEqual(code,1);self.assertEqual(solve.call_count,2)
        summary=self.summary();self.assertFalse(summary['analysis_completed']);self.assertEqual(len(summary['states']),2)
        self.assertTrue(summary['states'][0]['accepted']);self.assertFalse(summary['states'][1]['accepted'])
        state=json.loads((self.output/'states/phase_0001.json').read_text(encoding='utf-8'))
        self.assertEqual(state['result']['failure']['exception_type'],'ArithmeticError')
        self.assertNotIn('pressure_pa',state['result']);self.assertFalse((self.output/'states/phase_0002.json').exists())

    def test_unaccepted_soft_phase_remains_in_summary_and_other_phase_is_not_filtered(self):
        failed=dict(converged=False,status='not_converged')
        code,_,_=self.run_driver(selected=[0,1],solve_effect=[failed,accepted_result()])
        self.assertEqual(code,1);summary=self.summary()
        self.assertEqual([r['accepted'] for r in summary['states']],[False,True])
        self.assertIsNone(summary['states'][0]['maximum_pressure_pa'])

    def test_source_change_records_rejected_original_state(self):
        changed=[False]
        def solve(*args,**kwargs):changed[0]=True;return accepted_result()
        with self.assertRaises(RuntimeError):self.run_driver(selected=[1],solve_effect=solve,identity=lambda:dict(kernel='after' if changed[0] else 'before'))
        summary=self.summary();self.assertFalse(summary['source_identity_unchanged']);self.assertFalse(summary['states'][0]['accepted'])
        state=json.loads((self.output/'states/phase_0001.json').read_text(encoding='utf-8'))
        self.assertTrue(state['result']['converged']);self.assertFalse(state['accepted'])

    def test_no_success_from_boolean_residual_or_missing_kkt_certificate(self):
        for field,value in [('pinion_torque_relative_residual',False),('equilibrium',{})]:
            with self.subTest(field=field):
                self.output=self.root/field;result=accepted_result();result[field]=value
                code,_,_=self.run_driver(selected=[1],result=result)
                self.assertEqual(code,1);self.assertFalse(self.summary()['states'][0]['accepted'])

    def test_preparation_exception_keeps_incomplete_summary_and_current_source_identity(self):
        with self.assertRaises(ArithmeticError):self.run_driver(build_error=ArithmeticError('mock CAD失败'))
        summary=self.summary();self.assertFalse(summary['analysis_completed']);self.assertEqual(summary['failure_stage'],'preparation')
        self.assertTrue(summary['source_identity_unchanged']);self.assertEqual(summary['source_identity_before'],summary['source_identity_after'])

    def test_existing_workflow_records_cannot_be_overwritten(self):
        self.run_driver(selected=[1]);before=(self.output/'distributed_fe_summary.json').read_bytes()
        with self.assertRaises(FileExistsError):self.run_driver(selected=[1])
        self.assertEqual((self.output/'distributed_fe_summary.json').read_bytes(),before)

    def test_invalid_phase_and_duplicate_solver_binding_fail_before_output_preparation(self):
        with self.assertRaises(ValueError):self.run_driver(selected=[9])
        self.assertFalse((self.output/'distributed_fe_inputs.json').exists())
        self.case['solver']['ray_backend']='scalar';self.write_case()
        with self.assertRaises(ValueError):self.run_driver(selected=[1])
        self.assertFalse((self.output/'distributed_fe_inputs.json').exists())

    def test_source_identity_covers_contact_qp_track_producer_and_driver(self):
        identity=self.driver._identity()
        for key in ('contact_equilibrium','generated_nodal_ltca','driver','generated_distributed_fe'):
            self.assertIn(key,identity);self.assertEqual(len(identity[key]),64)


if __name__=='__main__':unittest.main()
