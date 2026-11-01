"""原生FE生产示例编排控制；全部CAD、体K、原生求解及档案以mock替代。"""
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import io
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

import test_distributed_fe_driver as support


def native_reply(accepted=True):
    result=support.accepted_result();result.pop('elastic_te_rad')
    result.update(total_loaded_te_rad=-.00012,geometry_source='FE_boundary',native_contact_accepted=accepted,body_fields={'original':'nodes'})
    return dict(accepted=accepted,result=result,system={'actual':'native-operator'},
        input_fingerprint='c'*64,geometry_outer_history=[{'iteration':0}],geometry_check={'passed':accepted},rejection_reasons=[] if accepted else ['mock拒绝'])


class NativeDriverTests(unittest.TestCase):
    def setUp(self):
        self.h=support.DistributedFeDriverTests('runTest');self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.h.case['geometry_source']='FE_boundary';self.h.write_case()

    def run_native(self,*,selected=(1,),replies=None,archive_effect=None,operator_effect=None,
            extraction_error=None,factor_count=0,close_failure=False,identity_effect=None):
        h=self.h
        def build(request,blank,**sampling):return h.models['pinion' if request['tooth_count']==17 else 'wheel']
        def prepare(*args,**kwargs):
            factory=kwargs.get('factorization_factory')
            if factory:
                for _ in range(factor_count):factory('original-K-mock')
            return SimpleNamespace(systems={m:SimpleNamespace(mesh=h.mesh(m)) for m in h.models},diagnostics={'mock_only':True},tca='prepared-tca')
        def geometry(surface,angle,torque,**kwargs):
            return dict(driving_angle_rad=angle,reference_follower_angle_rad=-angle*17/47+kwargs.get('follower_correction_rad',0.),
                pinion_torque_nm=torque,pinion_to_wheel_ratio=17/47,
                gaps_m=['forbidden-CAD-gap'],pressure_pa=['forbidden-old-pressure'],C=['forbidden-old-compliance'],
                patches=[dict(tooth_offset=0,grid={'mock_grid':True},valid_mask=[[True]],
                    plane_origin_world_m=[0,0,0],plane_tangent_rows=[[1,0,0],[0,1,0]],plane_outward_normal_world=[0,0,1],
                    gaps_m=['forbidden'],pressure_pa=['forbidden'],quadrature=['forbidden'],C=['forbidden'],
                    frames={'Rk':'source-reference'},source_material_reference_frames={'Rk':'explicit-reference'})])
        def operator(system,folder,**kwargs):
            if operator_effect:raise operator_effect
            folder.mkdir(parents=True)
            document=dict(schema='native_FE_typed_operator_manifest_v2',operator_fingerprint='b'*64,
                actual_mesh_sha256=kwargs['actual_mesh_sha256'])
            manifest=folder/'manifest.json';manifest.write_text(json.dumps(document),encoding='utf8')
            return dict(current_operator_manifest_file=str(manifest),
                current_operator_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),operator_fingerprint='b'*64)
        effects=iter(archive_effect) if isinstance(archive_effect,list) else None
        def response(prepared,system,result,folder,*,phase_index,input_fingerprint):
            effect=next(effects) if effects is not None else archive_effect
            if isinstance(effect,BaseException):raise effect
            if callable(effect):effect=effect(prepared,system,result,folder,
                phase_index=phase_index,input_fingerprint=input_fingerprint)
            if isinstance(effect,dict) and effect.get('passed') is not True:return effect
            target=folder/'responses'/f'phase_{phase_index:04d}';target.mkdir(parents=True)
            bodies={}
            for member in ('pinion','wheel'):
                body={}
                for file_key,sha_key,suffix in (('response_file','response_sha256','response'),('load_matrix_file','load_matrix_sha256','load_matrix')):
                    file=target/(member+'_'+suffix+'.npz')
                    file.write_bytes(b'explicit mock archive bytes')
                    body[file_key]=file.relative_to(folder).as_posix();body[sha_key]=hashlib.sha256(file.read_bytes()).hexdigest()
                body['load_matrix_shape']=[12,1];bodies[member]=body
            return dict(schema='bevel_distributed_FE_response_receipt_v1',passed=True,phase_index=phase_index,
                input_fingerprint=input_fingerprint,bodies=bodies,**({k:v for k,v in (effect or {}).items() if k!='passed'}))
        factors=[]
        def factor(*args,**kwargs):
            item=MagicMock();item.close.side_effect=RuntimeError('mock close失败') if close_failure and not factors else None
            factors.append(item);return item
        with ExitStack() as stack,redirect_stdout(io.StringIO()):
            mocks={}
            for name,effect in (
                ('bevel_gear.generated_solid.build_generated_gear',build),
                ('bevel_gear.generated_tca.prepare_generated_tca',lambda *a,**k:'tca'),
                ('bevel_gear.generated_nodal_ltca.generated_refinement_tracks',lambda *a,**k:{'tracks':{m:[[[0,0,0],[.001,0,0]]] for m in h.models}}),
                ('bevel_gear.generated_ltca.prepare_generated_ltca',prepare),
                ('bevel_gear.generated_halfspace.prepare_generated_halfspace',lambda *a,**k:'surface'),
                ('bevel_gear.generated_halfspace.build_generated_halfspace_patches',geometry),
                ('bevel_gear.generated_tca.generated_tca_frames',lambda *a,**k:{'whole_body_R0':'native-initial'}),
                ('bevel_gear.native_fe_preparation.extract_native_flank_compacts',extraction_error or (lambda *a,**k:({'pinion':'actual-compact','wheel':'actual-compact'},{'passed':True}))),
                ('bevel_gear.native_fe_operator.save_native_operator',operator),
                ('bevel_gear.sparse_spd.PardisoSPD',factor)):
                mocks[name.rsplit('.',1)[-1]]=stack.enter_context(patch(name,side_effect=effect))
            solve=stack.enter_context(patch('bevel_gear.native_fe_contact.solve_native_fe_contact',autospec=True,
                side_effect=(lambda *a,**k:native_reply()) if replies is None else replies))
            archive=stack.enter_context(patch('bevel_gear.native_fe_archive.save_native_fe_response',
                side_effect=response))
            oldsolve=stack.enter_context(patch('bevel_gear.generated_distributed_fe.solve_generated_distributed_fe_ltca',side_effect=AssertionError('native分支不能调用旧CAD求解入口')))
            stack.enter_context(patch.object(h.driver,'_identity',side_effect=identity_effect or (lambda:{'kernel':'stable'})))
            argv=[str(h.case_file),'--output',str(h.output)]
            if selected is not None:
                for i in selected:argv+=['--phase-index',str(i)]
            if factor_count:
                library=h.root/'mock_mkl_runtime.dll';library.write_bytes(b'not-a-runtime; calls are mocked')
                argv+=['--sparse-solver','pardiso','--mkl-runtime',str(library)]
            try:code=h.driver.main(argv)
            finally:self.last=(mocks,solve,archive,oldsolve,factors)
        return code,mocks,solve,archive,oldsolve,factors

    def state(self,i):return json.loads((self.h.output/f'states/phase_{i:04d}.json').read_text(encoding='utf8'))

    def test_native_template_is_explicit_whitelist_and_preserves_material_Rk(self):
        code,mocks,solve,archive,old,_=self.run_native()
        self.assertEqual(code,0);old.assert_not_called();self.assertEqual(archive.call_count,1)
        template=solve.call_args.args[1]
        self.assertEqual(set(template),{'frames','driving_angle_rad','reference_follower_angle_rad','pinion_torque_nm','pinion_to_wheel_ratio','patches'})
        self.assertEqual(template['frames'],{'whole_body_R0':'native-initial'})
        p=template['patches'][0]
        self.assertEqual(set(p),{'tooth_offset','grid','plane','requested_mask','source_material_reference_frames'})
        self.assertEqual(p['source_material_reference_frames'],{'Rk':'explicit-reference'})
        self.assertTrue(solve.call_args.kwargs['include_body_fields']);self.assertNotIn('body_fields',self.state(1)['result'])
        self.assertTrue(self.state(1)['accepted']);self.assertTrue(self.h.summary()['analysis_completed'])
        self.assertEqual(mocks['generated_tca_frames'].call_args.kwargs['tooth_offset'],0)

    def test_native_template_mutation_cannot_change_original_CAD_window(self):
        h=self.h;geometry=dict(driving_angle_rad=0.,reference_follower_angle_rad=.01,pinion_torque_nm=-64.,pinion_to_wheel_ratio=17/47,
            patches=[dict(tooth_offset=0,grid={'x':1},valid_mask=[[True]],plane_origin_world_m=[0,0,0],
                plane_tangent_rows=[[1,0,0],[0,1,0]],plane_outward_normal_world=[0,0,1],frames={'Rk':[1,2,3]})])
        before=deepcopy(geometry)
        with patch('bevel_gear.generated_tca.generated_tca_frames',return_value={'R0':'whole'}):template=h.driver._native_template(SimpleNamespace(tca='tca'),geometry)
        p=template['patches'][0];p['requested_mask'][0][0]=False;p['plane']['origin_world_m'][0]=99;p['source_material_reference_frames']['Rk'][0]=9;p['grid']['x']=2
        self.assertEqual(geometry,before)

    def test_native_phase_archive_keeps_global_phase_and_API_identities_separate(self):
        import gzip
        from bevel_gear.native_fe_archive import native_phase_input_fingerprint
        code,_,_,archive,*_=self.run_native();self.assertEqual(code,0)
        global_state=self.state(1);summary=self.h.summary();folder=self.h.output/'native_responses/phase_0001'
        inputs=json.loads((folder/'inputs.json').read_text(encoding='utf8'))
        with gzip.open(folder/'state.json.gz','rt',encoding='utf8') as stream:phase=json.load(stream)
        final=json.loads((folder/'final_state_receipt.json').read_text(encoding='utf8'))
        fp=native_phase_input_fingerprint(inputs)
        self.assertNotIn('input_fingerprint',inputs)
        self.assertEqual(global_state['input_fingerprint'],summary['input_fingerprint'])
        self.assertEqual(global_state['result']['source_input_fingerprint'],summary['input_fingerprint'])
        self.assertNotEqual(fp,summary['input_fingerprint']);self.assertNotEqual(fp,'c'*64)
        self.assertEqual(phase['input_fingerprint'],fp);self.assertEqual(phase['result']['source_input_fingerprint'],fp)
        self.assertEqual(phase['response_receipt']['input_fingerprint'],fp)
        self.assertEqual(phase['native_contact_api_input_fingerprint'],'c'*64)
        self.assertEqual(phase['result']['native_contact_api_input_fingerprint'],'c'*64)
        self.assertEqual(phase['result']['current_operator_manifest_file'],'operator/manifest.json')
        self.assertEqual(archive.call_args.args[3].resolve(),folder.resolve())
        self.assertFalse((self.h.output/'responses').exists())
        reference=global_state['native_phase_archive']
        self.assertEqual(reference,summary['states'][0]['native_phase_archive'])
        self.assertEqual(self.h.output/reference['state_file'],folder/'state.json.gz')
        self.assertEqual(reference['state_sha256'],hashlib.sha256((folder/'state.json.gz').read_bytes()).hexdigest())
        self.assertEqual(final['source_state_sha256'],reference['state_sha256'])
        self.assertEqual(final['input_fingerprint'],fp)

    def test_native_rejected_middle_phase_never_inherits_acceptance_or_warm_pressure(self):
        code,_,solve,archive,_,_=self.run_native(selected=None,replies=[native_reply(),native_reply(False),native_reply()])
        self.assertEqual(code,1);self.assertEqual([r['accepted'] for r in self.h.summary()['states']],[True,False,True])
        self.assertEqual(archive.call_count,2)
        for call in solve.call_args_list:
            self.assertNotIn('initial_total_te_rad',call.kwargs);self.assertNotIn('initial_active_cell_keys',call.kwargs)
        for i in range(3):self.assertIsNone(self.state(i)['warm_start_from_phase_index'])
        self.assertNotIn('original_response_receipt',self.state(1)['result'])

    def test_native_exception_after_accepted_phase_is_rejected_and_stops(self):
        code,_,solve,archive,_,_=self.run_native(selected=None,replies=[native_reply(),RuntimeError('mock native失败')])
        self.assertEqual(code,1);self.assertEqual(solve.call_count,2);self.assertEqual(archive.call_count,1)
        self.assertTrue(self.state(0)['accepted']);self.assertFalse(self.state(1)['accepted'])
        self.assertEqual(self.state(1)['result']['status'],'phase_execution_failed');self.assertEqual(self.h.summary()['failure_stage'],'phase_1')

    def test_response_archive_exception_after_good_phase_cannot_be_accepted(self):
        code,_,_,_,_,_=self.run_native(selected=None,replies=[native_reply(),native_reply()],
            archive_effect=[{'passed':True},RuntimeError('mock原响应存档失败')])
        self.assertEqual(code,1);self.assertTrue(self.state(0)['accepted']);self.assertFalse(self.state(1)['accepted'])
        self.assertFalse(self.h.summary()['analysis_completed']);self.assertEqual(self.h.summary()['failure_stage'],'phase_1')

    def test_operator_archive_failure_and_failed_receipt_reject_current_phase(self):
        code,*_=self.run_native(operator_effect=RuntimeError('mock operator存档失败'))
        self.assertEqual(code,1);self.assertFalse(self.state(1)['accepted'])
        self.h.output=self.h.root/'second_output'
        code,*_=self.run_native(archive_effect=lambda *a,**k:{'passed':False})
        self.assertEqual(code,1);self.assertFalse(self.state(1)['accepted'])

    def test_native_truthy_nonboolean_acceptance_cannot_archive_or_pass(self):
        reply=native_reply();reply['accepted']='true'
        code,_,_,archive,_,_=self.run_native(replies=[reply])
        self.assertEqual(code,1);archive.assert_not_called();self.assertFalse(self.state(1)['accepted'])

    def test_explicit_reference_corrections_are_only_phase_initial_geometry(self):
        self.h.case['analysis']['reference_phase_corrections_rad']=[.01,.02,.03];self.h.write_case()
        code,mocks,solve,*_=self.run_native()
        self.assertEqual(code,0);self.assertEqual(mocks['build_generated_halfspace_patches'].call_args.kwargs['follower_correction_rad'],.02)
        self.assertEqual(solve.call_args.args[1]['reference_follower_angle_rad'],.02)

    def test_native_flank_extraction_failure_records_stage_and_closes_factors(self):
        with self.assertRaisesRegex(RuntimeError,'提取失败'):
            self.run_native(extraction_error=RuntimeError('mock提取失败'),factor_count=2)
        self.assertEqual(self.h.summary()['failure_stage'],'native_flank_preparation')
        for factor in self.last[-1]:factor.close.assert_called_once()
        self.last[1].assert_not_called()

    def test_optional_pardiso_builds_two_and_closes_after_archive_failure(self):
        code,_,_,_,_,factors=self.run_native(factor_count=2,archive_effect=RuntimeError('mock档案失败'))
        self.assertEqual(code,1);self.assertEqual(len(factors),2)
        for factor in factors:factor.close.assert_called_once()

    def test_third_factor_is_rejected_before_construct_and_both_are_closed(self):
        with self.assertRaisesRegex(RuntimeError,'仅各准备一个'):self.run_native(factor_count=3)
        self.assertEqual(len(self.last[-1]),2)
        for factor in self.last[-1]:factor.close.assert_called_once()
        self.assertEqual(self.h.summary()['failure_stage'],'preparation')

    def test_close_failure_in_first_factor_still_attempts_second_close(self):
        with self.assertRaisesRegex(RuntimeError,'两体均已尝试close'):self.run_native(factor_count=2,close_failure=True)
        self.assertEqual(len(self.last[-1]),2)
        for factor in self.last[-1]:factor.close.assert_called_once()

    def test_source_sha_change_after_native_response_rejects_state_and_closes(self):
        with self.assertRaisesRegex(RuntimeError,'计算源码改变'):
            self.run_native(factor_count=2,identity_effect=[{'kernel':'stable'},{'kernel':'changed'}])
        self.assertFalse(self.state(1)['accepted']);self.assertFalse(self.state(1)['source_identity_unchanged'])
        self.assertFalse(self.h.summary()['analysis_completed'])
        for factor in self.last[-1]:factor.close.assert_called_once()


if __name__=='__main__':unittest.main()
