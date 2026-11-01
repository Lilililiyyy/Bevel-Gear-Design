"""64位OOC前端的资源、逐相位存档及失败释放控制；不创建齿轮K。"""
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_distributed_fe_driver_native as native_support


class OOC64DriverTests(unittest.TestCase):
    def setUp(self):
        self.native=native_support.NativeDriverTests('runTest');self.native.setUp()
        self.addCleanup(self.native.doCleanups);self.h=self.native.h
        self.library=self.h.root/'explicit_numeric_runtime';self.library.write_bytes(b'only mock resource control')
        self.cache=self.h.root/'new_cache';self.instances=[];self.events=[]
        self.fail_construction=None;self.fail_factor=None;self.fail_close=None

    def args(self,**changes):
        values=dict(sparse_solver='pardiso64-ooc',mkl_runtime=self.library,factor_threads=2,
            ooc_cache_root=self.cache,pinion_ooc_core_mib=32,wheel_ooc_core_mib=48,
            ooc_maximum_process_rss_gib=1.,ooc_minimum_free_disk_gib=2.,
            ooc_minimum_available_gib=None,output=self.h.output)
        values.update(changes);return SimpleNamespace(**values)

    def fake_class(self):
        test=self
        class FakeSPD:
            """仅模拟真实接口的资源状态；数值引擎由独立小SPD例验证。"""
            def __init__(self,matrix,**options):
                self.index=len(test.instances);test.instances.append(self);self.options=options
                self.calls=[];self.iparm=[0]*64;self._released=False;self.close_attempts=0
                self.last_fixed_rhs_metadata=dict(original_nrhs=16,actual_native_nrhs=24,scope='mock接口记录，非数值证据')
                for index,value in ((14,1024),(15,256),(16,2048),(62,256)):self.iparm[index]=value
                self._call(11)
                self.analysis_diagnostics=dict(backend='oneMKL_PARDISO_64',integer_width_bits=64,
                    maximum_rhs_columns=options['maximum_rhs_columns'],max_core_MiB=options['max_core_mib'],
                    analysis_peak_KiB=1024,cache_directory=str(options['cache_directory']))
                if test.fail_construction==self.index:
                    self.close();raise MemoryError('模拟第二体11预算不足')
            def _call(self,phase,rhs=None):
                test.events.append((self.index,phase));self.calls.append(dict(phase=phase,error=0,nrhs=24))
            def factor(self):
                document=json.loads((test.h.output/'linear_solver_runtime.json').read_text(encoding='utf8'))
                record=document['records'][self.index]
                test.assertEqual(record['stage'],'phase11_approved');test.assertTrue(record['phase11_engine_completed'])
                self._call(22)
                if test.fail_factor==self.index:
                    self.close();raise ArithmeticError('模拟22失败')
                self.positive_definite=True
            def close(self):
                self.close_attempts+=1
                if not self._released:
                    try:
                        self._call(-1)
                        if test.fail_close==self.index:raise RuntimeError('模拟close失败')
                    finally:self._released=True
        return FakeSPD

    def run_ooc(self,**options):
        # 复用既有native档案mock，保留逐phase真实文件协议及旧分支保护。
        original=self.h.driver.main
        def call(argv):
            argv=list(argv);argv[argv.index('pardiso')]='pardiso64-ooc'
            argv+=['--ooc-cache-root',str(self.cache),'--pinion-ooc-core-mib','32',
                '--wheel-ooc-core-mib','48','--ooc-maximum-process-rss-gib','1',
                '--ooc-minimum-free-disk-gib','2']
            return original(argv)
        with patch('bevel_gear.sparse_spd64.PardisoSPD64',self.fake_class()),patch.object(self.h.driver,'main',side_effect=call):
            return self.native.run_native(factor_count=options.pop('factor_count',2),**options)

    def runtime(self):return json.loads((self.h.output/'linear_solver_runtime.json').read_text(encoding='utf8'))

    def test_options_explicit_resources_default_host_and_no_side_effect(self):
        options=self.h.driver._ooc64_options(self.args(),[]);protocol=options['protocol']
        self.assertEqual(protocol['maximum_rhs_columns'],24);self.assertEqual(protocol['minimum_available_gib'],10.)
        self.assertEqual(protocol['actual_native_rhs_columns'],24);self.assertEqual(protocol['native_rhs_batch_policy'],'fixed_maximum_zero_padded')
        self.assertEqual(protocol['per_body_core_mib'],dict(pinion=32,wheel=48))
        self.assertEqual(protocol['library_sha256'],hashlib.sha256(self.library.read_bytes()).hexdigest())
        self.assertNotIn(str(self.cache),json.dumps(protocol));self.assertFalse(self.cache.exists());self.assertFalse(self.h.output.exists())

    def test_missing_invalid_resources_and_legacy_IC_limit_are_rejected(self):
        for name in ('mkl_runtime','ooc_cache_root','pinion_ooc_core_mib','wheel_ooc_core_mib',
                'ooc_maximum_process_rss_gib','ooc_minimum_free_disk_gib'):
            with self.subTest(missing=name),self.assertRaises(ValueError):self.h.driver._ooc64_options(self.args(**{name:None}),[])
        for name in ('ooc_maximum_process_rss_gib','ooc_minimum_free_disk_gib','ooc_minimum_available_gib'):
            for value in (False,0,-1,float('inf'),float('nan')):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):self.h.driver._ooc64_options(self.args(**{name:value}),[])
        for name in ('pinion_ooc_core_mib','wheel_ooc_core_mib'):
            for value in (False,0,-1,1.5,2**63):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):self.h.driver._ooc64_options(self.args(**{name:value}),[])
        for arg in ('--maximum-factor-gib','--maximum-factor-gib=1'):
            with self.assertRaisesRegex(ValueError,'旧IC'):self.h.driver._ooc64_options(self.args(),[arg])
        self.assertFalse(self.cache.exists());self.assertFalse(self.h.output.exists())

    def test_cache_conflict_overlap_and_old_solver_cross_options_rejected(self):
        self.cache.mkdir();marker=self.cache/'keep';marker.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):self.h.driver._ooc64_options(self.args(),[])
        self.assertEqual(marker.read_bytes(),b'keep')
        for path in (self.h.output,self.h.output/'cache',self.h.root/'parent'):
            output=self.h.output if path!=self.h.root/'parent' else path/'output'
            with self.assertRaises(ValueError):self.h.driver._ooc64_options(self.args(ooc_cache_root=path,output=output),[])
        for solver in ('scipy','pardiso'):
            with self.assertRaisesRegex(ValueError,'仅适用于'):self.h.driver._ooc64_options(self.args(sparse_solver=solver),[])
        old=dict(ooc_cache_root=None,pinion_ooc_core_mib=None,wheel_ooc_core_mib=None,
            ooc_maximum_process_rss_gib=None,ooc_minimum_free_disk_gib=None,ooc_minimum_available_gib=None)
        self.assertIsNone(self.h.driver._ooc64_options(self.args(sparse_solver='scipy',**old),[]))

    def test_two_body_resources_and_formal_archive_are_preserved(self):
        code,*_=self.run_ooc();self.assertEqual(code,0);self.assertEqual(len(self.instances),2)
        runtime=self.runtime();self.assertTrue(runtime['all_close_attempted']);self.assertEqual(runtime['close_failures'],[])
        for member,core,handle,record in zip(('pinion','wheel'),(32,48),self.instances,runtime['records']):
            self.assertFalse(handle.options['auto_factor']);self.assertEqual(handle.options['maximum_rhs_columns'],24)
            self.assertEqual(handle.options['max_core_mib'],core)
            # Windows临时目录可为8.3短路径；比较解析后的实际目标，不比较别名文本。
            self.assertEqual(handle.options['cache_directory'].resolve(),(self.cache/member).resolve())
            self.assertTrue(record['handle_released']);self.assertTrue(record['release_engine_completed'])
            self.assertIn('last_fixed_rhs_metadata',record)
            self.assertEqual(record['last_fixed_rhs_metadata']['original_nrhs'],16)
            self.assertEqual(record['last_fixed_rhs_metadata']['actual_native_nrhs'],24)
            self.assertEqual([c['phase'] for c in record['calls']],[11,22,-1]);self.assertEqual(record['phase11_raw_KiB']['analysis_peak'],1024)
        inputs=json.loads((self.h.output/'distributed_fe_inputs.json').read_text(encoding='utf8'))
        self.assertEqual(inputs['linear_solver']['maximum_rhs_columns'],24)
        self.assertEqual(set(inputs['linear_solver_analysis']),{'pinion','wheel'})
        self.assertNotIn(str(self.cache),json.dumps(inputs));self.assertNotIn('cache_directory',json.dumps(inputs))
        state=self.native.state(1);self.assertTrue(state['accepted']);self.assertIn('native_phase_archive',state)
        phase_root=self.h.output/'native_responses/phase_0001'
        self.assertTrue((phase_root/'response_receipt.json').is_file());self.assertTrue((phase_root/'final_state_receipt.json').is_file())

    def test_second_constructor_failure_retains11_and_releases_prior_handle(self):
        self.fail_construction=1
        with self.assertRaisesRegex(MemoryError,'11预算不足'):self.run_ooc()
        runtime=self.runtime();self.assertTrue(runtime['all_close_attempted']);self.assertEqual(runtime['records'][1]['stage'],'construction_failed')
        self.assertEqual(runtime['records'][1]['phase11_raw_KiB']['analysis_peak'],1024)
        self.assertTrue(runtime['records'][1]['release_engine_completed']);self.assertTrue(all(h._released for h in self.instances))
        self.assertEqual([p for i,p in self.events if i==1],[11,-1]);self.assertEqual(self.h.summary()['failure_stage'],'preparation')

    def test_phase22_failure_registers_current_before_failure_and_closes_all(self):
        self.fail_factor=1
        with self.assertRaisesRegex(ArithmeticError,'22失败'):self.run_ooc()
        runtime=self.runtime();self.assertEqual(runtime['records'][1]['stage'],'phase22_failed')
        self.assertTrue(all(record['handle_released'] for record in runtime['records']))
        self.assertTrue(all(handle.close_attempts>=1 for handle in self.instances))

    def test_third_factor_does_not_construct_and_both_close(self):
        with self.assertRaisesRegex(RuntimeError,'仅各准备一个'):self.run_ooc(factor_count=3)
        self.assertEqual(len(self.instances),2);self.assertTrue(all(h._released for h in self.instances))

    def test_response_archive_failure_rejects_phase_and_closes_both(self):
        code,*_=self.run_ooc(archive_effect=RuntimeError('模拟原响应档案失败'))
        self.assertEqual(code,1);self.assertFalse(self.native.state(1)['accepted'])
        self.assertTrue(all(h._released for h in self.instances));self.assertTrue(self.runtime()['all_close_attempted'])

    def test_close_failure_still_attempts_second_and_records_error(self):
        self.fail_close=0
        with self.assertRaisesRegex(RuntimeError,'两体均已尝试close'):self.run_ooc()
        self.assertEqual([h.close_attempts for h in self.instances],[1,1])
        self.assertEqual(self.runtime()['close_failures'][0]['exception_type'],'RuntimeError')

    def test_runtime_record_failure_never_prevents_all_close_attempts(self):
        original=self.h.driver._save_ooc64_runtime
        def save(runtime,factors):
            if runtime.get('all_close_attempted'):raise OSError('模拟最终运行记录写失败')
            return original(runtime,factors)
        with patch.object(self.h.driver,'_save_ooc64_runtime',side_effect=save):
            with self.assertRaisesRegex(RuntimeError,'运行统计存档失败'):self.run_ooc()
        self.assertEqual([h.close_attempts for h in self.instances],[1,1]);self.assertTrue(all(h._released for h in self.instances))


if __name__=='__main__':unittest.main()
