"""接口控制的轻量测试；实际数值引擎由另行的小SPD对照验证。"""
import ctypes as ct
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from bevel_gear import sparse_spd64 as adapter


class FakeLibrary:
    """只用于状态／输入／资源控制，不用其结果宣称实际引擎通过。"""
    def __init__(self):
        self.phases=[];self.local=0;self.fail_phase=None;self.bad_output=False
        self.finite_bad_output=False;self.swap_columns=False
        self.version=b'Intel(R) oneAPI Math Kernel Library Version 2025.2-Product Build 20250620'
    def MKL_Get_Version_String(self,buffer,size):
        ct.memmove(buffer,self.version+b'\0',min(len(self.version)+1,size.value))
    def MKL_Get_Max_Threads(self):return self.local or 3
    def MKL_Set_Num_Threads_Local(self,value):
        previous=self.local;self.local=int(value);return previous
    def pardiso_64(self,pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x,error):
        scalar=lambda value:ct.cast(value,ct.POINTER(ct.c_longlong))[0]
        phase=scalar(phase);n=scalar(n);nrhs=scalar(nrhs)
        self.phases.append((phase,n,nrhs,self.MKL_Get_Max_Threads(),os.environ.get('MKL_PARDISO_OOC_FILE_NAME')))
        ip=np.ctypeslib.as_array(iparm,shape=(64,))
        if phase==11:
            ip[14]=1024;ip[15]=256;ip[16]=2048;ip[62]=256;ip[17]=n*2
            ptr=np.ctypeslib.as_array(ia,shape=(n+1,));columns=np.ctypeslib.as_array(ja,shape=(int(ptr[-1]),))
            data=np.ctypeslib.as_array(ct.cast(a,ct.POINTER(ct.c_double)),shape=(int(ptr[-1]),))
            upper=sparse.csr_matrix((data.copy(),columns.copy(),ptr.copy()),shape=(n,n)).toarray()
            self.matrix=upper+upper.T-np.diag(np.diag(upper))
        if phase==33:
            values=np.ctypeslib.as_array(ct.cast(b,ct.POINTER(ct.c_double)),shape=(n*nrhs,)).reshape((n,nrhs),order='F')
            self.last_rhs=values.copy(order='F')
            output=np.ctypeslib.as_array(ct.cast(x,ct.POINTER(ct.c_double)),shape=(n*nrhs,)).reshape((n,nrhs),order='F')
            output[:]=np.linalg.solve(self.matrix,values)
            if self.bad_output:output[0,0]=np.nan
            if self.finite_bad_output:output[0,0]+=1.
            if self.swap_columns:output[:]=output[:,::-1]
        if phase==self.fail_phase:ct.cast(error,ct.POINTER(ct.c_longlong))[0]=-4


@unittest.skipUnless(importlib.util.find_spec('psutil') is not None,'求解器构造控制需要可选psutil')
class AdapterControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.matrix=sparse.csr_matrix([[4.,-1.,0.],[-1.,4.,-1.],[0.,-1.,3.]])
        self.fake=FakeLibrary();self.loaded=patch.object(adapter,'_load_library',return_value=(self.fake,None));self.loaded.start()
        self.handles=[];self.counter=0
    def tearDown(self):
        for handle in self.handles:handle.close()
        self.loaded.stop();self.temp.cleanup()
    def make(self,matrix=None,**kw):
        self.counter+=1
        params=dict(library_path=self.base/'numeric_runtime',cache_directory=self.base/str(self.counter),
            max_core_mib=32,minimum_available_gib=.001,minimum_free_disk_gib=.001,maximum_process_rss_gib=4.)
        params.update(kw);handle=adapter.PardisoSPD64(self.matrix if matrix is None else matrix,**params)
        self.handles.append(handle);return handle

    def test_lazy_import_without_site_packages(self):
        script="import importlib.util,sys; s=importlib.util.spec_from_file_location('standalone',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert not any(x in sys.modules for x in ('numpy','scipy','psutil'))"
        result=subprocess.run([sys.executable,'-S','-B','-c',script,str(Path(adapter.__file__))],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_reject_nonreal_empty_nonsquare_nonfinite_or_invalid_sparse(self):
        cases=[np.eye(3),sparse.csr_matrix((0,0)),sparse.csr_matrix((3,3)),sparse.csr_matrix((2,3)),
            self.matrix.astype(complex),self.matrix.astype(bool),sparse.csr_matrix([[np.inf,0],[0,1]]),
            sparse.csr_matrix([[1.,2.],[0.,1.]]),sparse.csr_matrix([[-1.,0.],[0.,1.]])]
        invalid=sparse.csr_matrix(([1.],[4],[0,1,1]),shape=(2,2));cases.append(invalid)
        for matrix in cases:
            with self.subTest(shape=matrix.shape,dtype=matrix.dtype):
                with self.assertRaises((ValueError,TypeError)):self.make(matrix)
        self.assertFalse(self.fake.phases)

    def test_reject_nonpositive_noninteger_or_nonfinite_parameters(self):
        for key in ('max_core_mib','threads','maximum_rhs_columns'):
            for value in (False,0,-1,1.2,float('nan'),float('inf'),'2'):
                with self.subTest(key=key,value=value):
                    with self.assertRaises(ValueError):self.make(**{key:value})
        for key in ('maximum_IC_gib','minimum_available_gib','maximum_process_rss_gib','minimum_free_disk_gib'):
            for value in (False,0,-1,float('nan'),float('inf'),'2'):
                with self.subTest(key=key,value=value):
                    with self.assertRaises(ValueError):self.make(**{key:value})
        for key,value in [('threads',2**31),('mode','AUTO'),('auto_factor',0),('cache_directory',''),('library_path',None)]:
            with self.assertRaises(ValueError):self.make(**{key:value})
        self.assertFalse(self.fake.phases)

    def test_directory_conflict_preserves_existing_contents(self):
        folder=self.base/'old';folder.mkdir();marker=folder/'keep.txt';marker.write_text('keep',encoding='utf8')
        with self.assertRaises(FileExistsError):self.make(cache_directory=folder)
        self.assertEqual(marker.read_text(encoding='utf8'),'keep');self.assertFalse(self.fake.phases)

    def test_integer_duplicate_entries_convert_before_accumulation(self):
        large=np.iinfo(np.int64).max
        matrix=sparse.csr_matrix((np.array([large,large,1],dtype=np.int64),[0,0,1],[0,2,3]),shape=(2,2))
        original=matrix.data.copy();handle=self.make(matrix)
        self.assertEqual(handle.values[0],float(large)*2)
        np.testing.assert_array_equal(matrix.data,original)

    def test_analysis_then_factor_24_rhs_and_release_states(self):
        handle=self.make(auto_factor=False)
        self.assertEqual([p[0] for p in self.fake.phases],[11])
        with self.assertRaises(RuntimeError):handle.solve(np.ones(3))
        handle.factor();rhs=np.arange(72,dtype=float).reshape(3,24);before=rhs.copy()
        answer=handle.solve(rhs)
        np.testing.assert_array_equal(rhs,before);np.testing.assert_allclose(self.matrix@answer,rhs,atol=1e-12)
        self.assertTrue(all(p[2]==24 for p in self.fake.phases if p[0] in (11,22,33)))
        with self.assertRaises(RuntimeError):handle.factor()
        handle.close();n=len(self.fake.phases);handle.close();self.assertEqual(n,len(self.fake.phases))
        for action in (handle.factor,lambda:handle.solve(np.ones(3)),handle.__enter__):
            with self.assertRaises(RuntimeError):action()

    def test_invalid_rhs_does_not_destroy_valid_factor(self):
        handle=self.make()
        for rhs in (np.ones(3,dtype=complex),np.ones(3,dtype=bool),np.array([1.,np.nan,2.]),np.array([1.,np.inf,2.]),
                    np.ones((3,0)),np.ones((3,25)),np.ones((2,2)),np.ones((1,3,1)),['1','2','3'],1.):
            with self.assertRaises(ValueError):handle.solve(rhs)
        self.assertFalse(handle._released);np.testing.assert_allclose(self.matrix@handle.solve(np.ones(3)),1.)

    def test_phase11_budget_rejects_before22_and_releases(self):
        with self.assertRaises(MemoryError):self.make(max_core_mib=1)
        self.assertEqual([p[0] for p in self.fake.phases],[11,-1])
        self.fake.phases=[]
        with self.assertRaises(MemoryError):self.make(mode='IC',maximum_IC_gib=1e-9)
        self.assertEqual([p[0] for p in self.fake.phases],[11,-1])

    def test_resource_precheck_before_native_call(self):
        with patch.object(adapter.PardisoSPD64,'_resource_check',side_effect=MemoryError('模拟余量不足')):
            with self.assertRaises(MemoryError):self.make()
        self.assertFalse(self.fake.phases)

    def test_input_buffers_own_copies_and_mutation_is_rejected(self):
        original=self.matrix.copy();handle=self.make()
        np.testing.assert_array_equal(self.matrix.data,original.data)
        self.matrix.data[:]=100.
        np.testing.assert_allclose(original@handle.solve(np.ones(3)),1.)
        handle.values.flags.writeable=True;handle.values[0]+=1.
        with self.assertRaises(ArithmeticError):handle.solve(np.ones(3))
        self.assertTrue(handle._released)

    def test_per_call_environment_and_threads_restored_including_error(self):
        key='MKL_PARDISO_OOC_FILE_NAME';saved=os.environ.get(key);saved_path=os.environ.get('PATH');os.environ[key]='caller_marker'
        try:
            handle=self.make();handle.solve(np.ones(3))
            self.assertEqual(os.environ[key],'caller_marker');self.assertEqual(self.fake.local,0)
            for call in handle.calls:
                self.assertEqual(call['thread_context']['effective_threads'],1)
                self.assertEqual(call['thread_context']['effective_threads_after'],3)
                self.assertEqual(call['thread_context']['restore_setter_return'],1)
            self.fake.fail_phase=33
            with self.assertRaises(ArithmeticError):handle.solve(np.ones(3))
            self.assertEqual(os.environ[key],'caller_marker');self.assertEqual(self.fake.local,0);self.assertTrue(handle._released)
            self.assertEqual(os.environ.get('PATH'),saved_path)
        finally:
            if saved is None:os.environ.pop(key,None)
            else:os.environ[key]=saved

    def test_nonfinite_native_output_rejected_and_factor_released(self):
        handle=self.make();self.fake.bad_output=True
        with self.assertRaises(ArithmeticError):handle.solve(np.ones(3))
        self.assertTrue(handle._released)

    def test_actual_old_or_unparseable_runtime_rejected_before_phase11(self):
        for version in (b'Intel(R) Math Kernel Library Version 2019.0.4',b'unknown runtime'):
            self.fake.version=version
            with self.assertRaisesRegex(RuntimeError,'2025.2') as caught:self.make()
            self.assertEqual(self.fake.phases,[]);self.assertEqual(self.fake.local,0)
            failure=caught.exception.spd64_failure
            self.assertTrue(failure['handle_released']);self.assertEqual(failure['calls'],[])
            self.assertEqual(failure['runtime_version']['text'],version.decode('ascii'))

    def test_residual_operator_shares_all_original_int64_arrays(self):
        handle=self.make(auto_factor=False);guard=handle.analysis_diagnostics['residual_guard']
        self.assertTrue(all(guard['shared_arrays'].values()));self.assertEqual(guard['indices_dtype'],'int64')
        self.assertTrue(np.shares_memory(handle._residual_upper.data,handle.values))
        self.assertTrue(np.shares_memory(handle._residual_upper.indices,handle.columns))
        self.assertTrue(np.shares_memory(handle._residual_upper.indptr,handle.pointers))
        self.assertEqual(guard['relative_tolerance'],1e-8);self.assertEqual(guard['column_block'],1)
        handle.factor();right=np.arange(6,dtype=float).reshape(3,2);before=right.copy()
        answer=handle.solve(right);np.testing.assert_array_equal(right,before)
        np.testing.assert_allclose(self.matrix@answer,right,atol=1e-12)
        self.assertTrue(handle.last_solve_certificate['passed']);self.assertEqual(handle.last_solve_certificate['checked_columns'],24)

    def test_finite_wrong_U_is_rejected_with_column_values_and_released(self):
        handle=self.make();self.fake.finite_bad_output=True;right=np.array([1.,2.,3.]);before=right.copy()
        with self.assertRaisesRegex(ArithmeticError,'原upper K残差') as caught:handle.solve(right)
        np.testing.assert_array_equal(right,before);self.assertTrue(handle._released)
        proof=handle.last_solve_certificate;self.assertFalse(proof['passed']);self.assertEqual(proof['first_failed_column'],0)
        self.assertGreater(proof['columns'][0]['relative_residual'],1e-8)
        self.assertIs(caught.exception.spd64_failure,handle.last_failure)
        self.assertTrue(handle.last_failure['handle_released']);self.assertEqual(self.fake.phases[-1][0],-1)

    def test_swapped_distinct_RHS_columns_are_rejected_not_returned(self):
        handle=self.make();self.fake.swap_columns=True;right=np.array([[1.,3.],[2.,-1.],[.5,2.]])
        with self.assertRaisesRegex(ArithmeticError,'原upper K残差'):handle.solve(right)
        self.assertTrue(handle._released);self.assertFalse(handle.last_solve_certificate['passed'])
        self.assertGreater(handle.last_solve_certificate['columns'][0]['norm_R'],0)

    def test_zero_RHS_needs_exact_zero_residual(self):
        handle=self.make();answer=handle.solve(np.zeros((3,2)))
        np.testing.assert_array_equal(answer,0.);self.assertTrue(handle.last_solve_certificate['passed'])
        self.assertTrue(all(row['relative_residual']==0 for row in handle.last_solve_certificate['columns']))
        self.fake.finite_bad_output=True
        with self.assertRaises(ArithmeticError):handle.solve(np.zeros(3))
        row=handle.last_solve_certificate['columns'][0]
        self.assertEqual(row['norm_B'],0.);self.assertGreater(row['norm_R'],0.);self.assertIsNone(row['relative_residual'])
        self.assertTrue(handle._released)

    def test_hidden_CSR_copy_is_rejected_and_phase11_handle_closed(self):
        original=sparse.csr_array
        def copied(args,**kwargs):
            values,columns,pointers=args
            return original((values.copy(),columns.copy(),pointers.copy()),**kwargs)
        with patch.object(sparse,'csr_array',side_effect=copied):
            with self.assertRaisesRegex(RuntimeError,'共享') as caught:self.make()
        self.assertEqual([phase[0] for phase in self.fake.phases],[11,-1])
        self.assertFalse(all(caught.exception.spd64_failure['analysis_diagnostics']['residual_guard']['shared_arrays'].values()))

    def test_solve_budget_failure_closes_existing_factor_with_diagnostics(self):
        handle=self.make()
        with patch.object(handle,'_resource_check',side_effect=MemoryError('模拟残差预留不足')):
            with self.assertRaises(MemoryError) as caught:handle.solve(np.ones(3))
        self.assertTrue(handle._released);self.assertTrue(caught.exception.spd64_failure['handle_released'])
        self.assertEqual(self.fake.phases[-1][0],-1)

    def test_fixed_actual_batch_preserves_max_one_small_and_default_parameter_semantics(self):
        for maximum in (1,3,24):
            handle=self.make(maximum_rhs_columns=maximum)
            for count in sorted({1,maximum,min(2,maximum),min(16,maximum)}):
                rhs=np.arange(3*count*2,dtype=np.float32).reshape(3,count*2)[:,::2]
                before=adapter.array_identity(rhs);answer=handle.solve(rhs)
                self.assertEqual(adapter.array_identity(rhs),before);self.assertTrue(answer.flags.owndata)
                self.assertFalse(np.shares_memory(answer,rhs));self.assertEqual(answer.shape,rhs.shape)
                np.testing.assert_allclose(self.matrix@answer,rhs,atol=1e-12)
                self.assertEqual(self.fake.last_rhs.shape,(3,maximum))
                np.testing.assert_array_equal(self.fake.last_rhs[:,:count],rhs)
                np.testing.assert_array_equal(self.fake.last_rhs[:,count:],0.)
                info=handle.last_fixed_rhs_metadata
                self.assertEqual(info['original_nrhs'],count);self.assertEqual(info['actual_native_nrhs'],maximum)
                self.assertEqual(info['original_rhs_identity'],before)
                self.assertEqual(info['actual_native_rhs_identity'],handle.last_rhs_identity)
                self.assertEqual(handle.last_solve_certificate['checked_columns'],maximum)
                self.assertTrue(info['all_actual_original_Ku_columns_passed'])
                telemetry=next(row for row in reversed(handle.calls) if row['phase']==33)['fixed_rhs_telemetry']
                self.assertEqual(telemetry['actual_nrhs'],maximum)
                self.assertEqual(telemetry['original_request']['original_nrhs'],count)
                self.assertEqual(telemetry['actual_native_rhs_identity'],handle.last_rhs_identity)
            vector=np.array([1.,-2.,3.]);answer=handle.solve(vector)
            self.assertEqual(answer.shape,(3,));self.assertTrue(answer.flags.owndata)
            self.assertEqual(handle.last_fixed_rhs_metadata['original_input_shape'],[3])
            self.assertTrue(all(row['nrhs']==maximum for row in handle.calls if row['phase'] in (11,22,33)))

    def test_fixed_batch_padding_budget_failure_precedes_padding_and_closes(self):
        handle=self.make();calls=[];original=np.zeros
        def resource(extra=0):
            calls.append(int(extra))
            if len(calls)==2:raise MemoryError('模拟显式补零预算不足')
        def zeros(shape,*args,**kwargs):
            if tuple(shape)==(3,24):raise AssertionError('预算拒绝后不应分配补零矩阵')
            return original(shape,*args,**kwargs)
        with patch.object(handle,'_resource_check',side_effect=resource),patch.object(np,'zeros',side_effect=zeros):
            with self.assertRaises(MemoryError):handle.solve(np.ones(3))
        self.assertTrue(handle._released)
        metadata=handle.last_fixed_rhs_metadata;parts=metadata['padding_reserve_breakdown']
        self.assertEqual(metadata['padding_preallocation_reserve_bytes'],sum(parts.values()))
        self.assertGreaterEqual(parts['converted_prefix_finite_bytes'],3)
        self.assertTrue(handle.last_failure['handle_released'])

    def test_early_native_call_rejection_never_relabels_previous_call_telemetry(self):
        handle=self.make();handle.solve(np.ones(3));position=len(handle.calls)-1;previous=deepcopy(handle.calls[position])
        handle.iparm[7]=3
        with self.assertRaisesRegex(ArithmeticError,'最大细化'):handle.solve(np.ones(3))
        self.assertEqual(handle.calls[position],previous);self.assertTrue(handle._released)
        self.assertEqual([row['phase'] for row in handle.calls].count(33),1)

    def test_unavailable_IO_telemetry_does_not_change_original_solution(self):
        handle=self.make()
        with patch.object(handle,'_read_process_bytes',return_value=None):answer=handle.solve(np.ones(3))
        np.testing.assert_allclose(self.matrix@answer,1.)
        row=handle.calls[-1]['fixed_rhs_telemetry'];self.assertEqual(row['process_read_bytes_delta'],'unavailable')


if __name__=='__main__':unittest.main(verbosity=2)
