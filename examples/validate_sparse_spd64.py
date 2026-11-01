"""有界小SPD：双OOC交替、首16/向量/满24请求及IC/SciPy制造解对照。

数值库、全新缓存根目录与结果文件均由调用者显式指定。没有工程模型
入口；阶数固定32³，不接受外部矩阵。要求实际oneMKL至少2025.2，建议在独立受限进程内运行。
"""
from pathlib import Path
from copy import deepcopy
import argparse
import ctypes as ct
import hashlib
import json
import os
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) in sys.path:sys.path.remove(str(PROJECT))
sys.path.insert(0,str(PROJECT))

from bevel_gear.sparse_spd64 import PardisoSPD64,array_identity


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def run_validation(library_path,cache_root):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import splu
    root=Path(cache_root).resolve()
    if root.exists():raise FileExistsError('小控制缓存根目录须是独立新目录')
    root.mkdir(parents=True,exist_ok=False)
    grid=32;e=sparse.eye(grid,format='csr')
    t=sparse.diags((-np.ones(grid-1),2*np.ones(grid),-np.ones(grid-1)),(-1,0,1),format='csr')
    a=(sparse.kron(sparse.kron(t,e),e)+sparse.kron(sparse.kron(e,t),e)+sparse.kron(sparse.kron(e,e),t)).tocsr()
    matrices=(a,(1.7*a+sparse.diags(np.linspace(.01,.1,a.shape[0]))).tocsr())
    exact=np.random.default_rng(20261031).normal(size=(a.shape[0],24))
    rhs=[np.asfortranarray(m@exact) for m in matrices]
    original=[dict(data=array_identity(m.data),indices=array_identity(m.indices),indptr=array_identity(m.indptr),rhs=array_identity(r)) for m,r in zip(matrices,rhs)]
    saved_env={k:v for k,v in os.environ.items() if k.startswith('MKL_PARDISO_OOC_')}
    handles=[];rows=[];responses={};first16={};local_restore=None;library=None;local_probe={}
    options=dict(library_path=library_path,max_core_mib=32,threads=4,maximum_rhs_columns=24,
        minimum_available_gib=1.,maximum_process_rss_gib=.9,minimum_free_disk_gib=1.,maximum_IC_gib=.5)
    try:
        for i,matrix in enumerate(matrices):
            handle=PardisoSPD64(matrix,cache_directory=root/f'OOC_{i}',auto_factor=False,**options)
            handles.append(handle)
            if i==0:
                library=handle.library;before=int(library.MKL_Get_Max_Threads())
                local_restore=int(library.MKL_Set_Num_Threads_Local(2))
                baseline=int(library.MKL_Get_Max_Threads())
                local_probe=dict(prior_effective_threads=before,prior_local_setting=local_restore,
                    requested_baseline_threads=2,effective_baseline_threads=baseline)
            else:baseline=int(library.MKL_Get_Max_Threads())
            if [call['phase'] for call in handle.calls]!=[11]:raise ArithmeticError('auto_factor=False未停在分析阶段')
            handle.factor()
            rows.append(dict(index=i,OOC_analysis=handle.analysis_diagnostics,local_baseline_effective_threads=baseline))
        requests=((0,16,False),(1,16,False),(0,1,True),(1,8,False),(0,24,False),(1,24,False),(0,16,False),(1,16,False))
        for i,count,vector in requests:
            requested=rhs[i][:,0] if vector else rhs[i][:,:count]
            target=exact[:,0] if vector else exact[:,:count]
            identity=array_identity(requested);answer=handles[i].solve(requested)
            residual=np.linalg.norm(matrices[i]@answer-requested,axis=0)/np.linalg.norm(requested,axis=0)
            manufactured=np.linalg.norm(answer-target,axis=0)/np.linalg.norm(target,axis=0)
            if max(residual.max(),manufactured.max())>2e-10:raise ArithmeticError('真实OOC残差／24列制造解不符')
            if array_identity(requested)!=identity or not answer.flags.owndata or answer.shape!=requested.shape:raise ArithmeticError('原请求字节／owned回形状不符')
            if count==16:
                if i in first16 and not np.array_equal(answer,first16[i]):raise ArithmeticError('双handle交替首16响应字节不一致')
                first16[i]=answer.copy()
            if count==24:responses[i]=answer.copy()
            rows[i].setdefault('OOC_solves',[]).append(dict(original_requested_shape=list(requested.shape),
                original_nrhs=count,actual_native_nrhs=24,maximum_relative_residual=float(residual.max()),
                maximum_manufactured_difference=float(manufactured.max()),fixed_rhs_metadata=deepcopy(handles[i].last_fixed_rhs_metadata)))
        for i,matrix in enumerate(matrices):
            with PardisoSPD64(matrix,cache_directory=root/f'IC_{i}',mode='IC',**options) as ic:
                ic_answer=ic.solve(rhs[i]);scipy_answer=splu(matrix.tocsc()).solve(rhs[i])
                ic_difference=np.linalg.norm(ic_answer-responses[i],axis=0)/np.linalg.norm(ic_answer,axis=0)
                scipy_difference=np.linalg.norm(scipy_answer-responses[i],axis=0)/np.linalg.norm(scipy_answer,axis=0)
                if max(ic_difference.max(),scipy_difference.max())>2e-10:raise ArithmeticError('真实IC/OOC/SciPy对照失败')
                rows[i]['IC_analysis']=ic.analysis_diagnostics
                rows[i]['maximum_IC_OOC_difference']=float(ic_difference.max())
                rows[i]['maximum_SciPy_OOC_difference']=float(scipy_difference.max())
                rows[i]['IC_calls_before_release']=ic.calls.copy()
            rows[i]['IC_calls_after_release']=ic.calls.copy()
        for i,handle in enumerate(handles):
            inventory=handle.cache_inventory()
            if not inventory or sum(p['bytes'] for p in inventory)==0:raise ArithmeticError('强制OOC没有真实盘外文件')
            rows[i]['OOC_cache_before_release']=inventory
            for call in handle.calls:
                info=call['thread_context']
                if info['effective_threads']!=4 or info['effective_threads_after']!=info['effective_threads_before']:
                    raise ArithmeticError('实际phase有效线程或线程恢复错误')
                if info['restore_setter_return']!=4:raise ArithmeticError('真实local setter恢复返回值不符')
            try:handle.solve(np.ones((a.shape[0],25)))
            except ValueError:pass
            else:raise ArithmeticError('未拒绝超出24列的RHS')
        # 保留原始输入和响应，供独立参考实现回读，禁止对象pickle。
        np.savez_compressed(root/'numeric_inputs_and_responses.npz',
            a_data=matrices[0].data,a_indices=matrices[0].indices,a_indptr=matrices[0].indptr,
            b_data=matrices[1].data,b_indices=matrices[1].indices,b_indptr=matrices[1].indptr,
            rhs0=rhs[0],rhs1=rhs[1],answer0=responses[0],answer1=responses[1],
            first16_answer0=first16[0],first16_answer1=first16[1],manufactured=exact,
            C0=rhs[0].T@responses[0],C1=rhs[1].T@responses[1],
            Ku0=matrices[0]@responses[0],Ku1=matrices[1]@responses[1])
    finally:
        for i,handle in enumerate(handles):
            handle.close()
            if i<len(rows):rows[i]['all_OOC_calls']=handle.calls.copy();rows[i]['OOC_cache_after_release']=handle.cache_inventory()
        if library is not None and local_restore is not None:
            local_probe['restore_setter_return']=int(library.MKL_Set_Num_Threads_Local(local_restore))
            local_probe['effective_threads_after_restore']=int(library.MKL_Get_Max_Threads())
            if local_probe['effective_threads_after_restore']!=local_probe['prior_effective_threads']:
                raise ArithmeticError('真实小控制结束时原线程局部设置未恢复')
    if saved_env!={k:v for k,v in os.environ.items() if k.startswith('MKL_PARDISO_OOC_')}:raise ArithmeticError('OOC环境未恢复')
    for before_identity,matrix,right in zip(original,matrices,rhs):
        if before_identity!=dict(data=array_identity(matrix.data),indices=array_identity(matrix.indices),indptr=array_identity(matrix.indptr),rhs=array_identity(right)):
            raise ArithmeticError('调用者原CSR/RHS数值字节改变')
    version=ct.create_string_buffer(512);library.MKL_Get_Version_String(version,ct.c_int(512))
    return dict(schema='standalone_pardiso64_real_small_control_v1',passed=True,matrix_dimension=a.shape[0],grid=grid,RHS_columns=24,
        RHS_columns_definition='实际原生固定批次宽度；各次原请求另列',original_request_columns=[r[1] for r in requests],
        native_rhs_batch_policy='fixed_maximum_zero_padded',actual_native_rhs_columns=24,
        library_sha256=sha(Path(library_path)),library_version=version.value.decode('ascii','replace'),
        actual_64bit_integer_protocol=True,global_interface_layer_changed=False,original_input_identities=original,
        inputs_and_responses_file='numeric_inputs_and_responses.npz',inputs_and_responses_sha256=sha(root/'numeric_inputs_and_responses.npz'),
        rows=rows,all_handles_released=all(h._released for h in handles),environment_restored=True,
        prior_local_setting_restored=True,actual_local_thread_probe=local_probe,real_engineering_matrix_used=False,
        validation_scope='仅有界小SPD真实IC/OOC/SciPy控制；不证明大模型填充、峰值、OOC性能或工程接触收敛')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--library-path',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True);parser.add_argument('--result-file',type=Path,required=True)
    args=parser.parse_args()
    if args.result_file.exists():raise FileExistsError('结果文件已经存在，拒绝覆盖')
    result=run_validation(args.library_path,args.cache_root)
    with args.result_file.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=True,matrix_dimension=result['matrix_dimension'],RHS_columns=24,all_handles_released=result['all_handles_released']),ensure_ascii=False))


if __name__=='__main__':main()
