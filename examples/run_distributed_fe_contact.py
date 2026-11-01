"""显式制造输入→整轮FE分布接触；支持实际FE交片与CAD积分两条几何路线。"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import math
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) in sys.path:sys.path.remove(str(PROJECT))
sys.path.insert(0,str(PROJECT))


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _read_hashed(path):
    """一次读取的实际原字节同时用于JSON及SHA，不能另读一个版本来绑定。"""
    raw=Path(path).read_bytes()
    return json.loads(raw.decode('utf-8-sig')),hashlib.sha256(raw).hexdigest()


def _actual_imports(expected=None):
    """核实实际执行模块来自本项目源码，拒绝被其它安装目录或缓存包抢占。"""
    records={}
    for name,module in list(sys.modules.items()):
        if name!='bevel_gear' and not name.startswith('bevel_gear.'):continue
        filename=getattr(module,'__file__',None)
        if filename is None:raise RuntimeError('实际bevel_gear模块没有可核验源码文件：'+name)
        path=Path(filename).resolve()
        try:relative=path.relative_to(PROJECT/'bevel_gear')
        except ValueError as error:raise RuntimeError('实际导入模块不属于本项目源码：'+name) from error
        if path.suffix!='.py':raise RuntimeError('实际计算模块须可核验Python源码：'+name)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and expected.get(relative.stem)!=digest:
            raise RuntimeError('实际导入模块与本次冻结源码SHA不符：'+name)
        records[name]=dict(file='bevel_gear/'+relative.as_posix(),sha256=digest)
    return records


def _identity():
    identity={path.stem:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((PROJECT/'bevel_gear').glob('*.py'))}
    identity['driver']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return identity


def _json_equal(first,second):
    """按JSON语义比对请求；缓存回读的list不与原数据类tuple误判不同。"""
    canonical=lambda value:json.loads(json.dumps(value,allow_nan=False))
    return canonical(first)==canonical(second)


def _load_regions_file(path,selected_angles,bulk_m):
    """读取纯SI本体加密区；声明源相位仅限制使用范围，不构成收敛证书。"""
    raw=Path(path).read_bytes();document=json.loads(raw.decode('utf-8-sig'))
    allowed={'schema','units','coordinate_system','source_driver_angles_rad','manufacturing','regions','description'}
    if (not isinstance(document,dict) or set(document)-allowed or document.get('schema')!='bevel_refinement_regions_v1'
            or document.get('units')!='m' or document.get('coordinate_system')!='gear_body_local'):
        raise ValueError('需要bevel_refinement_regions_v1的纯SI整轮本体加密区域文件')
    phases=document.get('source_driver_angles_rad')
    if (not isinstance(phases,list) or not phases or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in phases)
            or len(set(phases))!=len(phases)):
        raise ValueError('加密区必须明确声明不重复的有限源相位rad')
    if any(not any(math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for b in phases) for a in selected_angles):
        raise ValueError('所选相位不在局部加密区声明的源相位中；单相位区域不能作为全周期加密输入')
    manufacturing=document.get('manufacturing');regions=document.get('regions')
    if (not isinstance(manufacturing,dict) or set(manufacturing)!={'pinion','wheel'} or
            not isinstance(regions,dict) or set(regions)!={'pinion','wheel'}):
        raise ValueError('加密区制造身份和区域须明确给出pinion/wheel')
    normalized={}
    for member in ('pinion','wheel'):
        identity=manufacturing[member]
        if not isinstance(identity,dict) or set(identity)!={'request','blank'}:raise ValueError('每轮加密区须绑定完整标准化request/blank')
        if not isinstance(regions[member],list):raise ValueError('每轮附加加密区须为列表')
        normalized[member]=[]
        for region in regions[member]:
            fields={'refinement_polylines_m','refined_size_m','refinement_core_radius_m','refinement_radius_m'}
            if not isinstance(region,dict) or set(region)-fields:raise ValueError('加密区域含未知字段')
            values=[]
            for key in ('refined_size_m','refinement_core_radius_m','refinement_radius_m'):
                value=region.get(key,region.get('refined_size_m')) if key=='refinement_core_radius_m' else region.get(key)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('加密区fine/core/radius须为正有限SI长度')
                values.append(float(value))
            fine,core,radius=values
            if not fine<bulk_m or not fine<=core<radius:raise ValueError('加密区须满足fine<bulk且fine≤core<radius')
            tracks=region.get('refinement_polylines_m');lines=[]
            if not isinstance(tracks,list) or not tracks:raise ValueError('加密区必须包含非空真实SI折线')
            for line in tracks:
                if not isinstance(line,list) or len(line)<2:raise ValueError('加密折线至少需要两个点')
                points=[]
                for point in line:
                    if (not isinstance(point,list) or len(point)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in point)):
                        raise ValueError('加密折线点须为有限三维SI坐标')
                    points.append([float(v) for v in point])
                if any(sum((x-y)**2 for x,y in zip(a,b))<=1e-24 for a,b in zip(points,points[1:])):raise ValueError('加密折线相邻点不能重合')
                lines.append(points)
            normalized[member].append(dict(refinement_polylines_m=lines,refined_size_m=fine,
                refinement_core_radius_m=core,refinement_radius_m=radius))
    if not any(normalized.values()):raise ValueError('附加区域文件不能两轮均为空')
    expanded=dict(document,regions=normalized)
    return expanded,hashlib.sha256(raw).hexdigest(),raw


def _failure(summary,output,stage,error):
    """保留原已接受相位及异常原因，不把未执行或失败相位补成零压力。"""
    after=_identity()
    summary.update(analysis_completed=False,failure_stage=stage,
        failure=dict(exception_type=type(error).__name__,message=str(error)),
        source_identity_after=after,source_identity_unchanged=after==summary['source_identity_before'])
    from bevel_gear.surface_workflow import _write
    _write(summary,output/'distributed_fe_summary.json')


def _native_template(prepared,geometry):
    """仅传窗口、格、平面与明确初始角；不传CAD间隙、旧压力或旧柔度。"""
    from copy import deepcopy
    from bevel_gear.generated_tca import generated_tca_frames
    return dict(frames=generated_tca_frames(prepared.tca,geometry['driving_angle_rad'],
                    geometry['reference_follower_angle_rad'],tooth_offset=0),
        driving_angle_rad=geometry['driving_angle_rad'],
        reference_follower_angle_rad=geometry['reference_follower_angle_rad'],
        pinion_torque_nm=geometry['pinion_torque_nm'],pinion_to_wheel_ratio=geometry['pinion_to_wheel_ratio'],
        patches=[dict(tooth_offset=p['tooth_offset'],grid=deepcopy(p['grid']),
            plane=dict(origin_world_m=deepcopy(p['plane_origin_world_m']),tangent_rows=deepcopy(p['plane_tangent_rows']),
                normal_world=deepcopy(p['plane_outward_normal_world'])),requested_mask=deepcopy(p['valid_mask']),
            source_material_reference_frames=deepcopy(p.get('source_material_reference_frames',p['frames']))) for p in geometry['patches']])


def _ooc64_options(args,arguments):
    """校验调用者的硬件资源选择；不创建目录或加载数值运行库。"""
    names=('ooc_cache_root','pinion_ooc_core_mib','wheel_ooc_core_mib',
        'ooc_maximum_process_rss_gib','ooc_minimum_free_disk_gib','ooc_minimum_available_gib')
    if args.sparse_solver!='pardiso64-ooc':
        if any(getattr(args,name) is not None for name in names):raise ValueError('OOC参数仅适用于--sparse-solver pardiso64-ooc')
        return None
    if any(str(value)=='--maximum-factor-gib' or str(value).startswith('--maximum-factor-gib=') for value in arguments):
        raise ValueError('--maximum-factor-gib仅表示旧IC分析上限；OOC须明确给出两体core预算')
    if args.mkl_runtime is None or not args.mkl_runtime.is_file():raise ValueError('pardiso64-ooc需要显式可读的--mkl-runtime')
    for name in names[:-1]:
        if getattr(args,name) is None:raise ValueError('pardiso64-ooc缺少显式资源参数：--'+name.replace('_','-'))
    if not 1<=args.factor_threads<2**31:raise ValueError('分解线程数须为32位范围内的正整数')
    core={member:getattr(args,member+'_ooc_core_mib') for member in ('pinion','wheel')}
    if any(isinstance(value,bool) or not isinstance(value,int) or not 1<=value<2**63 for value in core.values()):
        raise ValueError('两体OOC core须为正整数MiB')
    minimum_available=10. if args.ooc_minimum_available_gib is None else args.ooc_minimum_available_gib
    resources=dict(maximum_process_rss_gib=args.ooc_maximum_process_rss_gib,
        minimum_free_disk_gib=args.ooc_minimum_free_disk_gib,minimum_available_gib=minimum_available)
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0 for value in resources.values()):
        raise ValueError('OOC进程RSS、磁盘及主机余量须为正有限GiB')
    cache=args.ooc_cache_root.resolve();output=args.output.resolve()
    if cache.exists():raise FileExistsError('OOC缓存根已经存在；每次运行须使用独立新目录')
    if cache==output or cache in output.parents or output in cache.parents:
        raise ValueError('OOC缓存根须与分析输出目录独立，不能互相嵌套')
    library=args.mkl_runtime.resolve()
    return dict(cache_root=cache,library_path=library,protocol=dict(backend='pardiso64-ooc',mode='OOC',
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),integer_width_bits=64,
        threads=args.factor_threads,maximum_rhs_columns=24,actual_native_rhs_columns=24,
        native_rhs_batch_policy='fixed_maximum_zero_padded',per_body_core_mib=core,
        minimum_runtime_version='2025.2',original_upper_K_relative_residual_limit=1e-8,**resources))


def _save_ooc64_runtime(runtime,factors):
    """只记录真实调用及资源状态；绝对环境路径不进入数学输入指纹。"""
    if not runtime:return
    for index,factor in enumerate(factors):
        record=runtime['records'][index]
        record['calls']=list(getattr(factor,'calls',[]))
        record['handle_released']=getattr(factor,'_released',None) is True
        record['last_solve_certificate']=getattr(factor,'last_solve_certificate',None)
        record['last_fixed_rhs_metadata']=getattr(factor,'last_fixed_rhs_metadata',None)
        record['solver_failure']=getattr(factor,'last_failure',None)
    from bevel_gear.surface_workflow import _write
    _write({key:value for key,value in runtime.items() if key!='record_file'},runtime['record_file'])


def _ooc64_factory(options,factors,runtime):
    """两体固定次序的线性分解工厂；phase11先记录，登记handle后才22。"""
    from bevel_gear.sparse_spd64 import PardisoSPD64
    protocol=options['protocol'];members=('pinion','wheel')
    def create(matrix):
        if len(factors)>=2:raise RuntimeError('整轮两体仅各准备一个原K分解')
        member=members[len(factors)];record=dict(member=member,cache_subdirectory=member,stage='construction_started')
        runtime['records'].append(record)
        class ObservedSPD(PardisoSPD64):
            def _call(self,phase,rhs=None):
                try:return super()._call(phase,rhs)
                finally:
                    if phase==11:
                        calls=[call for call in self.calls if call.get('phase')==11]
                        record['phase11_engine_completed']=bool(calls and calls[-1]['error']==0)
                        record['phase11_raw_KiB']={name:int(self.iparm[index]) for name,index in
                            (('analysis_peak',14),('symbolic_permanent',15),('IC_numerical_factor',16),('extra_minimum_OOC',62))}
                        record['calls']=list(self.calls)
                        _save_ooc64_runtime(runtime,factors)
                    elif phase==-1:
                        record['calls']=list(self.calls)
                        released=[call for call in self.calls if call.get('phase')==-1]
                        record['release_engine_completed']=bool(released and released[-1]['error']==0)
                        _save_ooc64_runtime(runtime,factors)
        try:
            factor=ObservedSPD(matrix,library_path=options['library_path'],cache_directory=options['cache_root']/member,
                mode='OOC',max_core_mib=protocol['per_body_core_mib'][member],threads=protocol['threads'],
                maximum_rhs_columns=24,minimum_available_gib=protocol['minimum_available_gib'],
                maximum_process_rss_gib=protocol['maximum_process_rss_gib'],minimum_free_disk_gib=protocol['minimum_free_disk_gib'],
                auto_factor=False)
        except BaseException as error:
            record.update(stage='construction_failed',failure=dict(exception_type=type(error).__name__,message=str(error)))
            record['solver_failure']=getattr(error,'spd64_failure',None)
            _save_ooc64_runtime(runtime,factors);raise
        # 构造器已通过11及预算；先登记，避免22失败时主调用方遗漏此handle。
        factors.append(factor)
        record.update(stage='phase11_approved',analysis_diagnostics=dict(factor.analysis_diagnostics))
        _save_ooc64_runtime(runtime,factors)
        try:factor.factor()
        except BaseException as error:
            record.update(stage='phase22_failed',failure=dict(exception_type=type(error).__name__,message=str(error)))
            _save_ooc64_runtime(runtime,factors);raise
        record.update(stage='phase22_accepted',analysis_diagnostics=dict(factor.analysis_diagnostics))
        _save_ooc64_runtime(runtime,factors)
        return factor
    return create


def _run(argv=None,*,factor_handles,factor_runtime=None):
    _actual_imports()
    from bevel_gear.surface_workflow import _write,_fingerprint
    from bevel_gear.generated_solid import build_generated_gear
    from bevel_gear.generated_tca import prepare_generated_tca
    from bevel_gear.generated_nodal_ltca import generated_refinement_tracks
    from bevel_gear.generated_ltca import prepare_generated_ltca
    from bevel_gear.generated_halfspace import prepare_generated_halfspace,build_generated_halfspace_patches
    from bevel_gear.generated_distributed_fe import solve_generated_distributed_fe_ltca
    from bevel_gear.contact_plane import build_principal_aligned_patches
    from bevel_gear.elasticity import TetraMesh
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',nargs='?',type=Path,default=PROJECT/'examples/spiral_17_47_relieved_distributed_fe_case.json')
    parser.add_argument('--output',type=Path,required=True,help='输入、实际体网格、逐相位结果及汇总输出目录')
    parser.add_argument('--mesh-dir',type=Path,help='显式复用pinion_mesh.json和wheel_mesh.json所在目录')
    parser.add_argument('--phase-index',type=int,action='append',help='只计算指定原相位索引，可重复提供')
    parser.add_argument('--include-body-fields',action='store_true',help='保存两轮原位移和节点载荷，文件较大')
    parser.add_argument('--cold-start-each-phase',action='store_true',help='禁用上一个已接受相位的TE/活动格初值；所有相位仍完整验收')
    parser.add_argument('--sparse-solver',choices=('scipy','pardiso','pardiso64-ooc'),default='scipy',help='原Kff的稀疏分解器，不改变FE公式')
    parser.add_argument('--mkl-runtime',type=Path,help='pardiso时显式提供oneMKL运行库；不附带二进制')
    parser.add_argument('--factor-threads',type=int,default=1,help='pardiso分解线程数')
    parser.add_argument('--maximum-factor-gib',type=float,default=17.,help='每体PARDISO分析阶段的内存上限GiB')
    parser.add_argument('--ooc-cache-root',type=Path,help='64位OOC运行独占的新缓存根，须与分析输出目录独立')
    parser.add_argument('--pinion-ooc-core-mib',type=int,help='小轮OOC内部工作内存上限MiB，64位OOC必填')
    parser.add_argument('--wheel-ooc-core-mib',type=int,help='大轮OOC内部工作内存上限MiB，64位OOC必填')
    parser.add_argument('--ooc-maximum-process-rss-gib',type=float,help='整个计算进程的RSS预算GiB，64位OOC必填')
    parser.add_argument('--ooc-minimum-free-disk-gib',type=float,help='独立缓存卷最低剩余空间GiB，64位OOC必填')
    parser.add_argument('--ooc-minimum-available-gib',type=float,help='调用前主机最低可用内存GiB，64位OOC默认10')
    arguments=sys.argv[1:] if argv is None else list(argv)
    args=parser.parse_args(argv);path=args.case.resolve();case=_read(path)
    if case.get('schema')!='bevel_distributed_fe_case_v1':raise ValueError('需要分布牵引FE案例schema')
    if set(case)-{'schema','name','description','base_case','analysis','mesh','solver','acceptance','geometry_source'}:
        raise ValueError('分布牵引FE案例含未知顶层字段')
    geometry_source=case.get('geometry_source','CAD_surface_gauss')
    if geometry_source not in ('CAD_surface_gauss','FE_boundary'):
        raise ValueError('geometry_source须为CAD_surface_gauss或FE_boundary')
    native=geometry_source=='FE_boundary'
    ooc_options=_ooc64_options(args,arguments)
    factor_factory=None
    if args.sparse_solver=='pardiso':
        from bevel_gear.sparse_spd import PardisoSPD
        if args.mkl_runtime is None or not args.mkl_runtime.is_file():raise ValueError('pardiso需要明确可读的--mkl-runtime')
        if args.factor_threads<1 or not math.isfinite(args.maximum_factor_gib) or args.maximum_factor_gib<=0:
            raise ValueError('分解线程数和内存上限须为正')
        def factor_factory(matrix):
            if len(factor_handles)>=2:raise RuntimeError('整轮两体仅各准备一个原K分解')
            factor=PardisoSPD(matrix,library_path=args.mkl_runtime,
                maximum_factor_memory_gib=args.maximum_factor_gib,threads=args.factor_threads)
            factor_handles.append(factor)
            return factor
    base_path=(path.parent/case['base_case']).resolve();base=_read(base_path)
    analysis=dict(case['analysis']);phases=analysis.pop('driver_angles_rad')
    reference_corrections=analysis.pop('reference_phase_corrections_rad',None)
    alignment=analysis.pop('plane_alignment','none')
    alignment_step=analysis.pop('alignment_step_m',1e-4)
    alignment_fallback=analysis.pop('alignment_fallback_to_surface_basis',False)
    if alignment not in ('none','gap_hessian'):raise ValueError('plane_alignment须为none或gap_hessian')
    if isinstance(alignment_step,bool) or not isinstance(alignment_step,(int,float)) or not math.isfinite(alignment_step) or alignment_step<=0:
        raise ValueError('alignment_step_m须为正有限m')
    if not isinstance(alignment_fallback,bool):raise ValueError('主轴fallback开关须为布尔值')
    if alignment=='gap_hessian' and 'plane_basis_angles_rad' in analysis:raise ValueError('自动主轴不能与显式plane_basis_angles_rad同时指定')
    if (not isinstance(phases,list) or not phases or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) for v in phases)):raise ValueError('主动相位须为非空有限rad数组')
    if reference_corrections is not None and (not native or not isinstance(reference_corrections,list)
            or len(reference_corrections)!=len(phases) or any(isinstance(v,bool) or not isinstance(v,(int,float))
            or not math.isfinite(v) for v in reference_corrections)):
        raise ValueError('原生参考相位初值须为与主动相位一一对应的有限rad修正列表')
    selected=list(range(len(phases))) if args.phase_index is None else args.phase_index
    if (not selected or len(set(selected))!=len(selected) or any(not 0<=i<len(phases) for i in selected)):
        raise ValueError('所选相位须存在且不重复')
    mesh_controls=dict(case['mesh']);intervals=mesh_controls.pop('refinement_phase_intervals',4)
    region_file=mesh_controls.pop('refinement_regions_file',None);region_document=None;region_source=None;region_raw=None
    if isinstance(intervals,bool) or not isinstance(intervals,int) or intervals<1:raise ValueError('加密轨迹相位区间数须为正整数')
    half_widths=mesh_controls.pop('refinement_half_widths_m',[.0125,.002])
    track_spacing=mesh_controls.pop('track_spacing_m',1e-4);sample_spacing=mesh_controls.pop('sample_spacing_m',5e-4)
    expected={'element_size_m','contact_refined_size_m','refinement_radius_m','refinement_core_radius_m','maximum_nodes'}
    if set(mesh_controls)-expected:raise ValueError('FE体网格控制含未知字段')
    if region_file is not None:
        if not isinstance(region_file,str) or not region_file or Path(region_file).anchor:raise ValueError('refinement_regions_file须为相对case的文件路径')
        region_document,digest,region_raw=_load_regions_file(path.parent/region_file,[phases[i] for i in selected],mesh_controls.get('element_size_m',.001))
        region_source=dict(file=region_file,sha256=digest,output_file='refinement_regions_input.json')
    materials={m:{k:base['materials'][m][k] for k in ('young_modulus_pa','poisson_ratio')} for m in ('pinion','wheel')}
    load=base['load'];sign=load['ltca_torque_sign'];magnitude=load['pinion_torque_n_m']
    if isinstance(sign,bool) or sign not in (-1,1) or isinstance(magnitude,bool) or not isinstance(magnitude,(int,float)) or not math.isfinite(magnitude) or magnitude<=0:
        raise ValueError('扭矩幅值须为正有限Nm，工作侧符号须为±1')
    torque=sign*magnitude;manufacturing=base['manufacturing'];sampling=base.get('cad_sampling',{})
    solver=dict(case.get('solver',{}));limit=case.get('acceptance',{}).get('maximum_pinion_torque_relative_residual',.02)
    if not native:
        solver.setdefault('quadrature_order',4)
        solver.setdefault('maximum_geometry_error_m',1e-10)
    else:
        from bevel_gear.native_fe_contact import solve_native_fe_contact
        import inspect
        allowed=set(inspect.signature(solve_native_fe_contact).parameters)-{'prepared','geometry_template',
            'compact_facets','include_body_fields','progress'}
        if set(solver)-allowed:raise ValueError('原生FE案例含不适用的求解参数：'+', '.join(sorted(set(solver)-allowed)))
        solver.setdefault('maximum_pinion_torque_relative_residual',limit)
    if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<=0:
        raise ValueError('主动轴扭矩相对残差门限须为正有限数')
    if analysis.get('ray_backend','scalar') not in ('scalar','batch'):
        raise ValueError('真实曲面射线后端须为scalar或batch')
    if set(solver)&{'prepared','prepared_surface','initial_geometry','driving_angle_rad','pinion_torque_nm','ray_backend','include_body_fields',
            'initial_total_te_rad','initial_active_cell_keys','state_observer','progress'}:
        raise ValueError('solver不得重复覆盖业务入口已绑定的几何、载荷或射线参数')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('distributed_fe_inputs.json','distributed_fe_summary.json','states','refinement_regions_input.json','linear_solver_runtime.json')):
        raise FileExistsError('输出目录已有原计算记录；请另选目录，不能覆写原输入或接受状态')
    if args.mesh_dir is not None and args.mesh_dir.resolve()==output:
        raise ValueError('复用网格目录须与输出目录不同，避免改写原网格文件')
    if ooc_options is not None:
        ooc_options['cache_root'].mkdir(parents=True,exist_ok=False)
        if factor_runtime is None:factor_runtime={}
        factor_runtime.update(schema='bevel_sparse_OOC_runtime_v1',record_file=output/'linear_solver_runtime.json',
            cache_root=str(ooc_options['cache_root']),library_path=str(ooc_options['library_path']),
            protocol=ooc_options['protocol'],records=[],all_close_attempted=False)
        _save_ooc64_runtime(factor_runtime,factor_handles)
        factor_factory=_ooc64_factory(ooc_options,factor_handles,factor_runtime)
    identity=_identity();actual_before=_actual_imports()
    # 先核实际路径，再以完整冻结源码身份比较；之后新增延迟导入仍逐相位核验。
    if '__init__' in identity:_actual_imports(identity)
    inputs=dict(base_case=base,distributed_fe_case=case,selected_phase_indices=selected,
        signed_pinion_torque_nm=torque,effective_solver=solver,warm_start_enabled=not args.cold_start_each_phase,
        plane_alignment=dict(method=alignment,step_m=alignment_step,fallback_to_surface_basis=alignment_fallback),
        geometry_source=geometry_source,reference_phase_corrections_rad=reference_corrections,
        initial_phase_values_are_numerical_guesses_only=True,
        refinement_regions_source=region_source,refinement_regions_document=region_document,
        source_identity=identity,actual_imports_before=actual_before)
    if ooc_options is not None:inputs['linear_solver']=ooc_options['protocol']
    if region_raw is not None:(output/'refinement_regions_input.json').write_bytes(region_raw)
    fingerprint=_fingerprint(inputs);_write(inputs,output/'distributed_fe_inputs.json')
    summary=dict(schema='bevel_distributed_fe_workflow_v1',analysis_completed=False,
        method='full_gear_fe',contact_representation='distributed_P0_surface_cells',
        geometry_source=geometry_source,
        input_file='distributed_fe_inputs.json',input_fingerprint=fingerprint,
        selected_phase_indices=selected,source_identity_before=identity,states=[])
    summary['result_artifact_contract']=dict(raw_P0_pressure_and_actual_geometry=True,
        body_response_fields_saved=args.include_body_fields or native,
        body_fields_embedded_in_state=args.include_body_fields,native_phase_archive_required=native,mesh_convergence_certified=False,
        external_tooth_coverage_certified=False,
        completeness='analysis_completed仅指所选索引全部接受；子集不会自动成为完整一齿距结果')
    if region_source is not None:
        summary['refinement_regions_source']=region_source
        summary['refinement_regions_scope']=dict(source_driver_angles_rad=region_document['source_driver_angles_rad'],
            coverage_or_convergence_certified=False,definition='输入声明的源相位；不是全周期或实际承载带覆盖证明')
    def emit(**values):print(json.dumps(values,ensure_ascii=False),flush=True)
    _write(summary,output/'distributed_fe_summary.json')
    def prepare_bodies():
        emit(stage='build_actual_solids')
        models={m:build_generated_gear(manufacturing[m]['request'],manufacturing[m]['blank'],**sampling)
            for m in ('pinion','wheel')}
        if region_document is not None:
            for m,model in models.items():
                identity=region_document['manufacturing'][m]
                if not _json_equal(identity['request'],asdict(model.request)) or not _json_equal(identity['blank'],asdict(model.settings)):
                    raise ValueError('附加加密区域与实际标准化制造request/blank身份不一致')
        meshes=None;tracks=None
        if args.mesh_dir is not None:
            meshes={};sources={}
            for m in models:
                source=args.mesh_dir/(m+'_mesh.json');data,digest=_read_hashed(source)
                meshes[m]=TetraMesh(**data)
                sources[m]=dict(file=m+'_mesh.json',sha256=digest)
            summary['reused_mesh_sources']=sources
            for m in models:
                metadata=meshes[m].metadata or {}
                if (not _json_equal(metadata.get('actual_manufacturing_request'),asdict(models[m].request)) or
                        not _json_equal(metadata.get('actual_blank_settings'),asdict(models[m].settings))):
                    raise ValueError('复用体网格必须明确绑定相同制造请求及齿坯设置')
                if region_document is not None and not _json_equal(metadata.get('refinement_regions',[]),region_document['regions'][m]):
                    raise ValueError('复用网格的已有加密区域与所绑定SI区域文件不一致，不能假设已应用新局部场')
        else:
            tca=prepare_generated_tca(models['pinion'].request,models['wheel'].request,cad_models=models)
            z=models['pinion'].request.tooth_count;tracks={m:[] for m in models}
            for i in range(intervals+1):
                angle=-math.pi/z+2*math.pi/z*i/intervals
                control=generated_refinement_tracks(tca,driving_angle_rad=angle,tooth_offsets=analysis['tooth_offsets'],
                    half_widths_m=half_widths,track_spacing_m=track_spacing,sample_spacing_m=sample_spacing)
                for m in models:tracks[m].extend(control['tracks'][m])
            _write(dict(tracks=tracks,role='实际曲面的体网格加密带，不限定接触域'),output/'refinement_tracks.json')
        emit(stage='prepare_full_gear_FE')
        prepared=prepare_generated_ltca(manufacturing['pinion']['request'],manufacturing['wheel']['request'],
            pinion_blank_settings=manufacturing['pinion']['blank'],wheel_blank_settings=manufacturing['wheel']['blank'],
            pinion_material=materials['pinion'],wheel_material=materials['wheel'],directory=output/'prepare',
            cad_models=models,meshes=meshes,refinement_polylines_m=tracks,
            **(dict(factorization_factory=factor_factory) if factor_factory is not None else {}),
            **(dict(refinement_regions=region_document['regions']) if region_document is not None and meshes is None else {}),
            **sampling,**mesh_controls)
        summary['mesh_files']={m:m+'_mesh.json' for m in models}
        for m in models:
            mesh=prepared.systems[m].mesh
            mesh.metadata.update(actual_manufacturing_request=asdict(models[m].request),actual_blank_settings=asdict(models[m].settings))
            _write(asdict(mesh),output/summary['mesh_files'][m])
        summary['mesh_sha256']={m:hashlib.sha256((output/file).read_bytes()).hexdigest() for m,file in summary['mesh_files'].items()}
        _write(prepared.diagnostics,output/'prepared_geometry.json')
        surface=prepare_generated_halfspace(manufacturing['pinion']['request'],manufacturing['wheel']['request'],cad_models=models,**sampling)
        return prepared,surface
    try:
        prepared,surface=prepare_bodies()
    except Exception as error:
        _failure(summary,output,'preparation',error)
        raise
    # 原相位结果的输入指纹绑定实际体网格字节，而非只绑定名义网格尺寸。
    inputs.update(mesh_files=summary['mesh_files'],mesh_sha256=summary['mesh_sha256'],mesh_reused=args.mesh_dir is not None,
        reused_mesh_sources=summary.get('reused_mesh_sources'))
    if ooc_options is not None:
        inputs['linear_solver_analysis']={record['member']:{key:value for key,value in record['analysis_diagnostics'].items()
            if key!='cache_directory'} for record in factor_runtime['records']}
    fingerprint=_fingerprint(inputs);summary['input_fingerprint']=fingerprint
    _write(inputs,output/'distributed_fe_inputs.json');_write(summary,output/'distributed_fe_summary.json')
    native_compact=None
    if native:
        from bevel_gear.native_fe_preparation import extract_native_flank_compacts
        try:
            native_compact,compact_audit=extract_native_flank_compacts(prepared,analysis['tooth_offsets'],
                progress=lambda event:emit(**event))
            _write(compact_audit,output/'native_flank_preparation.json')
        except Exception as error:
            _failure(summary,output,'native_flank_preparation',error)
            raise
    previous=None
    for i in selected:
        emit(stage='distributed_FE_state',phase_index=i,driving_angle_rad=phases[i])
        exception=None
        reply=None
        initial_alignment=None
        phase_inputs=None;phase_receipt=None;phase_archive=None
        try:
            phase_analysis=dict(analysis)
            if reference_corrections is not None:phase_analysis['follower_correction_rad']=reference_corrections[i]
            if alignment=='gap_hessian':
                geometry=build_principal_aligned_patches(surface,phases[i],torque,step_m=alignment_step,
                    fallback_to_surface_basis=alignment_fallback,**phase_analysis)
                initial_alignment=geometry['grid_alignment']
            else:geometry=build_generated_halfspace_patches(surface,phases[i],torque,**phase_analysis)
            warm={}
            if previous is not None and not args.cold_start_each_phase and not native:
                warm=dict(initial_total_te_rad=previous['elastic_te_rad'],
                    initial_active_cell_keys=previous.get('active_cell_keys'))
            if native:
                from bevel_gear.native_fe_operator import save_native_operator
                from bevel_gear.native_fe_archive import save_native_fe_response,native_phase_input_fingerprint
                phase_template=_native_template(prepared,geometry)
                reply=solve_native_fe_contact(prepared,phase_template,compact_facets=native_compact,
                    include_body_fields=True,progress=lambda event:emit(**event),**solver)
                result=reply['result'];phase_dir=output/'native_responses'/f'phase_{i:04d}'
                _write(dict(accepted=reply['accepted'],geometry_outer_history=reply['geometry_outer_history'],
                    geometry_check=reply['geometry_check'],rejection_reasons=reply['rejection_reasons']),phase_dir/'geometry_check.json')
                if reply['accepted'] is True:
                    system=reply['system']
                    operator_identity=save_native_operator(system,phase_dir/'operator',actual_mesh_sha256=summary['mesh_sha256'],
                        source_identity={'bevel_gear/'+name+'.py':digest for name,digest in identity.items() if name!='driver'})
                    result.update(operator_identity,actual_native_mesh_sha256=summary['mesh_sha256'],
                        source_input_fingerprint=fingerprint,
                        native_contact_api_input_fingerprint=reply['input_fingerprint'])
                    result['current_operator_manifest_file']=(phase_dir/'operator/manifest.json').relative_to(output).as_posix()
                    phase_inputs=dict(schema='bevel_native_FE_phase_inputs_v1',phase_index=i,driving_angle_rad=phases[i],
                        global_workflow_input_fingerprint=fingerprint,workflow_inputs=inputs,
                        geometry_template=phase_template,native_contact_solver=solver,
                        native_contact_api_input_fingerprint=reply['input_fingerprint'],
                        actual_native_mesh_sha256=summary['mesh_sha256'],
                        final_operator_binding=dict(operator_identity,current_operator_manifest_file='operator/manifest.json'))
                    phase_fingerprint=native_phase_input_fingerprint(phase_inputs)
                    phase_receipt=save_native_fe_response(prepared,system,result,phase_dir,
                        phase_index=i,input_fingerprint=phase_fingerprint)
                    result['original_response_receipt']=phase_receipt
                if not args.include_body_fields:result.pop('body_fields',None)
            else:
                result=solve_generated_distributed_fe_ltca(prepared,phases[i],torque,prepared_surface=surface,
                    initial_geometry=geometry,ray_backend=analysis.get('ray_backend','scalar'),
                    include_body_fields=args.include_body_fields,**solver,**warm)
        except Exception as error:
            exception=error
            result=dict(converged=False,status='phase_execution_failed',
                failure=dict(exception_type=type(error).__name__,message=str(error)))
        after=_identity();residual=result.get('pinion_torque_relative_residual');te=result.get('total_loaded_te_rad' if native else 'elastic_te_rad')
        import_failure=None
        try:actual=_actual_imports(identity if '__init__' in identity else None)
        except RuntimeError as error:
            actual={};import_failure=error
        accepted=bool(result.get('converged') is True and result.get('geometry_linearization_checked') is True
            and result.get('equilibrium',{}).get('certificate',{}).get('passed') is True
            and after==identity and import_failure is None
            and not isinstance(residual,bool) and isinstance(residual,(int,float)) and math.isfinite(residual) and abs(residual)<=limit
            and not isinstance(te,bool) and isinstance(te,(int,float)) and math.isfinite(te))
        if native:accepted=accepted and reply is not None and reply.get('accepted') is True and result.get('original_response_receipt',{}).get('passed') is True
        if native and accepted:
            from bevel_gear.native_fe_archive import save_native_fe_phase_state
            try:
                archived=save_native_fe_phase_state(phase_dir,inputs=phase_inputs,result=result,
                    response_receipt=phase_receipt,phase_index=i,driving_angle_rad=phases[i],
                    source_identity_before=identity,source_identity_after=after,
                    native_contact_api_input_fingerprint=reply['input_fingerprint'],
                    global_workflow_input_fingerprint=fingerprint,include_body_fields=args.include_body_fields)
                phase_archive=dict(archived,path_base='workflow_root',run_root=phase_dir.relative_to(output).as_posix())
                for name in ('state_file','final_state_receipt_file','response_receipt_file','current_operator_manifest_file'):
                    phase_archive[name]=(phase_dir/archived[name]).relative_to(output).as_posix()
            except Exception as error:
                # 数学core保留；档案未闭合的相位不能在global周期中接受。
                exception=error;accepted=False
                result.update(status='native_phase_archive_failed',
                    archive_failure=dict(exception_type=type(error).__name__,message=str(error)))
        state=dict(schema='bevel_distributed_fe_state_v1',phase_index=i,driving_angle_rad=phases[i],
            accepted=accepted,input_fingerprint=fingerprint,source_identity_before=identity,
            source_identity_after=after,source_identity_unchanged=after==identity,
            actual_imports=actual,strict_import_binding_verified=import_failure is None,
            import_binding_failure=None if import_failure is None else dict(exception_type=type(import_failure).__name__,message=str(import_failure)),
            warm_start_from_phase_index=previous['phase_index'] if previous is not None and not args.cold_start_each_phase else None,
            grid_alignment=initial_alignment,
            result=result)
        if phase_archive is not None:state['native_phase_archive']=phase_archive
        reference=f'states/phase_{i:04d}.json';_write(state,output/reference)
        summary['states'].append(dict(phase_index=i,driving_angle_rad=phases[i],state_file=reference,
            state_sha256=hashlib.sha256((output/reference).read_bytes()).hexdigest(),
            accepted=accepted,status=result.get('status'),maximum_pressure_pa=result.get('maximum_pressure_pa'),
            elastic_te_rad=result.get('elastic_te_rad'),minimum_gauss_remaining_gap_m=result.get('minimum_gauss_remaining_gap_m'),
            total_loaded_te_rad=result.get('total_loaded_te_rad'),geometry_source=geometry_source,
            pinion_torque_relative_residual=residual))
        if phase_archive is not None:summary['states'][-1]['native_phase_archive']=phase_archive
        summary.update(source_identity_after=after,source_identity_unchanged=after==identity)
        previous=dict(phase_index=i,elastic_te_rad=result['elastic_te_rad'],active_cell_keys=result.get('active_cell_keys')) if accepted and not native else None
        _write(summary,output/'distributed_fe_summary.json')
        if after!=identity:raise RuntimeError('接触计算期间计算源码改变，保留原结果并拒绝验收')
        if import_failure is not None:
            _failure(summary,output,'import_binding_phase_'+str(i),import_failure)
            emit(stage='import_binding_failed',phase_index=i,message='实际导入来源未通过；保留原结果并拒绝验收')
            return 1
        if exception is not None:
            _failure(summary,output,'phase_'+str(i),exception)
            emit(stage='phase_execution_failed',phase_index=i,message='保留原已完成结果，当前相位未被接受')
            return 1
    summary['analysis_completed']=len(summary['states'])==len(selected) and all(s['accepted'] for s in summary['states'])
    _write(summary,output/'distributed_fe_summary.json');emit(stage='completed',analysis_completed=summary['analysis_completed'])
    return 0 if summary['analysis_completed'] else 1


def main(argv=None):
    """保证可选外部分解器在失败或正常退出后释放；不改变原相位接受条件。"""
    factors=[];runtime={}
    try:return _run(argv,factor_handles=factors,factor_runtime=runtime)
    finally:
        failures=[]
        for factor in factors:
            try:factor.close()
            except Exception as error:failures.append(error)
        recording_failure=None
        if runtime:
            runtime.update(all_close_attempted=True,close_failures=[dict(exception_type=type(error).__name__,message=str(error)) for error in failures])
            try:_save_ooc64_runtime(runtime,factors)
            except Exception as error:recording_failure=error
        if failures:raise RuntimeError('可选分解器释放失败；两体均已尝试close') from failures[0]
        if recording_failure is not None:raise RuntimeError('可选分解器已尝试释放，但运行统计存档失败') from recording_failure


if __name__=='__main__':raise SystemExit(main())
