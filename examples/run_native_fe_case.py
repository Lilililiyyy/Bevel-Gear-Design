"""实际弧齿锥齿轮原生FE案例：CAD、分级网格、fresh几何模板和多相／多格研究。

默认只列计划；--through mesh仅生成网格，绝不装配K。
同一prepared用于全部压力格和实际相位，各次原生接触均fresh cold求解。
"""
from pathlib import Path
from dataclasses import fields,asdict
from copy import deepcopy
from types import SimpleNamespace
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) in sys.path:sys.path.remove(str(PROJECT))
sys.path.insert(0,str(PROJECT))
if str(Path(__file__).resolve().parent) not in sys.path:sys.path.append(str(Path(__file__).resolve().parent))


def _identity():
    result={p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROJECT/'bevel_gear').glob('*.py'))}
    result['driver']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['driver_resource_helpers']=hashlib.sha256((Path(__file__).parent/'run_distributed_fe_contact.py').read_bytes()).hexdigest()
    return result


def _write(path,value):
    from bevel_gear.native_fe_operator import safe
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x',encoding='utf8') as stream:json.dump(safe(value),stream,ensure_ascii=False,allow_nan=False,separators=(',',':'));stream.write('\n')
    temporary.replace(path)


def _pure_template(template):
    from bevel_gear.native_fe_operator import safe
    from bevel_gear.native_case_geometry import validate_frames,validate_grid,validate_plane
    if not isinstance(template,dict) or set(template)!={'frames','driving_angle_rad','reference_follower_angle_rad','pinion_torque_nm','pinion_to_wheel_ratio','patches'}:
        raise ValueError('原生模板只接受明确几何／载荷参数，不读取旧gap／压力／C')
    for patch in template['patches']:
        if set(patch)!={'tooth_offset','grid','plane','requested_mask','source_material_reference_frames'}:raise ValueError('patch含非几何输入或缺少fresh mask')
    result=safe(template);validate_frames(result['frames'])
    for p in result['patches']:
        validate_grid(p['grid']);validate_plane(p['plane']);validate_frames(p['source_material_reference_frames'])
    return result


def _mesh_documents(prepared,destination,source_files=None):
    """每level使用安全basename；优先同输出文件系统link，失败则流式复制。"""
    from bevel_gear.native_case_geometry import write_json,file_sha256
    result={};digest={}
    for member,body in prepared.systems.items():
        target=destination/(member+'_mesh.json')
        if source_files is None:write_json(target,{f.name:getattr(body.mesh,f.name) for f in fields(body.mesh)})
        else:
            source=Path(source_files[member]).resolve()
            try:os.link(source,target)
            except OSError:shutil.copyfile(source,target)
            if file_sha256(source)!=file_sha256(target):raise ArithmeticError('本level实际mesh链接／复制后SHA不同')
        result[member]=target.name;digest[member]=file_sha256(target)
    return result,digest


def _cad_documents(prepared,inputs,destination,source_files):
    """核raw默认值、当前制造、网格绑定和真实BRep字节后，写安全相对文件。"""
    from bevel_gear.native_case_geometry import file_sha256,json_equal,normalized_manufacturing
    if source_files is None:
        if inputs.get('cad_binding_state')!='synthetic_FE_fixture_no_CAD':raise ValueError('真实案例必须提供双轮CAD文件；合成FE例须明确无CAD')
        return dict(cad_binding_state='synthetic_FE_fixture_no_CAD')
    if set(source_files)!={'pinion','wheel'}:raise ValueError('需要双轮真实BRep文件')
    actual=normalized_manufacturing(inputs['base_case'])
    if not json_equal(inputs.get('actual_manufacturing'),actual):raise ValueError('raw制造默认值与显式actual_manufacturing不同')
    digest={};files={}
    for m in ('pinion','wheel'):
        metadata=prepared.systems[m].mesh.metadata;source=Path(source_files[m]).resolve();value=file_sha256(source)
        if (not json_equal(metadata.get('actual_manufacturing_request'),actual[m]['request']) or
                not json_equal(metadata.get('actual_blank_settings'),actual[m]['blank']) or
                metadata.get('actual_relief_cad_sha256')!=value or inputs.get('actual_cad_sha256',{}).get(m)!=value):
            raise ValueError('CAD文件、制造／齿坯与mesh绑定身份不一致')
        target=destination/(m+'.brep')
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
        if file_sha256(target)!=value:raise ArithmeticError('本level BRep链接／复制后SHA不同')
        digest[m]=value;files[m]=target.name
    return dict(cad_binding_state='manufacturing_BRep_mesh_bound',actual_manufacturing=actual,
        actual_cad_sha256=digest,cad_files=files)


def _model_archive_documents(inputs,destination,source_archives):
    """真实workflow必须保留可零K恢复的完整模型描述及两BRep原字节。"""
    if inputs['cad_binding_state']=='synthetic_FE_fixture_no_CAD':
        if source_archives is not None or 'generated_model_archives' in inputs:raise ValueError('合成FE例不能伪称真实CAD模型档案')
        return {}
    if not isinstance(source_archives,dict) or set(source_archives)!={'pinion','wheel'}:raise ValueError('真实workflow缺少双轮完整模型档案')
    from bevel_gear.generated_model_archive import copy_generated_model_archive
    bindings={}
    for m,source in source_archives.items():
        if not isinstance(source,dict) or set(source)!={'descriptor_path','descriptor_sha256'}:raise ValueError('原模型档案源必须明确描述路径和SHA')
        binding=copy_generated_model_archive(source['descriptor_path'],destination,expected_descriptor_sha256=source['descriptor_sha256'],
            expected_member=m,expected_manufacturing=inputs['actual_manufacturing'][m],expected_gear_sha256=inputs['actual_cad_sha256'][m])
        if binding['gear_brep_file']!=inputs['cad_files'][m]:raise ValueError('模型档案gear basename与本workflow CAD不同')
        bindings[m]=binding
    return dict(generated_model_archives=bindings)


class _TelemetryFactor:
    """只读记录每次真实solve的iparm输出和进程读字节差，不改变求解参数。"""
    def __init__(self,factor,member,path):self._factor=factor;self._member=member;self._path=Path(path);self._sequence=0
    def __getattr__(self,name):return getattr(self._factor,name)
    @staticmethod
    def _read_bytes():
        try:
            import psutil
            return int(psutil.Process().io_counters().read_bytes)
        except Exception:return None  # 只读计量缺失／权限拒绝不冒充零读盘，也不影响原求解。
    def solve(self,rhs):
        started=time.perf_counter();before=self._read_bytes();failure=None
        try:return self._factor.solve(rhs)
        except BaseException as error:failure=dict(exception_type=type(error).__name__,message=str(error));raise
        finally:
            self._sequence+=1;after=self._read_bytes();iparm=getattr(self._factor,'iparm',None)
            names=(('actual_refinement_count',6),('refinement_limit',7),('backward_tolerance',8),('perturbed_pivots',13))
            outputs={key:int(iparm[index]) if iparm is not None and len(iparm)>index else 'unavailable' for key,index in names}
            row=dict(member=self._member,sequence=self._sequence,rhs_shape=list(getattr(rhs,'shape',())),
                seconds=time.perf_counter()-started,iparm_zero_based=outputs,
                process_read_bytes_before=before,process_read_bytes_after=after,
                process_read_bytes_delta=after-before if before is not None and after is not None else 'unavailable',
                read_bytes_scope='整个当前进程，并非仅该factor缓存',failure=failure)
            with self._path.open('a',encoding='utf8') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def run_grid_study(prepared,entries,output,*,workflow_inputs,compact_facets=None,solver=None,
        source_mesh_files=None,source_cad_files=None,source_model_archives=None,include_body_fields=False,progress=None):
    """同prepared执行level×相位列表；只编排既有native与正式档案writer。

    entries每项为name/phase_index/template，template也可由明确file+SHA读取。
    返回root study清单；各level为独立distributed workflow，不能把压力格当相位。
    调用者拥有prepared的生命周期，本函数不创建或关闭调用者的原K。
    """
    from bevel_gear.native_case_geometry import read_json,file_sha256
    from bevel_gear.native_fe_contact import solve_native_fe_contact
    from bevel_gear.native_fe_operator import save_native_operator
    from bevel_gear.native_fe_archive import save_native_fe_response,save_native_fe_phase_state,native_phase_input_fingerprint
    from bevel_gear.surface_workflow import _fingerprint
    from run_distributed_fe_contact import _actual_imports
    if not isinstance(entries,list) or not entries:raise ValueError('研究须有明确level／phase模板')
    phases=workflow_inputs['effective']['phases'];selected=workflow_inputs['effective']['selected']
    if (not isinstance(phases,list) or not phases or not isinstance(selected,list) or not selected or len(selected)!=len(set(selected)) or
            any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(phases) for i in selected)):
        raise ValueError('workflow必须明确完整实际角列表及原selected索引')
    groups={}
    for item in entries:
        name=item['name'];index=item['phase_index']
        if not isinstance(name,str) or not name or Path(name).name!=name or name in ('.','..'):raise ValueError('level名称不安全')
        if isinstance(index,bool) or not isinstance(index,int) or index not in selected:raise ValueError('模板索引不属于明确所选相位')
        row=groups.setdefault(name,{})
        if index in row:raise ValueError('同level原相位重复')
        row[index]=item
    if any(set(row)!=set(selected) for row in groups.values()):raise ValueError('每level必须完整包含原selected相位，不能过滤失败／缺失模板')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    materials={m:{key:workflow_inputs['base_case']['materials'][m][key] for key in ('young_modulus_pa','poisson_ratio')} for m in ('pinion','wheel')}
    identity=_identity();_actual_imports(identity);levels=[];all_ok=True
    for name,by_index in groups.items():
        folder=output/name;folder.mkdir();mesh_files,mesh_sha=_mesh_documents(prepared,folder,source_mesh_files)
        inputs=deepcopy(workflow_inputs)
        cad_binding=_cad_documents(prepared,inputs,folder,source_cad_files)
        inputs.update(cad_binding)
        model_binding=_model_archive_documents(inputs,folder,source_model_archives);inputs.update(model_binding)
        inputs.update(schema='bevel_native_FE_workflow_inputs_v1',geometry_source='FE_boundary',pressure_grid_name=name,
            mesh_files=mesh_files,mesh_sha256=mesh_sha,selected_phase_indices=selected,source_identity=identity)
        inputs['effective'].update(materials=materials)
        fp=_fingerprint(inputs);_write(folder/'distributed_fe_inputs.json',inputs)
        summary=dict(schema='bevel_distributed_fe_workflow_v1',geometry_source='FE_boundary',method='full_gear_fe',
            input_file='distributed_fe_inputs.json',input_fingerprint=fp,selected_phase_indices=selected,
            source_identity_before=identity,mesh_files=mesh_files,mesh_sha256=mesh_sha,states=[],analysis_completed=False,
            mesh_convergence_certified=False,temporal_convergence_certified=False)
        if 'refinement_phase_scope' in inputs:summary['refinement_phase_scope']=deepcopy(inputs['refinement_phase_scope'])
        summary.update(cad_binding)
        summary.update(model_binding)
        _write(folder/'distributed_fe_summary.json',summary)
        for i in selected:
            item=by_index[i];reply=None;archive=None;accepted=False;error=None
            try:
                if 'template' in item:template=_pure_template(item['template'])
                else:
                    file=Path(item['file'])
                    if file_sha256(file)!=item['sha256']:raise ValueError('fresh几何模板原SHA改变')
                    template=_pure_template(read_json(file)['template'])
                if template['driving_angle_rad']!=phases[i]:raise ValueError('模板与原相位角不同')
                reply=solve_native_fe_contact(prepared,template,compact_facets=compact_facets,
                    include_body_fields=True,progress=progress,**(solver or {}))
                core=dict(reply['result']);accepted=reply.get('accepted') is True
                after=_identity();_actual_imports(identity)
                if after!=identity:raise RuntimeError('计算源码改变，不保存为正式接受态')
                if accepted:
                    phase=folder/'native_responses'/f'phase_{i:04d}'
                    operator=save_native_operator(reply['system'],phase/'operator',actual_mesh_sha256=mesh_sha,
                        source_identity={'bevel_gear/'+key+'.py':value for key,value in identity.items() if key not in ('driver','driver_resource_helpers')})
                    core.update(operator,actual_native_mesh_sha256=mesh_sha,source_input_fingerprint=fp,
                        native_contact_api_input_fingerprint=reply['input_fingerprint'])
                    core['current_operator_manifest_file']=(phase/'operator/manifest.json').relative_to(folder).as_posix()
                    binding={key:operator[key] for key in ('current_operator_manifest_sha256','operator_fingerprint')}
                    binding['current_operator_manifest_file']='operator/manifest.json'
                    phase_inputs=dict(schema='bevel_native_FE_phase_inputs_v1',phase_index=i,driving_angle_rad=phases[i],
                        global_workflow_input_fingerprint=fp,workflow_inputs=inputs,materials=materials,geometry_template=template,
                        native_contact_api_input_fingerprint=reply['input_fingerprint'],native_contact_solver=solver or {},
                        actual_native_mesh_sha256=mesh_sha,final_operator_binding=binding)
                    phase_fp=native_phase_input_fingerprint(phase_inputs)
                    response=save_native_fe_response(prepared,reply['system'],core,phase,phase_index=i,input_fingerprint=phase_fp)
                    core['original_response_receipt']=response
                    saved=save_native_fe_phase_state(phase,inputs=phase_inputs,result=core,response_receipt=response,
                        phase_index=i,driving_angle_rad=phases[i],source_identity_before=identity,source_identity_after=after,
                        native_contact_api_input_fingerprint=reply['input_fingerprint'],global_workflow_input_fingerprint=fp,
                        include_body_fields=include_body_fields)
                    archive=dict(saved,path_base='workflow_root',run_root=phase.relative_to(folder).as_posix())
                    for key in ('state_file','final_state_receipt_file','response_receipt_file','current_operator_manifest_file'):
                        archive[key]=(phase/saved[key]).relative_to(folder).as_posix()
                if not include_body_fields:core.pop('body_fields',None)
            except Exception as failure:
                accepted=False;error=dict(exception_type=type(failure).__name__,message=str(failure))
                core=dict(converged=False,status='phase_execution_failed',failure=error)
            after=_identity()
            if after!=identity:
                accepted=False;core.update(status='source_identity_changed',source_identity_unchanged=False)
            state=dict(schema='bevel_distributed_fe_state_v1',phase_index=i,driving_angle_rad=phases[i],accepted=accepted,
                input_fingerprint=fp,source_identity_before=identity,source_identity_after=after,source_identity_unchanged=after==identity,
                warm_start_from_phase_index=None,result=core)
            if archive is not None:state['native_phase_archive']=archive
            relative=f'states/phase_{i:04d}.json';_write(folder/relative,state)
            row=dict(phase_index=i,driving_angle_rad=phases[i],accepted=accepted,state_file=relative,state_sha256=file_sha256(folder/relative),
                maximum_pressure_pa=core.get('maximum_pressure_pa'),total_loaded_te_rad=core.get('total_loaded_te_rad'))
            if archive is not None:row['native_phase_archive']=archive
            summary['states'].append(row);summary.update(source_identity_after=after,source_identity_unchanged=after==identity)
            _write(folder/'distributed_fe_summary.json',summary)
            if progress:progress(dict(stage='native_case_state_saved',level=name,phase_index=i,accepted=accepted))
            reply=None;gc.collect()
            if not accepted:all_ok=False;break
        summary['analysis_completed']=len(summary['states'])==len(selected) and all(row['accepted'] for row in summary['states'])
        _write(folder/'distributed_fe_summary.json',summary)
        levels.append(dict(name=name,summary_file=(folder/'distributed_fe_summary.json').relative_to(output).as_posix(),
            summary_sha256=file_sha256(folder/'distributed_fe_summary.json'),accepted=summary['analysis_completed']))
        if not summary['analysis_completed']:all_ok=False;break
    study=dict(schema='bevel_native_FE_grid_study_v1',levels=levels,analysis_completed=all_ok and len(levels)==len(groups),
        same_prepared_used=True,cross_grid_force_or_C_reused=False,geometry_source='FE_boundary',
        pressure_grid_convergence_certified=False,temporal_convergence_certified=False)
    _write(output/'study_summary.json',study);return study


def _load_meshes(directory,models,regions,cad_sha,*,background_controls=None):
    from bevel_gear.elasticity import TetraMesh
    from bevel_gear.native_case_geometry import read_json,json_equal
    import numpy as np
    meshes={}
    for m in ('pinion','wheel'):
        mesh=TetraMesh(**read_json(Path(directory)/(m+'_mesh.json')))
        if (not json_equal(mesh.metadata.get('actual_manufacturing_request'),asdict(models[m].request)) or
                not json_equal(mesh.metadata.get('actual_blank_settings'),asdict(models[m].settings)) or
                mesh.metadata.get('actual_relief_cad_sha256')!=cad_sha[m] or
                not json_equal(mesh.metadata.get('refinement_regions'),regions['regions'][m])):raise ValueError('复用网格的制造／齿坯／局部区域身份不同')
        if background_controls is not None:
            expected={'requested_element_size_m':background_controls['element_size_m'],
                'refined_size_m':background_controls['contact_refined_size_m'],
                'refinement_core_radius_m':background_controls['refinement_core_radius_m'],
                'refinement_radius_m':background_controls['refinement_radius_m']}
            if any(not json_equal(mesh.metadata.get(key),value) for key,value in expected.items()):
                raise ValueError('允许背景网格时，原mesh的实际主体/背景场尺寸必须与案例声明一致')
            origin=mesh.metadata.get('local_refinement_source_phase_rad')
            if origin is not None and not any(abs(origin-value)<=1e-12 for value in regions['source_driver_angles_rad']):
                raise ValueError('原mesh记录的局部轨迹来源相位与区域声明不同')
        # 原JSON与metadata保持原样；尽早释放百万行Python容器，固定数值布局。
        mesh.nodes_m=np.asarray(mesh.nodes_m,dtype=np.float64)
        for name in ('tetrahedra','boundary_triangles','boundary_tags'):
            raw=np.asarray(getattr(mesh,name))
            if raw.dtype.kind not in 'iu' or (raw.dtype.kind=='u' and raw.size and raw.max()>np.iinfo(np.int64).max):
                raise ValueError('mesh拓扑／标签须为int64范围内整数，不能截断实数或溢出索引')
            setattr(mesh,name,np.asarray(raw,dtype=np.int64))
        meshes[m]=mesh
    return meshes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',nargs='?',type=Path,default=PROJECT/'examples/spiral_17_47_relieved_native_fe_case.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--through',choices=('plan','cad','mesh','templates','solve'),default='plan')
    parser.add_argument('--mesh-dir',type=Path);parser.add_argument('--phase-index',type=int,action='append');parser.add_argument('--include-body-fields',action='store_true')
    parser.add_argument('--sparse-solver',choices=('scipy','pardiso64-ooc'),default='pardiso64-ooc')
    parser.add_argument('--mkl-runtime',type=Path);parser.add_argument('--factor-threads',type=int,default=4)
    parser.add_argument('--ooc-cache-root',type=Path);parser.add_argument('--pinion-ooc-core-mib',type=int);parser.add_argument('--wheel-ooc-core-mib',type=int)
    parser.add_argument('--ooc-maximum-process-rss-gib',type=float);parser.add_argument('--ooc-minimum-free-disk-gib',type=float);parser.add_argument('--ooc-minimum-available-gib',type=float)
    args=parser.parse_args(argv)
    from bevel_gear.native_case_geometry import load_case,case_plan,read_json,write_json,file_sha256,build_case_models,generate_case_meshes,build_geometry_templates,load_case_refinement_regions
    from run_distributed_fe_contact import _load_regions_file,_ooc64_options,_ooc64_factory,_save_ooc64_runtime
    loaded=load_case(args.case,phase_indices=args.phase_index);case=loaded['case'];base=loaded['base_case']
    regions,_,_,refinement_scope=load_case_refinement_regions(loaded,_load_regions_file)
    output=args.output.resolve()
    if output.exists():raise FileExistsError('输出须为独立新目录，不覆盖既有计算')
    options=_ooc64_options(args,sys.argv[1:] if argv is None else list(argv)) if args.through=='solve' and args.sparse_solver=='pardiso64-ooc' else None
    output.mkdir(parents=True);write_json(output/'case_plan.json',case_plan(loaded))
    if args.through=='plan':return 0
    models=build_case_models(loaded)
    from bevel_gear.cad_import import ImportedCadModel,export_model_step,export_model_brep
    from bevel_gear.generated_model_archive import save_generated_model_archive
    cad=output/'cad';cad.mkdir();model_archives={}
    for m,model in models.items():
        export_model_step(ImportedCadModel(model.shape,model.diagnostics),cad/(m+'.step'));export_model_brep(ImportedCadModel(model.shape,model.diagnostics),cad/(m+'.brep'))
        model_archives[m]=save_generated_model_archive(model,cad/(m+'_model.json'),member=m,gear_brep_path=cad/(m+'.brep'))
    cad_files={m:cad/(m+'.brep') for m in models};cad_sha={m:file_sha256(file) for m,file in cad_files.items()}
    write_json(cad/'model_archives.json',dict(schema='bevel_generated_model_workflow_archive_v1',generated_model_archives=model_archives,
        base_case=base,actual_manufacturing={m:dict(request=asdict(model.request),blank=asdict(model.settings)) for m,model in models.items()},
        actual_cad_sha256=cad_sha,cad_files={m:file.name for m,file in cad_files.items()},cad_binding_state='manufacturing_BRep_bound'))
    if args.through=='cad':return 0
    meshes=generate_case_meshes(loaded,models,regions,output/'meshes',cad_brep_files=cad_files) if args.mesh_dir is None else _load_meshes(args.mesh_dir,models,regions,cad_sha,background_controls=case['mesh'] if refinement_scope['policy']=='allow_global_background' else None)
    mesh_dir=output/'meshes' if args.mesh_dir is None else args.mesh_dir
    if args.through=='mesh':return 0
    from bevel_gear.generated_halfspace import prepare_generated_halfspace
    surface=prepare_generated_halfspace(models['pinion'].request,models['wheel'].request,cad_models=models)
    manifest=build_geometry_templates(loaded,surface,output/'templates')
    from bevel_gear.native_fe_preparation import extract_native_flank_compacts
    geometry_prepared=SimpleNamespace(models=models,systems={m:SimpleNamespace(mesh=mesh) for m,mesh in meshes.items()})
    compact,proof=extract_native_flank_compacts(geometry_prepared,loaded['tooth_offsets']);write_json(output/'compact_geometry_checks.json',proof)
    # 仅几何的真实FE覆盖预检；partial不得通过删mask进入后续QP。
    from bevel_gear.native_fe_operator import build_native_fe_contact_operator
    from bevel_gear.native_fe_contact import _coverage
    entries=[]
    for entry in manifest['entries']:
        file=output/'templates'/entry['file'];template=read_json(file)['template']
        operator=build_native_fe_contact_operator(compact,template['patches'],template['frames'],template['reference_follower_angle_rad'],
            driving_angle_rad=template['driving_angle_rad'],pinion_torque_nm=template['pinion_torque_nm'],pinion_to_wheel_ratio=template['pinion_to_wheel_ratio'])
        coverage=_coverage(operator);write_json(output/'preflight'/entry['name']/f"phase_{entry['phase_index']:04d}.json",coverage)
        if coverage['passed'] is not True:raise ValueError('原请求包含partial，已在创建K和QP前拒绝')
        entries.append(dict(entry,file=str(file)));del operator;gc.collect()
    if args.through=='templates':return 0
    from bevel_gear.generated_ltca import prepare_generated_ltca
    factors=[];runtime={};factory=None
    try:
        if options is not None:
            options['cache_root'].mkdir(parents=True,exist_ok=False)
            runtime.update(schema='bevel_sparse_OOC_runtime_v1',record_file=output/'linear_solver_runtime.json',
                cache_root=str(options['cache_root']),library_path=str(options['library_path']),protocol=options['protocol'],records=[],all_close_attempted=False)
            factory=_ooc64_factory(options,factors,runtime)
            original_factory=factory
            def factory(matrix):
                factor=original_factory(matrix)
                return _TelemetryFactor(factor,('pinion','wheel')[len(factors)-1],output/'linear_solver_solve_calls.jsonl')
        materials={m:{key:base['materials'][m][key] for key in ('young_modulus_pa','poisson_ratio')} for m in ('pinion','wheel')}
        prepared=prepare_generated_ltca(models['pinion'].request,models['wheel'].request,
            pinion_blank_settings=models['pinion'].settings,wheel_blank_settings=models['wheel'].settings,
            pinion_material=materials['pinion'],wheel_material=materials['wheel'],cad_models=models,meshes=meshes,
            directory=output/'elastic_preparation',maximum_nodes=case['mesh']['maximum_nodes'],
            **(dict(factorization_factory=factory) if factory else {}))
        if options is None:
            for m,body in prepared.systems.items():
                body.factorization=_TelemetryFactor(body.factorization,m,output/'linear_solver_solve_calls.jsonl')
        memory_policy=case.get('memory_policy','full')
        if memory_policy=='contact_only':
            from bevel_gear.native_contact_memory import release_prepared_assembly_caches
            released=release_prepared_assembly_caches(prepared)
            _write(output/'contact_memory_release.json',released)
        inputs=dict(base_case=base,native_case=case,case_sha256=loaded['case_sha256'],base_case_sha256=loaded['base_sha256'],regions_sha256=loaded['regions_sha256'],
            refinement_phase_scope=refinement_scope,
            geometry_source='FE_boundary',effective=dict(phases=[p['driving_angle_rad'] for p in case['phases']],selected=loaded['selected_phase_indices'],materials=materials),
            distributed_fe_case=dict(schema='bevel_distributed_fe_case_v1',geometry_source='FE_boundary',analysis=dict(driver_angles_rad=[p['driving_angle_rad'] for p in case['phases']])),
            selected_phase_indices=loaded['selected_phase_indices'])
        inputs.update(actual_manufacturing={m:dict(request=asdict(model.request),blank=asdict(model.settings)) for m,model in models.items()},
            actual_cad_sha256=cad_sha,cad_binding_state='manufacturing_BRep_mesh_bound')
        inputs['capabilities']=dict(memory_policy=memory_policy,element_stress_available=memory_policy=='full',
            standard_strength_workflow_independent=True)
        if options is not None:inputs['linear_solver']=options['protocol']
        study=run_grid_study(prepared,entries,output/'levels',workflow_inputs=inputs,compact_facets=compact,solver=case['solver'],
            source_mesh_files={m:Path(mesh_dir)/(m+'_mesh.json') for m in meshes},include_body_fields=args.include_body_fields,
            source_cad_files=cad_files,
            source_model_archives={m:dict(descriptor_path=cad/model_archives[m]['descriptor_file'],
                descriptor_sha256=model_archives[m]['descriptor_sha256']) for m in models},
            progress=lambda event:print(json.dumps(event,ensure_ascii=False),flush=True))
        return 0 if study['analysis_completed'] else 1
    finally:
        failures=[]
        for factor in factors:
            try:factor.close()
            except Exception as error:failures.append(dict(exception_type=type(error).__name__,message=str(error)))
        recording_failure=None
        if runtime:
            runtime.update(all_close_attempted=True,close_failures=failures)
            try:_save_ooc64_runtime(runtime,factors)
            except Exception as error:recording_failure=error
        if failures:raise RuntimeError('两体均尝试释放，但有分解器close失败')
        if recording_failure is not None:raise RuntimeError('两体已尝试释放，但运行证据写出失败') from recording_failure


if __name__=='__main__':raise SystemExit(main())
