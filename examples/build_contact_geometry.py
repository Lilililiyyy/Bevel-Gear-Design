"""重建显式弧齿接触制造案例的真实CAD，并导出双语几何细节图。

直接读取bevel_contact_manufacturing_case_v1，独立构建实际修缘闭实体。
只复用已有建模、相切、交换及预览接口；不附加名义强度或受载结果。
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
import math
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))


def _sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def _default(value):
    for name in ('as_dict','tolist','item'):
        if callable(getattr(value,name,None)):return getattr(value,name)()
    raise TypeError('几何结果包含不能保存的类型：'+type(value).__name__)


def _write(value,path):
    """流式保存实际网格，不一次复制大型三角数组为JSON字符串。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=_default);stream.write('\n')


def _identity(plots):
    names=['generated_cutter','generated_surface','flank_relief','generated_solid','generated_tca','cad','cad_import','rigid_contact','workflow']
    if plots:names+=['workflow_plotting','cad_preview','publication_style']
    return {f'bevel_gear/{name}.py':_sha(PROJECT/'bevel_gear'/(name+'.py')) for name in names}


def _canonical_face_ids(shape,mesh):
    """将三角化遍历号显式映射为当前BRep唯一面号，再使用槽区域标签。

    两种编号不能直接混用，STEP回读后的编号也不能继承。本映射仅
    对此shape的当前三角网格有效，不通过表面位置或颜色猜区域。
    """
    from OCP.TopExp import TopExp,TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopAbs import TopAbs_FACE
    mapping=TopTools_IndexedMapOfShape();TopExp.MapShapes_s(shape,TopAbs_FACE,mapping)
    explorer=TopExp_Explorer(shape,TopAbs_FACE);indices=[]
    while explorer.More():
        index=mapping.FindIndex(explorer.Current())-1
        if index<0:raise ArithmeticError('遍历面未进入当前BRep唯一面映射')
        indices.append(index);explorer.Next()
    mesh['face_indices']=[indices[i] for i in mesh['face_indices']]
    mesh['metadata'].update(face_index_scope='当前BRep唯一面0起始序号；由实际遍历映射，不适用于STEP回读',
        source_unique_face_count=mapping.Extent(),traversal_to_unique_face_indices=indices)


def build_contact_geometry(case_path,output,*,plots=False,languages=('zh','en'),driving_angle_rad=0.,
        tooth_offsets=(-1,0,1),linear_deflection_m=20e-6,angular_deflection_rad=.08,progress=None):
    """返回本次制造CAD的相对文件清单、真实相切姿态及选区面号。"""
    case_path=Path(case_path);output=Path(output)
    case=json.loads(case_path.read_text(encoding='utf-8-sig'))
    if not isinstance(case,dict) or case.get('schema')!='bevel_contact_manufacturing_case_v1':
        raise ValueError('仅支持schema=bevel_contact_manufacturing_case_v1的显式制造案例')
    if case.get('kind')!='spiral':raise ValueError('本入口仅支持实际弧齿制造案例kind=spiral')
    manufacturing=case.get('manufacturing')
    if not isinstance(manufacturing,dict) or set(manufacturing)!={'pinion','wheel'}:
        raise ValueError('manufacturing须显式包含pinion和wheel')
    for member,value in manufacturing.items():
        if not isinstance(value,dict) or set(value)!={'request','blank'}:
            raise ValueError(member+'须直接给出request和blank，不能使用隐式名义齿轮替换')
    sampling=case.get('cad_sampling',{})
    if not isinstance(sampling,dict) or set(sampling)-{'width_count','profile_count'}:raise ValueError('cad_sampling含未支持字段')
    for key,value in sampling.items():
        if isinstance(value,bool) or not isinstance(value,int) or value<2:raise ValueError(key+'须为不小于2的整数')
    for key,value in dict(driving_angle_rad=driving_angle_rad,linear_deflection_m=linear_deflection_m,angular_deflection_rad=angular_deflection_rad).items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (key!='driving_angle_rad' and value<=0):
            raise ValueError(key+'须为合法有限SI数值')
    if not isinstance(plots,bool) or not languages or set(languages)-{'zh','en'}:raise ValueError('plots须为布尔值，languages仅支持zh/en')
    if not tooth_offsets or len(set(tooth_offsets))!=len(tooth_offsets) or any(isinstance(k,bool) or not isinstance(k,int) for k in tooth_offsets):
        raise ValueError('无载相切候选齿号须为不重复的非空整数序列')
    sign=case.get('load',{}).get('ltca_torque_sign')
    if isinstance(sign,bool) or sign not in (-1,1):raise ValueError('案例须显式给出load.ltca_torque_sign为-1或+1，用于选择首次相切方向')
    if output.exists():raise FileExistsError('输出目录已经存在，请指定新的独立目录')

    from bevel_gear.generated_solid import build_generated_gear
    from bevel_gear.generated_tca import prepare_generated_tca,solve_generated_tca
    from bevel_gear.cad_import import ImportedCadModel,export_model_step,export_model_brep,triangulate_imported_model
    from bevel_gear.workflow import _assembly,_surface_subset,_selected_tca
    output.mkdir(parents=True,exist_ok=False);_write(case,output/'inputs.json')
    identity_before=_identity(plots)
    result=dict(schema='bevel_contact_geometry_v1',kind='spiral',geometry_completed=False,
        input_file_name=case_path.name,input_sha256=_sha(case_path),length_unit='m',angle_unit='rad',
        cad_internal_length_unit='mm',brep_coordinate_unit='mm',source_identity_before=identity_before,
        scope='显式制造CAD与无载相切装配；未执行标准强度或受载接触',members={},cad={},plots=[])
    _write(result,output/'geometry_summary.json')
    def report(stage,**values):
        if progress:progress(dict(stage=stage,**values))
    models={}
    for member,settings in manufacturing.items():
        report('build_solid',member=member)
        model=build_generated_gear(settings['request'],settings['blank'],**sampling)
        if model.diagnostics.get('solids')!=1 or model.diagnostics.get('valid') is not True:
            raise ArithmeticError(member+'实际材料未形成单一有效闭实体')
        models[member]=model;folder=output/'cad'/member
        imported=ImportedCadModel(model.shape,model.diagnostics)
        export_model_step(imported,folder/'gear.step');export_model_brep(imported,folder/'gear.brep')
        export_model_brep(ImportedCadModel(model.blank.shape,model.blank.diagnostics),folder/'blank.brep')
        record=dict(request=asdict(model.request),settings=asdict(model.settings),diagnostics=model.diagnostics,
            blank_diagnostics=model.blank.diagnostics,blank_profile_m=model.blank.profile_m,
            face_groups=model.face_groups,slot_face_groups=model.slot_face_groups)
        _write(record,folder/'model.json');_write(model.grid,folder/'grid.json')
        result['members'][member]=dict(request=record['request'],blank_settings=record['settings'],diagnostics=model.diagnostics,
            files={name:dict(path=(folder/name).relative_to(output).as_posix(),sha256=_sha(folder/name)) for name in ('gear.step','gear.brep','blank.brep','model.json','grid.json')})
        report('solid_built',member=member,solids=model.diagnostics['solids'],volume_m3=model.diagnostics['volume_m3'])
        mesh=triangulate_imported_model(imported,linear_deflection_m=linear_deflection_m,angular_deflection_rad=angular_deflection_rad)
        _canonical_face_ids(model.shape,mesh)
        _write(mesh,folder/'mesh.json');result['cad'][member+'_mesh']=(folder/'mesh.json').relative_to(output).as_posix()
        if member=='pinion':
            slot=model.slot_face_groups[0]
            flank=_surface_subset(mesh,set(slot['left_flank']))
            flank['metadata'].update(face_regions={'left_flank':slot['left_flank']},source_slot=0,
                selection='单一实际左工作侧；面号取当前BRep历史映射')
            root_groups={'adjacent_working_flanks':slot['left_flank']+slot['right_flank'],
                'left_fillet':slot['left_fillet'],'right_fillet':slot['right_fillet'],'root_bottom':slot['root_bottom']}
            root=_surface_subset(mesh,{i for ids in root_groups.values() for i in ids})
            root['metadata'].update(face_regions=root_groups,source_slot=0,selection='实际刀尖根过渡、连接槽底及同槽邻接工作面上下文')
            for name,subset in (('flank_mesh',flank),('root_mesh',root)):
                _write(subset,output/'cad'/(name+'.json'));result['cad'][name]='cad/'+name+'.json'
            result['selected_regions']=dict(member='pinion',source_slot=0,left_working_flank_face_indices=slot['left_flank'],root_context_face_groups=root_groups)
            del flank,root
        del mesh,imported;gc.collect();_write(result,output/'geometry_summary.json')

    report('unloaded_tca')
    prepared=prepare_generated_tca(models['pinion'].request,models['wheel'].request,cad_models=models)
    candidates=[solve_generated_tca(prepared,driving_angle_rad,tooth_offset=k,position_tolerance_m=1e-10) for k in tooth_offsets]
    selected=_selected_tca(candidates,sign)
    if selected is None:raise ArithmeticError('没有有效真实相切姿态，不能按理想齿数比补画装配')
    result['unloaded_tca']=dict(driving_angle_rad=driving_angle_rad,selection_torque_sign=sign,candidates=candidates,selected=selected)
    _write(result['unloaded_tca'],output/'unloaded_tca.json')
    assembly=_assembly(models,selected['frames']);report('assembly_export',tooth_offset=selected['tooth_offset'])
    export_model_step(assembly,output/'cad/pair.step');export_model_brep(assembly,output/'cad/pair.brep')
    pair_mesh=triangulate_imported_model(assembly,linear_deflection_m=linear_deflection_m,angular_deflection_rad=angular_deflection_rad)
    if pair_mesh['metadata'].get('solid_count')!=2 or not pair_mesh['metadata'].get('closed_solid'):
        raise ArithmeticError('真实装配没有保留两个有效闭实体')
    _write(pair_mesh,output/'cad/pair_mesh.json');result['cad']['pair_mesh']='cad/pair_mesh.json'
    result['assembly']=dict(step='cad/pair.step',brep='cad/pair.brep',frames=selected['frames'],
        solid_count=2,closed_solid=True,vertices=len(pair_mesh['vertices_m']),triangles=len(pair_mesh['triangles']),
        definition='无载真实相切装配，不构成全轮干涉或承载接触验证')
    del pair_mesh,assembly,prepared,models;gc.collect()
    if plots:
        from bevel_gear.workflow_plotting import plot_workflow_cad
        bundle=dict(schema='bevel_workflow_v1',kind='spiral',case=case,cad=result['cad'])
        for language in languages:
            report('plot_geometry',language=language)
            result['plots'].append(plot_workflow_cad(bundle,output/'figures',lang=language,data_root=output,detail_layout='contact'))
    result.update(geometry_completed=True,source_identity_after=_identity(plots))
    result['source_identity_unchanged']=identity_before==result['source_identity_after']
    if not result['source_identity_unchanged']:result['geometry_completed']=False
    _write(result,output/'geometry_summary.json');report('completed',geometry_completed=result['geometry_completed'])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',nargs='?',type=Path,default=PROJECT/'examples/spiral_17_47_relieved_contact_design.json')
    parser.add_argument('--output',type=Path,required=True,help='新的独立CAD输出目录')
    parser.add_argument('--plots',action='store_true',help='绘制同一制造案例的实际CAD与工作面／根区细节')
    parser.add_argument('--language',choices=('zh','en','both'),default='both',help='科研图语言；默认中英双版')
    parser.add_argument('--driving-angle-rad',type=float,default=0.,help='实际无载相切主动角，单位rad')
    parser.add_argument('--tooth-offsets',type=int,nargs='+',default=[-1,0,1],help='实际相切候选齿号，不使用理想传动比补位姿')
    parser.add_argument('--linear-deflection-m',type=float,default=20e-6,help='仅用于显示三角化的弦差m，不是制造精度')
    parser.add_argument('--angular-deflection-rad',type=float,default=.08,help='显示三角化角度rad')
    args=parser.parse_args(argv)
    result=build_contact_geometry(args.case,args.output,plots=args.plots,languages=('zh','en') if args.language=='both' else (args.language,),
        driving_angle_rad=args.driving_angle_rad,tooth_offsets=tuple(args.tooth_offsets),linear_deflection_m=args.linear_deflection_m,
        angular_deflection_rad=args.angular_deflection_rad,progress=lambda value:print(json.dumps(value,ensure_ascii=False),flush=True))
    return 0 if result['geometry_completed'] else 1


if __name__=='__main__':raise SystemExit(main())
