"""从显式双轮制造输入生成真实主轴窄带加密区；不划网、不组装柔度或求压力。"""
from pathlib import Path
import argparse,hashlib,json,math,sys
PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) in sys.path:sys.path.remove(str(PROJECT))
sys.path.insert(0,str(PROJECT))
if str(Path(__file__).resolve().parent) not in sys.path:sys.path.append(str(Path(__file__).resolve().parent))
from run_distributed_fe_contact import _actual_imports,_identity


def _source_identity():
    identity=_identity();identity['driver']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return identity


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_case',nargs='?',type=Path,default=PROJECT/'examples/spiral_17_47_relieved_contact_design.json')
    parser.add_argument('--output',type=Path,required=True,help='纯SI区域JSON；同时保存输入及逐相位几何诊断')
    phases=parser.add_mutually_exclusive_group();phases.add_argument('--angles-rad',type=float,nargs='+')
    phases.add_argument('--intervals',type=int,help='在±半主动齿距之间等分，默认8区间/9相位；不是连续周期覆盖证书')
    parser.add_argument('--tooth-offsets',type=int,nargs='+',default=[-1,0,1])
    parser.add_argument('--half-length-m',type=float,default=.008);parser.add_argument('--half-width-m',type=float,default=.0006)
    parser.add_argument('--long-step-m',type=float,default=.00025);parser.add_argument('--short-step-m',type=float,default=.0001)
    parser.add_argument('--fine-um',type=float,default=75.);parser.add_argument('--core-m',type=float,default=.0003)
    parser.add_argument('--radius-m',type=float,default=.0012);parser.add_argument('--alignment-step-m',type=float,default=.0001)
    parser.add_argument('--strict-alignment',action='store_true',help='主轴差分失败直接停止；默认保留原基且完整记录原因')
    parser.add_argument('--ray-backend',choices=('scalar','batch'),default='batch');args=parser.parse_args(argv)
    raw=args.base_case.read_bytes();base=json.loads(raw.decode('utf-8-sig'))
    if base.get('schema')!='bevel_contact_manufacturing_case_v1':raise ValueError('需要显式制造输入base case')
    manufacturing=base['manufacturing'];intervals=args.intervals if args.intervals is not None else 8
    if intervals<1:raise ValueError('相位等分区间数须为正整数')
    z=manufacturing['pinion']['request']['tooth_count']
    angles=args.angles_rad if args.angles_rad is not None else [-math.pi/z+2*math.pi/z*i/intervals for i in range(intervals+1)]
    if any(not math.isfinite(a) for a in angles) or any(b<=a for a,b in zip(angles,angles[1:])):raise ValueError('源相位须为严格递增的有限rad')
    if len(set(args.tooth_offsets))!=len(args.tooth_offsets):raise ValueError('控制齿对索引不能重复')
    sign=base.get('load',{}).get('ltca_torque_sign',-1)
    output=args.output.resolve();diagnostics=output.with_name(output.stem+'.diagnostics.json');input_path=output.with_name(output.stem+'.inputs.json')
    if any(p.exists() for p in (output,diagnostics,input_path)):raise FileExistsError('已有区域或诊断记录，不能覆写原输入/结果')
    _actual_imports();before=_source_identity();output.parent.mkdir(parents=True,exist_ok=True)
    parameters=dict(driving_angles_rad=angles,tooth_offsets=args.tooth_offsets,half_widths_m=[args.half_length_m,args.half_width_m],
        longitudinal_step_m=args.long_step_m,transverse_step_m=args.short_step_m,refined_size_m=args.fine_um*1e-6,
        refinement_core_radius_m=args.core_m,refinement_radius_m=args.radius_m,working_side_sign=sign,
        alignment_step_m=args.alignment_step_m,alignment_fallback_to_surface_basis=not args.strict_alignment,
        ray_backend=args.ray_backend,verify_surface_points=True)
    def write(value,path):
        temporary=path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8')
        temporary.replace(path)
    def progress(value):print(json.dumps(value,ensure_ascii=False),flush=True)
    write(dict(base_case_file=args.base_case.name,base_case_sha256=hashlib.sha256(raw).hexdigest(),
        manufacturing=manufacturing,cad_sampling=base.get('cad_sampling',{}),parameters=parameters,
        source_identity_before=before,actual_imports_before=_actual_imports()),input_path)
    try:
        from bevel_gear.generated_solid import build_generated_gear
        from bevel_gear.generated_halfspace import prepare_generated_halfspace
        from bevel_gear.contact_refinement import create_contact_refinement
        models={}
        for member in ('pinion','wheel'):
            progress(dict(stage='build_actual_CAD',member=member))
            models[member]=build_generated_gear(manufacturing[member]['request'],manufacturing[member]['blank'],**base.get('cad_sampling',{}))
        prepared=prepare_generated_halfspace(manufacturing['pinion']['request'],manufacturing['wheel']['request'],cad_models=models)
        result=create_contact_refinement(prepared,progress=progress,**parameters)
        after=_source_identity();actual=_actual_imports(before)
        if before!=after:raise RuntimeError('生成控制区期间计算源码改变，不能验收该次输出')
        write(result['document'],output)
        write(dict(**result['diagnostics'],output_regions_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            source_identity_before=before,source_identity_after=after,source_identity_unchanged=True,
            actual_imports=actual,strict_import_binding_verified=True),diagnostics)
        progress(dict(stage='completed',phase_count=len(angles),track_counts=result['diagnostics']['track_counts'],
            pressure_solved=False,actual_loaded_region_coverage_certified=False))
        return 0
    except Exception as error:
        write(dict(all_requested_phases_generated=False,failure=dict(exception_type=type(error).__name__,message=str(error)),
            source_identity_before=before,source_identity_after=_source_identity(),
            pressure_solved=False,actual_loaded_region_coverage_certified=False),diagnostics)
        raise


if __name__=='__main__':raise SystemExit(main())
