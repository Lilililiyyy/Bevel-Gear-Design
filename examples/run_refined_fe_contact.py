"""从显式制造输入冷启动整轮高分辨率节点FE接触；不依赖私有缓存。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from bevel_gear.refined_fe_workflow import run_refined_fe_workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',nargs='?',type=Path,default=PROJECT/'examples/spiral_17_47_contact_design_refined_fe_case.json',help='相对base_case绑定制造、齿坯和材料的JSON案例')
    parser.add_argument('--output',type=Path,required=True,help='结果、完整状态、网格与CSV输出目录')
    parser.add_argument('--phase-index',type=int,action='append',help='只求指定零起始相位；可重复')
    parser.add_argument('--mesh-size',type=float,metavar='H_M',help='覆盖局部网格控制尺寸，单位m；200/100/75μm分别用0.0002/0.0001/0.000075')
    parser.add_argument('--torque-nm',type=float,help='覆盖有符号小轮扭矩，单位Nm；默认-64')
    parser.add_argument('--no-mesh-export',action='store_true',help='不另写网格JSON；保留计算输出与网格准备诊断')
    parser.add_argument('--include-fields',action='store_true',help='在完整状态保存两轮位移等原始FE场，文件可能很大')
    parser.add_argument('--plots',action='store_true',help='计算后输出中英完整齿面/接触斑点图，并缓存同请求实际CAD网格')
    args=parser.parse_args(argv)
    plot_enabled=args.plots
    extra={};models=None
    if plot_enabled:
        # 先完整校验实际覆盖参数，再创建一次两轮实体供FE与绘图共同使用。
        from bevel_gear.refined_fe_workflow import load_refined_fe_case,_configuration
        from bevel_gear.generated_solid import build_generated_gear
        loaded=load_refined_fe_case(args.case)
        effective=_configuration(loaded,args.phase_index,args.mesh_size,args.torque_nm)
        manufacturing=loaded['base_case']['manufacturing']
        models={m:build_generated_gear(manufacturing[m]['request'],manufacturing[m]['blank'],**effective['cad_sampling'])
            for m in ('pinion','wheel')}
        extra['cad_models']=models
    summary=run_refined_fe_workflow(args.case,args.output,phase_indices=args.phase_index,
        mesh_size_m=args.mesh_size,pinion_torque_nm=args.torque_nm,export_meshes=not args.no_mesh_export,
        include_fields=args.include_fields,progress=lambda item:print(json.dumps(item,ensure_ascii=False),flush=True),**extra)
    if plot_enabled and any(state.get('accepted') for state in summary.get('states',[])):
        from bevel_gear.refined_fe_postprocessing import plot_refined_fe_workflow
        plot_refined_fe_workflow(args.output/'refined_fe_summary.json',cad_models=models)
        print(json.dumps(dict(plot_manifest='plots/refined_fe_plot_manifest.json'),ensure_ascii=False),flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if summary['analysis_completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
