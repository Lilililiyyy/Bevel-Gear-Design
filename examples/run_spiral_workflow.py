"""以同一配置运行直齿/弧齿几何、强度、模型、TCA与LTCA；可按阶段停止。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from bevel_gear.workflow import analyze_bevel_case, load_bevel_case, STAGES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',nargs='?',type=Path,default=PROJECT/'examples/spiral_17_47_case.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--through',choices=STAGES,default='ltca')
    parser.add_argument('--plots',action='store_true',help='导出已完成阶段的中文和英文科研图')
    parser.add_argument('--render3d',action='store_true',help='实际CAD的中文和英文VTK预览')
    parser.add_argument('--reference',type=Path,help='同工况几何/强度数值参照JSON，不调用外部软件')
    args=parser.parse_args()
    reference=None if args.reference is None else json.loads(args.reference.read_text(encoding='utf-8-sig'))
    result=analyze_bevel_case(load_bevel_case(args.case),args.output,through=args.through,reference=reference,
        render3d=args.render3d,progress=lambda value:print(json.dumps(value,ensure_ascii=False),flush=True))
    if args.plots:
        from bevel_gear.workflow_plotting import plot_spiral_workflow
        for language in ('zh','en'):
            plot_spiral_workflow(result,args.output/'figures',lang=language,data_root=args.output,
                                 require_all=args.through=='ltca' and reference is not None)
    print(json.dumps(dict(analysis_completed=result['analysis_completed'],stages=result['completed_stages'],
        design_meets_stress_targets=result['quality'].get('design_meets_stress_targets'),
        reference_comparison=result.get('masta_benchmark',{}).get('comparison_passed')),ensure_ascii=False,indent=2))
    return 0 if result['analysis_completed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
