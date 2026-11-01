"""从显式机床/齿坯输入运行实际弧齿局部半空间接触；默认不运行旧FE。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from bevel_gear.surface_workflow import run_surface_workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', nargs='?', type=Path, default=PROJECT/'examples/spiral_17_47_contact_design_surface_case.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase-index', type=int, action='append', help='只求指定零起始相位，可重复用于子集')
    parser.add_argument('--spacing', type=float, nargs=2, metavar=('DX_M', 'DY_M'), help='覆盖实际接触格距，单位m')
    parser.add_argument('--plots', action='store_true', help='为已接受相位生成真实裁切齿面双语连续压力云图')
    parser.add_argument('--nominal', action='store_true', help='另在nominal目录计算同工况名义几何/强度')
    parser.add_argument('--reference', type=Path, help='仅用于--nominal的同工况几何/强度参照JSON')
    parser.add_argument('--allow-initial-penetration', action='store_true', default=None,
        help='显式允许数值门限内未加载负间隙；超限穿透仍拒绝')
    args = parser.parse_args(argv)
    summary = run_surface_workflow(args.case, args.output, phase_indices=args.phase_index,
        spacing_m=args.spacing, plots=args.plots, nominal=args.nominal, reference=args.reference,
        allow_initial_penetration=args.allow_initial_penetration,
        progress=lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary['analysis_completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
