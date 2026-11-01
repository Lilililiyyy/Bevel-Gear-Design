"""批量审核整个啮合周期的压力外域；未知区域保留，原物理解只读。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path, help='已全部接受的完整周期FE或半解析汇总')
    parser.add_argument('--method', required=True, choices=['full_gear_fe', 'local_halfspace'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cad-cache', type=Path, help='同输入绑定的1微米齿面三角化缓存')
    parser.add_argument('--maximum-depth', type=int, default=2, help='保守距离界细分深度，未决不会强制设零')
    parser.add_argument('--geometry-engine', choices=['brep', 'triangles'], default='brep',
        help='实际有限BRep，或需要VTK的双面弦差三角距离加速')
    parser.add_argument('--verify-window', action='store_true', help='同时审核请求窗口外邻齿，输出独立window_coverage.json')
    args = parser.parse_args(argv)
    from bevel_gear.contact_cycle_workflow import audit_mesh_cycle_pressure_workflow
    result = audit_mesh_cycle_pressure_workflow(args.summary, args.output, method=args.method,
        cad_cache=args.cad_cache, maximum_depth=args.maximum_depth,
        geometry_engine=args.geometry_engine,
        verify_window=args.verify_window,
        progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    passed = result['complete'] and (not args.verify_window or result.get('window_coverage_passed') is True)
    return 0 if passed else 2


if __name__ == '__main__': raise SystemExit(main())
