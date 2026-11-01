"""将完整啮合周期的实际结果映射到同一参考齿面，输出最大压力包络。"""
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
    parser.add_argument('--fe-summary', type=Path, help='完整一齿距FE汇总，含重复端点')
    parser.add_argument('--surface-summary', type=Path, help='完整一齿距半解析汇总，含重复端点')
    parser.add_argument('--fe-domain-manifest', type=Path, help='FE逐相位/齿对压力域证书清单')
    parser.add_argument('--surface-domain-manifest', type=Path, help='半解析逐相位/齿对压力域证书清单')
    parser.add_argument('--cad-cache', type=Path, help='同请求/齿坯绑定的实际CAD显示缓存')
    parser.add_argument('--window-coverage', type=Path, action='append', help='周期外邻齿独立审核证据，可分别提供两方法文件')
    parser.add_argument('--spacing-m', type=float, default=2e-5, help='最大值显示采样格距m，不改变求解网格')
    parser.add_argument('--shared-pressure-scale', action='store_true', help='两方法使用共同绝对色标；默认各自明确标注MPa的线性色标')
    parser.add_argument('--output', type=Path, required=True, help='双语图、压缩SI数组及身份清单目录')
    args = parser.parse_args(argv)
    if args.fe_summary is None and args.surface_summary is None:
        parser.error('至少提供--fe-summary或--surface-summary')
    from bevel_gear.contact_cycle_workflow import plot_mesh_cycle_workflows
    manifest = plot_mesh_cycle_workflows(args.output, fe_summary=args.fe_summary,
        surface_summary=args.surface_summary, fe_domain_manifest=args.fe_domain_manifest,
        surface_domain_manifest=args.surface_domain_manifest, cad_cache=args.cad_cache,
        spacing_m=args.spacing_m, window_coverage=args.window_coverage, shared_pressure_scale=args.shared_pressure_scale)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
