"""从已接受完整FE状态复绘中英接触图；有绑定CAD网格时不重建实体。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from bevel_gear.refined_fe_postprocessing import plot_refined_fe_workflow


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path,help='原refined_fe_summary.json，按相对引用读取有效输入与接受状态')
    parser.add_argument('--output',type=Path,help='图、CSV/VTP及CAD网格目录；默认原结果的plots子目录')
    parser.add_argument('--domain-certificates',type=Path,help='按相位索引字符串组织的完整工作面覆盖证书JSON')
    parser.add_argument('--cad-cache',type=Path,help='含cad子目录的同输入绑定缓存；与审核时的背景一致')
    parser.add_argument('--tooth-offset',type=int,help='实际物理齿对偏移；默认采用案例plot_tooth_offset或0')
    args=parser.parse_args(argv)
    certificates=None if args.domain_certificates is None else json.loads(args.domain_certificates.read_text(encoding='utf-8-sig'))
    result=plot_refined_fe_workflow(args.summary,args.output,domain_certificates=certificates,cad_cache=args.cad_cache,tooth_offset=args.tooth_offset)
    print(json.dumps(dict(schema=result['schema'],figure_count=len(result['figures']),cad_mode=result['cad_mode'],
        manifest='refined_fe_plot_manifest.json'),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
