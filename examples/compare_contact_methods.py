"""读取同几何/材料/相位/载荷的已接受结果，重绘双语连续压力比较图。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from bevel_gear.refined_fe_postprocessing import compare_contact_workflows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fe-summary',type=Path,required=True,help='整轮FE的refined_fe_summary.json')
    parser.add_argument('--surface-summary',type=Path,required=True,help='半解析surface_summary.json')
    parser.add_argument('--output',type=Path,required=True,help='双语图、源CSV及绑定CAD缓存目录')
    parser.add_argument('--phase-index',type=int,default=0,help='FE业务内的零起始相位索引')
    parser.add_argument('--tooth-offset',type=int,default=0,help='实际物理齿对偏移k')
    parser.add_argument('--fe-domain-certificate',type=Path,help='整轮FE完整工作面覆盖审核生成的证书JSON')
    parser.add_argument('--surface-domain-certificate',type=Path,help='半解析完整工作面覆盖审核生成的证书JSON')
    parser.add_argument('--cad-cache',type=Path,help='含cad子目录的同输入绑定缓存；完整蓝底须与覆盖审核使用同一背景')
    args=parser.parse_args(argv)
    def certificate(path):
        return None if path is None else json.loads(path.read_text(encoding='utf-8-sig'))
    result=compare_contact_workflows(args.fe_summary,args.surface_summary,args.output,phase_index=args.phase_index,
        tooth_offset=args.tooth_offset,fe_domain_certificate=certificate(args.fe_domain_certificate),
        surface_domain_certificate=certificate(args.surface_domain_certificate),cad_cache=args.cad_cache)
    print(json.dumps(dict(completed=True,figures=len(result['figures']),cad_mode=result['cad_mode'],
        physics_resolved=False,manifest='contact_comparison_manifest.json'),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
