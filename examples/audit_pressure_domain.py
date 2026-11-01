"""审核真实完整工作面；仅在原状态绑定的覆盖证书通过后允许完整蓝色零压背景。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path,help='已接受的FE或半解析业务汇总JSON')
    parser.add_argument('--method',choices=('full_gear_fe','local_halfspace'),required=True)
    parser.add_argument('--output',type=Path,required=True,help='覆盖证书和绑定的实际CAD三角化目录')
    parser.add_argument('--phase-index',type=int,default=0)
    parser.add_argument('--tooth-offset',type=int,default=0)
    parser.add_argument('--maximum-depth',type=int,default=8,help='几何区域自适应细分深度上限')
    parser.add_argument('--maximum-regions',type=int,default=300000,help='几何审核区域数上限；未决保留而非填零')
    parser.add_argument('--cad-cache',type=Path,help='含cad子目录的同输入绑定缓存，供两方法共用同一真实背景三角化')
    parser.add_argument('--flank-deflection',type=float,default=1e-6,help='真实工作面三角化弦差m，默认1µm；比粗显示背景更精细')
    parser.add_argument('--allow-partial',action='store_true',help='身份/KKT通过且仅有明确未决微片时允许输出部分证据；仍保留全域passed=false')
    args=parser.parse_args(argv)
    from bevel_gear.pressure_domain_workflow import audit_pressure_workflow
    result=audit_pressure_workflow(args.summary,args.output,method=args.method,phase_index=args.phase_index,
        tooth_offset=args.tooth_offset,maximum_depth=args.maximum_depth,maximum_regions=args.maximum_regions,cad_cache=args.cad_cache,
        flank_linear_deflection_m=args.flank_deflection,progress=lambda event:print(json.dumps(event,ensure_ascii=False),flush=True))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if result['passed'] or args.allow_partial and result['partial_domain_evidence_available'] else 2


if __name__=='__main__':raise SystemExit(main())
