"""从随附SI三角压力数据重绘四面板总图，无需CAD、Gmsh或重新求解接触。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents
    if (p/'pyproject.toml').is_file() and (p/'bevel_gear').is_dir())
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from bevel_gear.contact_composite import load_triangle_pressure_map
from bevel_gear.contact_display import plot_pressure_comparison as plot_contact_composite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=PROJECT/'examples/results/relieved_contact',
        help='四份三角压力JSON或JSON.GZ及comparison_case.json所在目录')
    parser.add_argument('--output',type=Path,required=True,help='重绘图表及数值输出目录')
    parser.add_argument('--lang',choices=('zh','en','both'),default='both',help='图中文字语言')
    parser.add_argument('--view',choices=('engineering','raw'),default='engineering',
        help='engineering为FE合力守恒恢复/相位插值；raw显示原始离散采样')
    parser.add_argument('--stem',default='17_contact_composite',help='输出图的文件名前缀')
    parser.add_argument('--smoothing-length-um',type=float,help='FE显示恢复物理核宽μm，默认随附源中设置')
    parser.add_argument('--scope',choices=('certified','computed'),default='certified',
        help='certified遮罩未认证域；computed显示计算值并叠加未认证网纹')
    parser.add_argument('--instantaneous-zoom',action='store_true',
        help='增加瞬时压力区的真实mm坐标等比例细节窗，不改变原压力或色标')
    args=parser.parse_args()
    rows=[]
    for method in ('full_gear_fe','local_halfspace'):
        row=dict(method=method)
        for key,kind in (('instantaneous','instantaneous'),('cycle_maximum','sampled_cycle_maximum')):
            name=method+'_'+kind+'_triangle_pressure.json'
            compressed=args.data/(name+'.gz');plain=args.data/name
            if compressed.exists() and plain.exists():
                raise ValueError('同一压力数据同时存在压缩与未压缩版本，请明确保留一份')
            path=compressed if compressed.exists() else plain
            if not path.is_file():raise FileNotFoundError('缺少原三角压力数据：'+str(path))
            # 校验原节点摘要；显示器从原P1三角压力重新采样，不读取缓存颜色。
            row[key]=load_triangle_pressure_map(path,recompute=False)
        rows.append(row)
    evidence_path=args.data/'comparison_case.json'
    evidence=json.loads(evidence_path.read_text(encoding='utf-8-sig')) if evidence_path.is_file() else None
    for language in ('zh','en') if args.lang=='both' else (args.lang,):
        # 绘图器自己创建zh/en；两种语言共用output/data的四份数值源。
        plot_contact_composite(rows,args.output,lang=language,
            display_scope=args.scope,comparison_evidence=evidence,source_compression='gzip',
            instantaneous_zoom=args.instantaneous_zoom,stem=args.stem,view=args.view,
            recovery_source_file=args.data/'native_pressure_recovery_sources.json.gz',
            smoothing_length_m=None if args.smoothing_length_um is None else args.smoothing_length_um*1e-6)
    print('接触压力总图及可复绘数据已保存：'+str(args.output.resolve()))


if __name__=='__main__':main()
