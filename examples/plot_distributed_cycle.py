"""由原FE状态汇总或随附SI周期数据重绘TE、刚度和分载，不重新求解。"""
from pathlib import Path
import argparse
import json
import sys

PROJECT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))

from bevel_gear.distributed_fe_results import load_distributed_fe_cycle,plot_distributed_fe_cycle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='distributed_fe_summary.json或已汇总的SI周期响应JSON')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lang',choices=('zh','en','both'),default='both')
    args=parser.parse_args();data=json.loads(args.input.read_text(encoding='utf-8-sig'))
    if data.get('schema')=='bevel_distributed_fe_workflow_v1':data=load_distributed_fe_cycle(args.input)
    if data.get('schema')!='bevel_distributed_fe_cycle_response_v1':raise ValueError('需要原分布FE汇总或已汇总SI周期响应')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'18_loaded_cycle_response_source.json').write_text(
        json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    for lang in ('zh','en') if args.lang=='both' else (args.lang,):
        plot_distributed_fe_cycle(data,args.output,lang=lang)
    print('整轮FE周期响应及双语图表已保存：'+str(args.output.resolve()))


if __name__=='__main__':main()
