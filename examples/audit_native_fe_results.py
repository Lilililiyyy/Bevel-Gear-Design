"""正式原生FE结果的零K审核和压力源导出；按response/geometry/domain/maps/cycle分阶段。"""
from pathlib import Path
import argparse
import json
import os
import sys
import threading

PROJECT=next((p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file() and (p/'bevel_gear').is_dir()),None)
if PROJECT is not None:sys.path.insert(0,str(PROJECT))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('all','response','geometry','domain','maps','cycle','comparison'),default='all');p.add_argument('--include-cycle',action='store_true')
    p.add_argument('--phase-indices',type=int,nargs='+');p.add_argument('--boundary-conditions',type=Path)
    p.add_argument('--pressure-levels',type=Path);p.add_argument('--instantaneous-phase-index',type=int);p.add_argument('--instantaneous-tooth-offset',type=int,default=0)
    p.add_argument('--source-deflection-m',type=float,default=1e-6);p.add_argument('--target-deflection-m',type=float,default=1e-6)
    p.add_argument('--base-depth',type=int,default=2);p.add_argument('--material-depth',type=int,default=3);p.add_argument('--maximum-regions',type=int,default=150000)
    p.add_argument('--maximum-rss-gib',type=float,default=3.);p.add_argument('--minimum-available-gib',type=float,default=10.)
    args=p.parse_args(argv)
    import math
    if any(not math.isfinite(v) or v<=0 for v in (args.maximum_rss_gib,args.minimum_available_gib)):raise ValueError('资源门限须为正有限GiB')
    try:import psutil
    except ImportError as error:raise ImportError('后处理资源监测需要psutil') from error
    def sample():
        memory=psutil.Process().memory_info();available=psutil.virtual_memory().available
        peak=getattr(memory,'peak_wset',memory.rss)
        if peak>args.maximum_rss_gib*2**30 or available<args.minimum_available_gib*2**30:raise MemoryError('后处理RSS/主机余量超出显式门限；不产生通过证明')
        return dict(rss_bytes=memory.rss,host_available_bytes=available)
    sample();failure=[];stop=threading.Event()
    # 只终止本后处理进程；没有K或因子需要清理。原生调用内瞬时峰仍需外部监控。
    def watch():
        while not stop.wait(.25):
            try:sample()
            except Exception as error:
                failure.append(error);args.output.mkdir(parents=True,exist_ok=True)
                (args.output/'resource_failure.json').write_text(json.dumps(dict(schema='bevel_native_FE_postprocess_resource_failure_v1',execution_completed=False,
                    stage=args.stage,error_type=type(error).__name__,message=str(error)),ensure_ascii=False)+'\n',encoding='utf8')
                os._exit(74)
    def progress(event):
        if failure:raise failure[0]
        print(json.dumps(dict(event,resource=sample()),ensure_ascii=False),flush=True)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    from bevel_gear.native_results_workflow import audit_native_fe_results
    conditions=None if args.boundary_conditions is None else json.loads(args.boundary_conditions.read_text(encoding='utf-8-sig'))
    levels=None if args.pressure_levels is None else json.loads(args.pressure_levels.read_text(encoding='utf-8-sig'))
    try:
        result=audit_native_fe_results(args.summary,args.output,stage=args.stage,phase_indices=args.phase_indices,include_cycle=args.include_cycle,
            boundary_conditions=conditions,source_deflection_m=args.source_deflection_m,target_deflection_m=args.target_deflection_m,
            base_depth=args.base_depth,material_depth=args.material_depth,maximum_regions=args.maximum_regions,levels_pa=levels,
            instantaneous_phase_index=args.instantaneous_phase_index,instantaneous_tooth_offset=args.instantaneous_tooth_offset,progress=progress)
        if failure:raise failure[0]
        print(json.dumps(dict(completed_stages=list(result['stages']),cycle_maximum_generated=result['real_cycle_maximum_generated'],resource=sample(),stiffness_assembled=False),ensure_ascii=False),flush=True)
    finally:stop.set();thread.join()
    return 0


if __name__=='__main__':raise SystemExit(main())
