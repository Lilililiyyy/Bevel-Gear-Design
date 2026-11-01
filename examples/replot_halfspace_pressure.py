"""从随附SI三角压力源重绘半解析瞬时/周期最大压力，输出双语科研图。"""
from pathlib import Path
import argparse
import json
import math
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0, str(PROJECT))


def main(argv=None):
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from bevel_gear.contact_composite import load_triangle_pressure_map, build_triangle_pressure_map, _draw_geometry
    from bevel_gear.publication_style import contact_pressure_colormap
    from bevel_gear.workflow_plotting import _paper, _save
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=PROJECT/'examples/results/relieved_contact')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lang', choices=('zh', 'en', 'both'), default='both')
    args = parser.parse_args(argv)
    case = json.loads((args.data/'halfspace_case_summary.json').read_text(encoding='utf8'))
    if case.get('schema') != 'bevel_halfspace_case_summary_v1':
        raise ValueError('需要同案例的bevel_halfspace_case_summary_v1数值摘要')
    kinds = ('instantaneous', 'sampled_cycle_maximum')
    sources = [load_triangle_pressure_map(args.data/('local_halfspace_'+kind+'_triangle_pressure.json.gz'), recompute=False) for kind in kinds]
    peaks = (case['instantaneous']['maximum_pressure_pa'], case['sampled_cycle']['maximum_pressure_pa'])
    if any(not math.isclose(s['source_peak_pressure_pa'], p, rel_tol=1e-12, abs_tol=1e-6) for s,p in zip(sources,peaks)):
        raise ValueError('摘要峰压与原三角压力不符，不能混用不同案例')
    if not sources[0]['outline_geometry'].equals(sources[1]['outline_geometry']):
        raise ValueError('两图必须使用同一参考材料齿面')
    support_area = sources[1]['positive_support_projected_area_m2']
    projected_area = sources[1]['outline_geometry'].area
    if (not math.isclose(case['sampled_cycle']['P1_support_projected_area_m2'], support_area, rel_tol=1e-11) or
            not math.isclose(case['sampled_cycle']['whole_flank_projected_area_m2'], projected_area, rel_tol=1e-11) or
            not math.isclose(case['sampled_cycle']['support_fraction_of_projected_flank'], support_area/projected_area, rel_tol=1e-11)):
        raise ValueError('摘要投影面积与同一压力源的实际支撑/齿面轮廓不符')
    limit = math.ceil(max(peaks)/100e6)*100e6
    levels = np.linspace(0., limit, 66)
    maps = [build_triangle_pressure_map(s['source_fields'], method=s['method'], kind=kind,
        outline_geometry=s['outline_geometry'], levels_pa=levels, unknown_geometry=s['unknown_geometry'],
        certified_zero_geometry=s['certified_zero_geometry'], source_identity=s['source_identity']) for s,kind in zip(sources,kinds)]
    cmap = contact_pressure_colormap(); norm = Normalize(0., limit*1e-6)
    for lang in (('zh', 'en') if args.lang == 'both' else (args.lang,)):
        zh = lang == 'zh'
        with _paper(lang) as plt:
            fig, axes = plt.subplots(2,1)
            fig.subplots_adjust(left=.105, right=.855, bottom=.25, top=.84, hspace=.70)
            names = ((f"瞬时压力：α={case['instantaneous']['driver_angle_rad']:g} rad, k={case['instantaneous']['tooth_offset']}",
                      f"全啮合采样最大压力：{case['sampled_cycle']['state_count']}状态") if zh else
                     (f"Instantaneous: α={case['instantaneous']['driver_angle_rad']:g} rad, k={case['instantaneous']['tooth_offset']}",
                      f"Sampled mesh-cycle maximum: {case['sampled_cycle']['state_count']} states"))
            for index,(ax,data) in enumerate(zip(axes,maps)):
                _draw_geometry(ax,data['certified_zero_geometry'],facecolor=cmap(0),edgecolor='none',antialiased=False)
                for j,geometry in enumerate(data['superlevel_geometries']):
                    _draw_geometry(ax,geometry,facecolor=cmap(norm((levels[j]+levels[j+1])*.5e-6)),
                        edgecolor='none',antialiased=False,rasterized=True)
                _draw_geometry(ax,data['unknown_geometry'],facecolor='#B5BBC0',edgecolor='#6B7278',hatch='////',linewidth=.25)
                _draw_geometry(ax,data['outline_geometry'],facecolor='none',edgecolor='#4c5258',linewidth=.55)
                a,b,c,d = data['outline_geometry'].bounds
                ax.set(xlim=(a*1e3-.1,c*1e3+.1),ylim=(b*1e3-.1,d*1e3+.1));ax.set_aspect('equal')
                ax.set_xlabel(('齿宽位置' if zh else 'Face-width position')+r' $w_\mathrm{f}$ (mm)')
                ax.set_ylabel(('节锥法向高度' if zh else 'Normal height')+'\n'+r'$h$ (mm)')
                ax.set_title(f'({chr(97+index)}) '+names[index]+'   '+r'$p_\mathrm{max}$'+f' = {peaks[index]*1e-6:.1f} MPa',loc='left',fontsize=8)
            fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.895,.20,.018,.63]),
                label=('接触压力' if zh else 'Contact pressure')+r' $p$ (MPa)')
            pair = '/'.join(str(z) for z in case['gear_teeth']);relief = case['normal_tip_relief_m']*1e6
            fig.suptitle((f'局部半解析接触压力｜{pair}弧齿锥齿轮' if zh else f'Local semi-analytical pressure | {pair} spiral bevel gears')+
                '\n'+f"$T_1={case['pinion_torque_nm']:g}$ N·m; "+(f'双轮真实齿顶修缘{relief:g} μm' if zh else f'{relief:g} μm normal tip relief on both gears'),y=.98,fontsize=10)
            area = case['instantaneous']['original_P0_plane_area_m2']*1e6
            support = case['sampled_cycle']['P1_support_projected_area_m2']*1e6
            fraction = case['sampled_cycle']['support_fraction_of_projected_flank']*100
            note = ((f'零压为蓝色；瞬时原P0面积：{area:.4f} mm²。周期显示支撑投影：{support:.3f} mm²（{fraction:.2f}%）。\n'
                     '周期最大值不是同时承载压力；'+r'$w_\mathrm{f}$'+'从外端向内端，h沿节锥法向。') if zh else
                    (f'Blue denotes zero pressure. Original instantaneous P0 area: {area:.4f} mm².\n'
                     f'Swept projected support: {support:.3f} mm² ({fraction:.2f}%); not a simultaneous loaded area.\n'+
                     r'$w_\mathrm{f}$'+' runs from heel to toe; h is normal to the pitch cone.'))
            fig.text(.105,.045,note,fontsize=6.1,va='bottom',linespacing=1.5)
            rows = [dict(method=x['method'],kind=x['kind'],source_peak_pressure_pa=x['source_peak_pressure_pa'],
                source_field_count=x['source_field_count'],positive_support_projected_area_m2=x['positive_support_projected_area_m2'],
                unknown_projected_area_fraction=x['unknown_projected_area_fraction']) for x in maps]
            _save(fig,args.output,'17_halfspace_contact',lang,rows,
                '原SI三角压力的标量等值图；共同线性色阶，原压力与几何未修改。',width_mm=190,height_mm=118,
                extra=dict(method='local_halfspace',pressure_colormap=cmap.name,FE_results_included=False,original_pressure_modified=False))


if __name__ == '__main__':
    main()
