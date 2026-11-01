"""从随附SI标量CSV重绘原生FE总受载TE和从动轴切线刚度，不重新求解。"""
from pathlib import Path
import argparse
import csv
import hashlib
import math
import sys


PROJECT = next(p for p in Path(__file__).resolve().parents
    if (p/'pyproject.toml').is_file() and (p/'bevel_gear').is_dir())
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from bevel_gear.publication_style import BLUE, RED
from bevel_gear.workflow_plotting import _paper, _save


NUMERIC = ('driving_angle_rad', 'pinion_to_wheel_ratio', 'loaded_follower_angle_rad',
    'reference_follower_angle_rad', 'last_linearization_step_rad', 'total_loaded_te_rad',
    'follower_tangent_stiffness_nm_per_rad', 'minimum_piece_gap_m')
BOOLS = ('original_accepted', 'original_loaded_geometry_check_passed')


def load_source(path):
    """只验证CSV标量与原角度簿记；不把源数据验收标记改成True。"""
    raw = path.read_bytes()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2: raise ValueError('至少需要两个真实相位；缺值不得补点')
    for row in rows:
        index = row['phase_index']
        row['phase_index'] = int(index)
        if str(row['phase_index']) != index or row['phase_index'] < 0: raise ValueError('相位索引须为非负整数')
        for name in NUMERIC:
            row[name] = float(row[name])
            if not math.isfinite(row[name]): raise ValueError(name+'存在未知或非有限值，停止绘图')
        for name in BOOLS:
            value = row[name].lower()
            if value not in ('true', 'false'): raise ValueError('原验收标记须为布尔值')
            row[name] = value == 'true'
        if row['pinion_to_wheel_ratio'] <= 0 or row['follower_tangent_stiffness_nm_per_rad'] <= 0:
            raise ValueError('传动比或从动轴切线刚度非正')
        direct = row['loaded_follower_angle_rad']+row['pinion_to_wheel_ratio']*row['driving_angle_rad']
        if not math.isclose(direct, row['total_loaded_te_rad'], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('总TE与实际加载角／理想运动定义不符')
        if not math.isclose(row['reference_follower_angle_rad']+row['last_linearization_step_rad'],
                row['loaded_follower_angle_rad'], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('末步参考角与实际加载角不符')
    rows.sort(key=lambda row: row['driving_angle_rad'])
    if (len({r['phase_index'] for r in rows}) != len(rows)
            or len({r['driving_angle_rad'] for r in rows}) != len(rows)
            or len({r['pinion_to_wheel_ratio'] for r in rows}) != 1):
        raise ValueError('相位／角度重复或传动比不一致')
    return rows, hashlib.sha256(raw).hexdigest()


def plot_source(rows, output, lang, source_sha256):
    """总TE保留原绝对值；切线刚度为固定本态几何和承载活动集的量。"""
    zh = lang == 'zh'; x = [r['driving_angle_rad']*180/math.pi for r in rows]
    with _paper(lang) as plt:
        fig, axes = plt.subplots(1, 2)
        fig.subplots_adjust(left=.105, right=.985, top=.86, bottom=.32, wspace=.42)
        axes[0].plot(x, [r['total_loaded_te_rad']*180/math.pi*3600 for r in rows],
            'o-', color=BLUE, linewidth=1.15, markersize=3.4, markeredgewidth=.4, markeredgecolor='white')
        axes[1].plot(x, [r['follower_tangent_stiffness_nm_per_rad']/1000 for r in rows],
            's-', color=RED, linewidth=1.15, markersize=3.3, markeredgewidth=.4, markeredgecolor='white')
        axes[0].set_ylabel(('总受载传动误差' if zh else 'Total loaded transmission error')+'\n'+r'$\mathrm{TE}_{\mathrm{tot}}$ (arcsec)')
        axes[1].set_ylabel(('从动轴切线刚度' if zh else 'Follower tangent stiffness')+'\n'+r'$K_{\theta_2}$ (kN'+r'$\cdot$'+'m/rad)')
        for letter, ax in zip('ab', axes):
            ax.set_xlabel(('主动轮增量角' if zh else 'Pinion increment angle')+' '+r'$\theta_1$ (deg)')
            ax.text(-.20, 1.10, letter, transform=ax.transAxes, fontsize=10, weight='bold', va='top')
            ax.grid(axis='y', color='#DCE1E5', linewidth=.45); ax.set_axisbelow(True); ax.margins(x=.035, y=.12)
        fig.suptitle(('整轮有限元受载响应' if zh else 'Sampled full-gear FE response')+f', N = {len(rows)}', y=.985, fontsize=9)
        note = ('TE为实际从动角相对理想传动的总偏差，未分解纯弹性项。\n刚度为本态固定几何／承载活动集的切线；连线仅连接真实计算点。' if zh else
            'TE is the total loaded deviation from ideal motion; no elastic-only split.\nStiffness is tangent to the fixed geometry / active set.\nLines join computed samples only.')
        fig.text(.105, .035, note, fontsize=6.5, color='#40474C', va='bottom', linespacing=1.4)
        caption = ('同一原生FE计算的离散标量响应。总TE=实际从动角+(z1/z2)主动角；未求卸载FE相切，故不分解纯弹性TE。'
            '从动轴切线刚度=固定几何／承载活动集的l_AᵀC_AA⁻¹l_A，SI单位N·m/rad；显示单位kN·m/rad。'
            '连线只连接真实样本，不平滑、不补点、不代表相位或网格收敛；原accepted标记保留于源CSV，不被绘图器改写。')
        return _save(fig, output, '18_native_te_stiffness', lang, rows, caption,
            width_mm=183, height_mm=82, extra=dict(source_si_csv_sha256=source_sha256,
                sample_count=len(rows), geometry_source='FE_boundary',
                original_accepted_count=sum(r['original_accepted'] for r in rows),
                original_accepted_not_changed=True, no_smoothing_or_missing_phase_fill=True,
                temporal_or_spatial_convergence_claimed=False,
                units=dict(angles='rad', stiffness='N·m/rad', gap='m')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
        default=PROJECT/'examples/results/relieved_contact/native_te_stiffness_SI.csv',
        help='随附SI标量CSV，无需CAD或有限元求解依赖')
    parser.add_argument('--output', type=Path, required=True, help='重绘图表及SI数据目录')
    parser.add_argument('--lang', choices=('zh', 'en', 'both'), default='both')
    args = parser.parse_args(); rows, source_sha256 = load_source(args.data)
    for lang in ('zh', 'en') if args.lang == 'both' else (args.lang,):
        plot_source(rows, args.output, lang, source_sha256)
    print('原生FE总TE／从动轴切线刚度及SI源数据已重绘：'+str(args.output.resolve()))


if __name__ == '__main__': main()
