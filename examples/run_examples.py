"""用合成设计输入运行工程案例、完整参数扫描并导出真实计算图。

脚本从项目标记查找独立包，结果全部写入显式指定的目录。
扫描失败点保留在清单中；存在非预期失败时以非零状态退出。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import math
from pathlib import Path
import platform
import sys
import time


def _project_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "bevel_gear").is_dir():
            return candidate
    raise RuntimeError("没有找到包含pyproject.toml和bevel_gear的独立项目目录")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from bevel_gear import analyze_design, sweep_parameter, export_assembly_view
from bevel_gear.io import export_result, read_config, write_json


def case_configs(base):
    """明确各案例的改动；材料、曲线和工况从公开基准配置复制。"""
    cases = {}
    canonical = deepcopy(base)
    canonical["geometry"].update(pinion_teeth=20, wheel_teeth=40, shaft_angle_rad=math.pi / 2,
                                  spiral_angle_rad=0.)
    specifications = (
        ("straight", "正交直齿", {}),
        ("spiral", "正交弧齿", {"spiral_angle_rad": math.radians(35), "cutter_radius_m": .100}),
        ("shaft_75", "75°轴交角", {"shaft_angle_rad": math.radians(75)}),
        ("ratio_18_47", "非整数齿比", {"pinion_teeth": 18, "wheel_teeth": 47}),
    )
    for key, label, changes in specifications:
        config = deepcopy(canonical)
        config["name"] = label
        config["description"] = "合成设计输入；用于计算演示，不代表实测材料或已认证产品"
        config["geometry"].update(changes)
        config["rating"]["required_bending_safety"] = 1.5 if config["geometry"]["spiral_angle_rad"] == 0 else 1.3
        cases[key] = config
    return cases


def _metrics(key, category, result):
    g, f, r = result["geometry"], result["forces"], result["rating"]
    load = result["inputs"]["load"]
    normal = f["normal_force_n"]
    errors = [math.sqrt(sum(f[f"{gear}_{axis}_n"]**2 for axis in ("tangential", "radial", "axial"))) - normal
              for gear in ("pinion", "wheel")]
    return {
        "case": key, "category": category, "pinion_teeth": g["pinion"]["teeth"],
        "wheel_teeth": g["wheel"]["teeth"], "ratio": g["ratio"],
        "shaft_angle_rad": g["inputs"]["shaft_angle_rad"],
        "spiral_angle_rad": g["inputs"]["spiral_angle_rad"],
        "face_width_m": g["inputs"]["face_width_m"],
        "pinion_torque_n_m": load["pinion_torque_n_m"],
        "pinion_mean_pitch_diameter_m": g["pinion"]["mean_pitch_diameter_m"],
        "tangential_force_n": f["pinion_tangential_n"],
        "torque_balance_residual_n_m": f["pinion_tangential_n"] * g["pinion"]["mean_pitch_diameter_m"] / 2 - load["pinion_torque_n_m"],
        "normal_magnitude_residual_n": max(abs(value) for value in errors),
        **{key: r[key] for key in (
            "contact_stress_pa", "pinion_bending_stress_pa", "wheel_bending_stress_pa",
            "pinion_contact_safety", "wheel_contact_safety", "pinion_bending_safety", "wheel_bending_safety",
            "required_contact_safety", "required_bending_safety", "meets_targets")},
    }


def _write_metrics(rows, path):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _relative(files, root):
    return [Path(path).resolve().relative_to(root).as_posix() for path in files]


def run_examples(base, output_dir, *, plots=True):
    """每个设计点重新调用内核；返回带计算状态和相对输出路径的清单。"""
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if plots:
        from bevel_gear.plotting import plot_design, plot_sweep, plot_comparison
    manifest = {"inputs_are_synthetic": True, "model": "零偏置锥齿轮宏观几何、局部力和钢材常幅B1疲劳评估",
                "python_version": platform.python_version(), "cases": [], "sweeps": [], "expected_rejections": [],
                "limitations": ["材料和三点S-N曲线为合成输入", "无物理实验验证或完整标准认证",
                                "几何剖面不表示刀具生成齿形或接触斑", "安全系数应与各自输入目标比较"]}
    metrics, failed, results, labels = [], [], [], []
    configs = case_configs(base)
    for key, config in configs.items():
        target = output / key
        config_path = write_json(config, target / "config.json")
        try:
            result = analyze_design(config)
            files = [str(config_path)] + export_result(result, target) + export_assembly_view(result, target)
            if plots:
                files.extend(plot_design(result, target, title="合成案例：" + config["name"]))
            manifest["cases"].append({"case": key, "label": config["name"], "status": "ok",
                                      "files": _relative(files, output), "diagnostics": result["diagnostics"],
                                      "geometry_diagnostics": result["geometry"]["diagnostics"]})
            metrics.append(_metrics(key, "design", result))
            results.append(result)
            labels.append(config["name"])
            print(f"案例完成：{config['name']}", flush=True)
        except (ValueError, TypeError, KeyError, NotImplementedError) as error:
            manifest["cases"].append({"case": key, "status": "failed", "diagnostic": str(error)})
            failed.append(key)
            print(f"案例失败：{key}，{error}", flush=True)
    if plots and len(results) == len(configs):
        manifest["comparison_files"] = _relative(plot_comparison(results, labels, output), output)

    scans = (
        ("torque_sweep", "load.pinion_torque_n_m", [40., 60., 80., 100., 120., 140., 160., 180., 200.],
         "小轮扭矩 / N·m", "pinion_torque_n_m", 1., "合成直齿设计：扭矩扫描"),
        ("width_sweep", "geometry.face_width_m", [.016, .018, .020, .022, .025, .028, .030, .032],
         "齿宽 / mm", "face_width_mm", 1e3, "合成直齿设计：齿宽扫描"),
    )
    for key, parameter, values, xlabel, x_field, display_scale, title in scans:
        target = output / key
        rows = sweep_parameter(configs["straight"], parameter, values)
        files = [write_json({"base_config": configs["straight"], "points": rows}, target / "points.json")]
        for index, row in enumerate(rows):
            if row["status"] == "ok":
                files.extend(export_result(row["result"], target / f"point_{index:02d}"))
                metrics.append(_metrics(f"{key}_{index:02d}", key, row["result"]))
        invalid = [row for row in rows if row["status"] != "ok"]
        if invalid:
            failed.append(key)
        elif plots:
            files.extend(plot_sweep([row["result"] for row in rows], [value * display_scale for value in values],
                                    target, xlabel=xlabel, x_field=x_field, stem=key, title=title))
        manifest["sweeps"].append({"case": key, "parameter": parameter, "status": "failed" if invalid else "ok",
                                   "point_count": len(rows), "invalid_points": invalid, "files": _relative(files, output)})
        print(f"扫描完成：{key}，{len(rows) - len(invalid)}/{len(rows)}点有效", flush=True)

    for key, updates in (("degenerate_120_20_40", {"shaft_angle_rad": math.radians(120)}),
                          ("nonzero_offset", {"offset_m": .001})):
        config = deepcopy(configs["straight"])
        config["geometry"].update(updates)
        record = {"case": key, "config": config}
        try:
            analyze_design(config)
        except (ValueError, NotImplementedError) as error:
            record.update(status="rejected_as_expected", diagnostic=str(error))
        else:
            record.update(status="unexpectedly_accepted")
            failed.append(key)
        manifest["expected_rejections"].append(record)
    _write_metrics(metrics, output / "case_metrics.csv")
    manifest.update(status="failed" if failed else "ok", failures=failed, result_count=len(metrics),
                    runtime_seconds=time.perf_counter() - started)
    write_json(manifest, output / "manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="运行合成锥齿轮设计案例、逐点扫描和图表导出")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("baseline_example.json"),
                        help="公开SI基准配置；材料和三点曲线必须显式给定")
    parser.add_argument("--output", type=Path, required=True, help="输入、结果、CSV及图表的输出目录")
    parser.add_argument("--skip-plots", action="store_true", help="只计算并导出数据，不加载Matplotlib")
    args = parser.parse_args(argv)
    manifest = run_examples(read_config(args.input), args.output, plots=not args.skip_plots)
    print(f"输出设计点：{manifest['result_count']}，总状态：{manifest['status']}，用时{manifest['runtime_seconds']:.2f}s")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
