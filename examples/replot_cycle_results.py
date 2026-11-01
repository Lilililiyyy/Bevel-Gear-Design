"""读取显式给定的原格式SI周期数组，重绘采样最大压力，不重新求解接触。"""
from pathlib import Path
import argparse
import sys

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from bevel_gear.contact_cycle_plotting import load_mesh_cycle_pressure_data, plot_mesh_cycle_pressure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="调用者实际计算输出中原格式周期JSON/NPZ所在目录")
    parser.add_argument("--output", type=Path, required=True, help="重绘输出目录")
    args = parser.parse_args()
    data = [load_mesh_cycle_pressure_data(args.data / (name + "_cycle_pressure.json"))
            for name in ("full_gear_fe", "local_halfspace")]
    plot_mesh_cycle_pressure(data, args.output, lang="zh")
    print("周期压力图已保存：" + str(args.output.resolve()))


if __name__ == "__main__":
    main()
