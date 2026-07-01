#!/usr/bin/env python3
"""Analyze IDD curves and compare BP/P90/D90/D20 with paper Table 2."""

import argparse
import math
import sys

import numpy as np


TABLE2 = {
    ("water", 50): {"BP": 2.150, "P90": 2.075, "D90": 2.205, "D20": 2.345},
    ("water", 100): {"BP": 7.560, "P90": 7.418, "D90": 7.671, "D20": 7.945},
    ("water", 230): {"BP": 32.540, "P90": 32.172, "D90": 32.828, "D20": 33.513},
    ("bone", 50): {"BP": 1.360, "P90": 1.319, "D90": 1.402, "D20": 1.488},
    ("bone", 100): {"BP": 4.770, "P90": 4.675, "D90": 4.837, "D20": 5.005},
    ("bone", 230): {"BP": 20.390, "P90": 20.168, "D90": 20.585, "D20": 21.023},
}


def crossing_x(x0, y0, x1, y1, target):
    if y1 == y0:
        return x0
    alpha = (target - y0) / (y1 - y0)
    return x0 + alpha * (x1 - x0)


def find_left_crossing(x, y, peak_idx, target):
    for i in range(peak_idx, 0, -1):
        y0, y1 = y[i - 1], y[i]
        if (y0 - target) * (y1 - target) <= 0 and y0 <= target <= y1:
            return crossing_x(x[i - 1], y0, x[i], y1, target)
    return math.nan


def find_right_crossing(x, y, peak_idx, target):
    for i in range(peak_idx, len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - target) * (y1 - target) <= 0 and y0 >= target >= y1:
            return crossing_x(x[i], y0, x[i + 1], y1, target)
    return math.nan


def analyze(x, dose):
    finite = np.isfinite(x) & np.isfinite(dose)
    x = x[finite]
    dose = dose[finite]
    if x.size == 0:
        raise ValueError("no finite IDD samples")
    if np.max(dose) <= 0:
        raise ValueError("IDD has no positive dose values")

    peak_idx = int(np.argmax(dose))
    dmax = float(dose[peak_idx])
    norm = dose / dmax
    return {
        "BP": float(x[peak_idx]),
        "P90": find_left_crossing(x, norm, peak_idx, 0.9),
        "D90": find_right_crossing(x, norm, peak_idx, 0.9),
        "D20": find_right_crossing(x, norm, peak_idx, 0.2),
        "Dmax": dmax,
    }


def load_idd(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns: x dose")
    return data[:, 0], data[:, 1]


def print_table(metrics, reference):
    print("metric   gpu_cm      paper_cm    diff_cm     rel_diff_pct")
    for name in ("BP", "P90", "D90", "D20"):
        gpu = metrics[name]
        ref = reference.get(name) if reference else None
        if ref is None or math.isnan(gpu):
            print(f"{name:<6} {gpu:10.4f} {'-':>10} {'-':>10} {'-':>14}")
        else:
            diff = gpu - ref
            rel = diff / ref * 100.0
            print(f"{name:<6} {gpu:10.4f} {ref:10.4f} {diff:10.4f} {rel:14.4f}")
    print(f"Dmax   {metrics['Dmax']:10.6g}")


def main():
    parser = argparse.ArgumentParser(description="Analyze IDD BP/P90/D90/D20 metrics")
    parser.add_argument("-i", "--input", default="idd_output.txt", help="IDD file")
    parser.add_argument("--material", choices=["water", "bone"], help="material for Table 2 comparison")
    parser.add_argument("--energy", type=int, choices=[50, 100, 230], help="beam energy for Table 2 comparison")
    args = parser.parse_args()

    x, dose = load_idd(args.input)
    metrics = analyze(x, dose)
    reference = None
    if args.material and args.energy:
        reference = TABLE2[(args.material, args.energy)]
    elif args.material or args.energy:
        print("--material and --energy must be provided together for Table 2 comparison", file=sys.stderr)
        return 2

    print_table(metrics, reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
