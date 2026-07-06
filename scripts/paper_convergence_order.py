#!/usr/bin/env python3
"""Compute paper Eq.25 convergence errors from energy_moments.txt outputs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_nxs(text):
    values = [int(item) for item in text.replace(" ", "").split(",") if item]
    if len(values) < 2:
        raise argparse.ArgumentTypeError("provide at least two Nx values")
    if values != sorted(values):
        raise argparse.ArgumentTypeError("Nx values must be sorted coarse-to-fine")
    for a, b in zip(values, values[1:]):
        if b != 2 * a:
            raise argparse.ArgumentTypeError("paper Eq.25 comparison expects adjacent Nx doubling")
    return values


def load_moments(path, ng):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"{path} must have columns: step x_cm g psi1 psi2")
    steps = data[:, 0].astype(int)
    groups = data[:, 2].astype(int)
    if np.any(groups < 0) or np.any(groups >= ng):
        raise ValueError(f"{path} has energy group outside [0, {ng})")
    nsteps = int(np.max(steps))
    out = np.zeros((nsteps, ng, 2), dtype=float)
    out[steps - 1, groups, 0] = data[:, 3]
    out[steps - 1, groups, 1] = data[:, 4]
    return out


def eq25_error(coarse, fine, dx, dg):
    usable_steps = min(coarse.shape[0], fine.shape[0] // 2)
    if usable_steps <= 0:
        raise ValueError("not enough steps to compare coarse and fine outputs")
    coarse_view = coarse[:usable_steps, :, :]
    fine_view = fine[1:2 * usable_steps:2, :, :]
    diff = np.abs(coarse_view - fine_view)
    errors = np.sum(diff, axis=(0, 1)) * dx * dg
    return float(errors[0]), float(errors[1])


def observed_orders(rows):
    for i in range(len(rows) - 1):
        h0 = rows[i]["dx"]
        h1 = rows[i + 1]["dx"]
        for key in ("error1", "error2"):
            e0 = rows[i][key]
            e1 = rows[i + 1][key]
            out_key = key.replace("error", "order")
            rows[i][out_key] = math.log(e0 / e1) / math.log(h0 / h1) if e0 > 0 and e1 > 0 else math.nan
    if rows:
        rows[-1]["order1"] = math.nan
        rows[-1]["order2"] = math.nan


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nx", "dx", "error1", "error2", "order1", "order2"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_plot_label(path):
    text = str(path).lower()
    parts = []
    if "bone" in text:
        parts.append("bone")
    elif "water" in text:
        parts.append("water")
    elif "air" in text:
        parts.append("air")
    name = Path(path).name.lower()
    if "230" in name:
        parts.append("230 MeV")
    elif "100" in name:
        parts.append("100 MeV")
    return " ".join(parts)


def plot(path, rows, reference_order, label=""):
    hs = np.array([row["dx"] for row in rows], dtype=float)
    err1 = np.array([row["error1"] for row in rows], dtype=float)
    err2 = np.array([row["error2"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.loglog(hs, err1, "o-", label="psi1 error")
    ax.loglog(hs, err2, "s-", label="psi2 error")
    finite = err1 > 0
    if np.count_nonzero(finite) >= 2:
        h0 = hs[finite][0]
        e0 = err1[finite][0]
        ax.loglog(hs, e0 * (hs / h0) ** reference_order,
                  "--", label=f"O(h^{reference_order:g}) guide")
    ax.invert_xaxis()
    ax.set_xlabel("depth step dx")
    ax.set_ylabel("psi1 / psi2 absolute error")
    title_prefix = f"{label} " if label else ""
    ax.set_title(f"{title_prefix}psi1 / psi2 Convergence Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main():
    parser = argparse.ArgumentParser(description="Compute paper Eq.25 convergence errors.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nxs", type=parse_nxs, required=True)
    parser.add_argument("--lx", type=float, default=40.0)
    parser.add_argument("--ng", type=int, required=True)
    parser.add_argument("--lg", type=float, default=259.0)
    parser.add_argument("--reference-order", type=float, default=2.0)
    parser.add_argument(
        "--label",
        default=None,
        help="optional title label, for example water or bone; defaults to inferring from --out",
    )
    args = parser.parse_args()

    args.out = args.out.resolve()
    rows = []
    cached = {}
    for nx in args.nxs:
        path = args.out / f"nx_{nx}" / "energy_moments.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        cached[nx] = load_moments(path, args.ng)

    for nx, fine_nx in zip(args.nxs, args.nxs[1:]):
        dx = args.lx / nx
        dg = args.lg / args.ng
        error1, error2 = eq25_error(cached[nx], cached[fine_nx], dx, dg)
        rows.append({
            "nx": nx,
            "dx": dx,
            "error1": error1,
            "error2": error2,
        })

    observed_orders(rows)
    csv_path = args.out / "paper_convergence.csv"
    plot_path = args.out / "paper_convergence_order.png"
    write_csv(csv_path, rows)
    plot_label = args.label if args.label is not None else infer_plot_label(args.out)
    plot(plot_path, rows, args.reference_order, plot_label)

    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    print("nx,dx,error1,error2,order1,order2")
    for row in rows:
        def fmt(value):
            return "nan" if math.isnan(value) else f"{value:.6g}"
        print(
            f"{row['nx']},{row['dx']:.12g},{row['error1']:.6e},"
            f"{row['error2']:.6e},{fmt(row['order1'])},{fmt(row['order2'])}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
