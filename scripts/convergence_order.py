#!/usr/bin/env python3
"""Run depth-step convergence tests and plot observed order."""

import argparse
import csv
import math
import os
import shutil
import select
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_nxs(text):
    values = [int(item) for item in text.replace(" ", "").split(",") if item]
    if len(values) < 3:
        raise argparse.ArgumentTypeError("provide at least three Nx values")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Nx values must be positive")
    if values != sorted(values):
        raise argparse.ArgumentTypeError("Nx values must be sorted coarse-to-fine")
    return values


def load_curve(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: x value")
    x = data[:, 0]
    y = data[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    order = np.argsort(x)
    return x[order], y[order]


def relative_l2(x, y, x_ref, y_ref):
    lo = max(float(np.min(x)), float(np.min(x_ref)))
    hi = min(float(np.max(x)), float(np.max(x_ref)))
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError("curves do not overlap on at least two samples")

    xc = x[mask]
    yc = y[mask]
    yr = np.interp(xc, x_ref, y_ref)
    diff = yc - yr
    numerator = np.trapezoid(diff * diff, xc)
    denominator = np.trapezoid(yr * yr, xc)
    if denominator <= 0:
        raise ValueError("reference curve has zero L2 norm")
    return math.sqrt(numerator / denominator)


def relative_linf(x, y, x_ref, y_ref):
    lo = max(float(np.min(x)), float(np.min(x_ref)))
    hi = min(float(np.max(x)), float(np.max(x_ref)))
    mask = (x >= lo) & (x <= hi)
    xc = x[mask]
    yc = y[mask]
    yr = np.interp(xc, x_ref, y_ref)
    scale = max(float(np.max(np.abs(yr))), 1e-300)
    return float(np.max(np.abs(yc - yr)) / scale)


def run_logged_command(cmd, cwd, log_path, nx, heartbeat_seconds):
    start = time.monotonic()
    last_status = start
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError("failed to capture subprocess output")

        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        pending = ""
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                chunk = proc.stdout.read()
                if chunk:
                    pending += chunk
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        text = f"[Nx={nx}] {line}"
                        print(text, flush=True)
                        log.write(line + "\n")
                        log.flush()

            rc = proc.poll()
            now = time.monotonic()
            if rc is not None:
                rest = proc.stdout.read()
                if rest:
                    pending += rest
                if pending:
                    text = f"[Nx={nx}] {pending.rstrip()}"
                    print(text, flush=True)
                    log.write(pending)
                    if not pending.endswith("\n"):
                        log.write("\n")
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, cmd)
                elapsed = now - start
                print(f"[Nx={nx}] finished in {elapsed:.1f}s", flush=True)
                return

            if heartbeat_seconds > 0 and now - last_status >= heartbeat_seconds:
                elapsed = now - start
                print(f"[Nx={nx}] still running, elapsed {elapsed:.0f}s", flush=True)
                last_status = now


def run_case(args, nx, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.exe),
        str(nx),
        "--time",
        str(args.time),
        "--material",
        args.material,
        "--energy",
        str(args.energy),
        "--ny",
        str(args.ny),
        "--nz",
        str(args.nz),
        "--ng",
        str(args.ng),
        "--nmu",
        str(args.nmu),
        "--nom",
        str(args.nom),
        "--energy-model",
        args.energy_model,
    ]
    if args.primary_only:
        cmd.append("--primary-only")
    if args.eq15_straggling:
        cmd.append("--eq15-straggling")
    if args.save_energy_moments:
        cmd.append("--save-energy-moments")
    if args.energy_only:
        cmd.append("--energy-only")
    if args.no_transport:
        cmd.append("--no-transport")
    if args.no_angle:
        cmd.append("--no-angle")
    if args.no_spatial_clipping:
        cmd.append("--no-spatial-clipping")
    if args.extra:
        cmd.extend(args.extra)

    with (case_dir / "command.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")
    run_logged_command(cmd, args.repo, case_dir / "run.log", nx, args.heartbeat)

    src = args.repo / "idd_output.txt"
    if not src.exists():
        raise FileNotFoundError(f"{src} was not produced")
    dst = case_dir / "idd_output.txt"
    shutil.copy2(src, dst)
    moments = args.repo / "energy_moments.txt"
    if args.save_energy_moments:
        if not moments.exists():
            raise FileNotFoundError(f"{moments} was not produced")
        shutil.copy2(moments, case_dir / "energy_moments.txt")
    return dst


def observed_orders(rows):
    for i in range(len(rows) - 1):
        e0 = rows[i]["rel_l2"]
        e1 = rows[i + 1]["rel_l2"]
        h0 = rows[i]["h"]
        h1 = rows[i + 1]["h"]
        if e0 > 0 and e1 > 0 and h0 != h1:
            rows[i]["order_to_next"] = math.log(e0 / e1) / math.log(h0 / h1)
        else:
            rows[i]["order_to_next"] = math.nan
    if rows:
        rows[-1]["order_to_next"] = math.nan


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["nx", "h", "rel_l2", "rel_linf", "order_to_next"],
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
    hs = np.array([row["h"] for row in rows], dtype=float)
    errs = np.array([row["rel_l2"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.loglog(hs, errs, "o-", label="IDD relative L2 error")
    finite = errs > 0
    if np.count_nonzero(finite) >= 2:
        h0 = hs[finite][0]
        e0 = errs[finite][0]
        ref = e0 * (hs / h0) ** reference_order
        ax.loglog(hs, ref, "--", label=f"O(h^{reference_order:g}) guide")
    ax.invert_xaxis()
    ax.set_xlabel("depth step h = Lx / Nx")
    ax.set_ylabel("IDD relative L2 error")
    title_prefix = f"{label} " if label else ""
    ax.set_title(f"{title_prefix}IDD Convergence Error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main():
    parser = argparse.ArgumentParser(
        description="Run BFPn depth-step convergence and plot observed order."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--exe", type=Path, default=Path("build/bin/bfp_solver"))
    parser.add_argument("--out", type=Path, default=Path("results/convergence_strang"))
    parser.add_argument("--nxs", type=parse_nxs, default=parse_nxs("500,1000,2000,4000"))
    parser.add_argument("--time", type=float, default=40.0)
    parser.add_argument("--lx", type=float, default=40.0)
    parser.add_argument("--material", default="water", choices=["water", "bone", "air"])
    parser.add_argument("--energy", type=float, default=230.0)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--ng", type=int, default=500)
    parser.add_argument("--nmu", type=int, default=20)
    parser.add_argument("--nom", type=int, default=20)
    parser.add_argument("--energy-model", default="eq15", choices=["eq15", "legacy"])
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--eq15-straggling", action="store_true")
    parser.add_argument("--save-energy-moments", action="store_true")
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--no-transport", action="store_true")
    parser.add_argument("--no-angle", action="store_true")
    parser.add_argument("--no-spatial-clipping", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="reuse existing case outputs")
    parser.add_argument("--reference-order", type=float, default=2.0)
    parser.add_argument(
        "--label",
        default=None,
        help="optional title label, for example water or bone; defaults to inferring from --out",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="print an elapsed-time heartbeat while a case is running; 0 disables it",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="extra solver arguments after --, for example -- --sigma-yz 0.3",
    )
    args = parser.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]

    args.repo = args.repo.resolve()
    args.exe = args.exe if args.exe.is_absolute() else args.repo / args.exe
    args.out = args.out if args.out.is_absolute() else args.repo / args.out
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.exe.exists() and not args.skip_run:
        print(f"executable not found: {args.exe}", file=sys.stderr)
        return 2

    curve_paths = {}
    for nx in args.nxs:
        case_dir = args.out / f"nx_{nx}"
        if args.skip_run:
            path = case_dir / "idd_output.txt"
            if not path.exists():
                print(f"missing existing output: {path}", file=sys.stderr)
                return 2
        else:
            print(f"running Nx={nx} ...", flush=True)
            path = run_case(args, nx, case_dir)
        curve_paths[nx] = path

    ref_nx = args.nxs[-1]
    x_ref, y_ref = load_curve(curve_paths[ref_nx])
    rows = []
    for nx in args.nxs[:-1]:
        x, y = load_curve(curve_paths[nx])
        rows.append(
            {
                "nx": nx,
                "h": args.lx / nx,
                "rel_l2": relative_l2(x, y, x_ref, y_ref),
                "rel_linf": relative_linf(x, y, x_ref, y_ref),
            }
        )
    observed_orders(rows)

    csv_path = args.out / "convergence.csv"
    plot_path = args.out / "convergence_order.png"
    write_csv(csv_path, rows)
    plot_label = args.label if args.label is not None else infer_plot_label(args.out)
    plot(plot_path, rows, args.reference_order, plot_label)

    print(f"reference Nx: {ref_nx}")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    print("nx,h,rel_l2,rel_linf,order_to_next")
    for row in rows:
        order = row["order_to_next"]
        order_text = "nan" if math.isnan(order) else f"{order:.6g}"
        print(
            f"{row['nx']},{row['h']:.12g},{row['rel_l2']:.6e},"
            f"{row['rel_linf']:.6e},{order_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
