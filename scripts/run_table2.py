#!/usr/bin/env python3
"""Run Table 2 style IDD cases and analyze BP/P90/D90/D20."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


CASES = [
    ("water", 50),
    ("water", 100),
    ("water", 230),
    ("bone", 50),
    ("bone", 100),
    ("bone", 230),
]


def run(cmd, cwd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run and analyze paper Table 2 cases")
    parser.add_argument("--exe", default="./build/bin/bfp_solver", help="solver executable")
    parser.add_argument("--nx", type=int, default=4000, help="x grid count")
    parser.add_argument("--time", type=float, default=40.0, help="final depth")
    parser.add_argument("--outdir", default="table2_runs", help="output directory")
    parser.add_argument(
        "--paper-grid",
        action="store_true",
        help="use 80x80 space and 20x20 angle grid; this needs a chunked/streaming solver on 16 GB GPUs",
    )
    parser.add_argument("--skip-run", action="store_true", help="only analyze existing outputs")
    args = parser.parse_args()

    repo = Path.cwd()
    outdir = repo / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.paper_grid:
        print(
            "Warning: --paper-grid requests Ny=80, Nz=80, Nmu=20, Nom=20. "
            "The current full phase-space GPU layout needs far more than 16 GB; "
            "the solver will stop early with a memory estimate unless a chunked "
            "implementation is added.",
            flush=True,
        )

    summary = outdir / "summary.txt"
    with summary.open("w", encoding="utf-8") as summary_file:
        for material, energy in CASES:
            case_dir = outdir / f"{material}_{energy}"
            case_dir.mkdir(parents=True, exist_ok=True)
            idd_path = case_dir / "idd_output.txt"

            if not args.skip_run:
                cmd = [
                    args.exe,
                    str(args.nx),
                    "--time", str(args.time),
                    "--material", material,
                    "--energy", str(energy),
                ]
                if args.paper_grid:
                    cmd += ["--ny", "80", "--nz", "80", "--nmu", "20", "--nom", "20"]
                run(cmd, repo)
                shutil.copyfile(repo / "idd_output.txt", idd_path)
                if (repo / "dose_output.txt").exists():
                    shutil.copyfile(repo / "dose_output.txt", case_dir / "dose_output.txt")

            result = subprocess.run([
                sys.executable,
                "scripts/analyze_idd.py",
                "-i", str(idd_path),
                "--material", material,
                "--energy", str(energy),
            ], cwd=repo, check=True, capture_output=True, text=True)
            block = f"\n[{material} {energy} MeV]\n{result.stdout}"
            print(block)
            summary_file.write(block)

    print(f"Summary written to {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
