#!/usr/bin/env python3
"""Run and plot the four clipped Figure 3 cases on the paper grid."""

import argparse
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path


CASES = (("water", 100), ("water", 230), ("bone", 100), ("bone", 230))
NX = 2000
FINAL_DEPTH = 40.0


def valid_idd(path: Path) -> bool:
    if not path.is_file():
        return False
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    return False
                rows.append((float(fields[0]), float(fields[1])))
    except (OSError, ValueError):
        return False
    return (
        len(rows) == NX
        and math.isclose(rows[-1][0], FINAL_DEPTH, rel_tol=0.0, abs_tol=1.0e-10)
        and all(math.isfinite(x) and math.isfinite(y) for x, y in rows)
    )


def run_case(repo: Path, exe: Path, out_root: Path, material: str, energy: int) -> None:
    case_dir = out_root / f"{material}_{energy}"
    case_dir.mkdir(parents=True, exist_ok=True)
    idd = case_dir / "idd_output.txt"
    complete = case_dir / ".complete"
    if complete.is_file() and valid_idd(idd):
        print(f"Skipping completed case {material} {energy} MeV", flush=True)
        return

    stream_dir = Path("/tmp") / f"bfpn_paper_grid_clip_{material}_{energy}"
    cmd = [
        str(exe), str(NX),
        "--time", str(FINAL_DEPTH),
        "--material", material,
        "--data", str(repo / "BFPn_CPU_Solver" / material),
        "--energy", str(energy),
        "--ny", "80", "--nz", "80", "--ng", "500",
        "--nmu", "20", "--nom", "20",
        "--energy-model", "eq15",
        "--streaming-full",
        "--energy-chunk", "32",
        "--lane-chunk", "262144",
        "--stream-dir", str(stream_dir),
        "--idd-stride", "1",
        "--profile-steps",
    ]
    command_text = shlex.join(cmd) + "\n"
    (case_dir / "command.txt").write_text(command_text, encoding="utf-8")
    complete.unlink(missing_ok=True)
    print("+", command_text, end="", flush=True)

    started = time.time()
    with (case_dir / "run.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=case_dir, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    (case_dir / "elapsed_seconds.txt").write_text(f"{elapsed:.6f}\n", encoding="ascii")
    if result.returncode != 0:
        raise RuntimeError(f"{material} {energy} MeV failed; see {case_dir / 'run.log'}")
    if not valid_idd(idd):
        raise RuntimeError(f"{material} {energy} MeV produced an invalid {idd}")
    complete.write_text("spatial_clipping=enabled\n", encoding="ascii")


def plot(out_root: Path, figure: Path, repo: Path) -> None:
    inputs = [str(out_root / f"{m}_{e}" / "idd_output.txt") for m, e in CASES]
    labels = [f"{energy} MeV" for _, energy in CASES]
    panels = [material.capitalize() for material, _ in CASES]
    cmd = [
        sys.executable, str(repo / "scripts/plot_idd_overlay.py"),
        "-i", *inputs,
        "-l", *labels,
        "--panel", *panels,
        "--normalize",
        "--title", "Eq. 15 IDD on the clipped paper grid",
        "--params", "Nx=2000, Ny=Nz=80, Ng=500, Nmu=Nom=20\nspatial clipping enabled",
        "-o", str(figure),
    ]
    subprocess.run(cmd, cwd=repo, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/bin/bfp_solver")
    parser.add_argument("--outdir", default="results/eq15_paper_grid_clip")
    parser.add_argument(
        "--figure",
        default="results/eq15_paper_grid_clip_water_bone_100_230.png",
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    exe = (repo / args.exe).resolve()
    out_root = (repo / args.outdir).resolve()
    figure = (repo / args.figure).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        if not exe.is_file():
            parser.error(f"solver executable not found: {exe}")
        for material, energy in CASES:
            run_case(repo, exe, out_root, material, energy)
    missing = [str(out_root / f"{m}_{e}" / "idd_output.txt") for m, e in CASES
               if not valid_idd(out_root / f"{m}_{e}" / "idd_output.txt")]
    if missing:
        raise RuntimeError("Cannot plot until all four valid outputs exist: " + ", ".join(missing))
    plot(out_root, figure, repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
