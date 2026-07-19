#!/usr/bin/env python3
"""Compare IDD runs in which exactly one grid family changes."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_idd import analyze, load_idd


def aligned_curves(
    x: np.ndarray,
    y: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = max(float(np.min(x)), float(np.min(x_ref)))
    hi = min(float(np.max(x)), float(np.max(x_ref)))
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError("curves do not overlap on at least two samples")
    xc = x[mask]
    return xc, y[mask], np.interp(xc, x_ref, y_ref)


def relative_l2(x, y, x_ref, y_ref) -> float:
    xc, yc, yr = aligned_curves(x, y, x_ref, y_ref)
    denominator = np.trapezoid(yr * yr, xc)
    if denominator <= 0.0:
        raise ValueError("reference curve has zero L2 norm")
    return math.sqrt(np.trapezoid((yc - yr) ** 2, xc) / denominator)


def relative_linf(x, y, x_ref, y_ref) -> float:
    _, yc, yr = aligned_curves(x, y, x_ref, y_ref)
    return float(np.max(np.abs(yc - yr)) / max(float(np.max(np.abs(yr))), 1.0e-300))


@dataclass(frozen=True)
class Case:
    axis: str
    label: str
    nx: int
    ny: int
    nz: int
    nmu: int
    nom: int
    path: Path

    @property
    def grid(self) -> tuple[int, int, int, int, int]:
        return self.nx, self.ny, self.nz, self.nmu, self.nom


def parse_case(values: list[str], axis: str) -> Case:
    label, nx, ny, nz, nmu, nom, path = values
    grid = [int(nx), int(ny), int(nz), int(nmu), int(nom)]
    if any(value <= 0 for value in grid):
        raise ValueError(f"{label}: grid sizes must be positive")
    return Case(axis, label, *grid, Path(path))


def validate_variant(baseline: Case, variant: Case) -> None:
    changed = {
        "depth": variant.nx != baseline.nx,
        "transverse": (variant.ny, variant.nz) != (baseline.ny, baseline.nz),
        "angular": (variant.nmu, variant.nom) != (baseline.nmu, baseline.nom),
    }
    if not changed[variant.axis]:
        raise ValueError(f"{variant.label}: the requested {variant.axis} grid did not change")
    unexpected = [axis for axis, did_change in changed.items()
                  if axis != variant.axis and did_change]
    if unexpected:
        raise ValueError(
            f"{variant.label}: changes {variant.axis} together with {', '.join(unexpected)}"
        )


def curve_metrics(case: Case) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    x, dose = load_idd(case.path)
    metrics = analyze(x, dose)
    norm = dose / metrics["Dmax"]
    metrics["entrance"] = float(dose[0])
    metrics["norm0"] = float(norm[0])
    return metrics, x, norm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", nargs=7, required=True,
        metavar=("LABEL", "NX", "NY", "NZ", "NMU", "NOM", "IDD"),
    )
    parser.add_argument(
        "--variant", nargs=8, action="append", required=True,
        metavar=("AXIS", "LABEL", "NX", "NY", "NZ", "NMU", "NOM", "IDD"),
        help="AXIS must be depth, transverse, or angular",
    )
    parser.add_argument("--csv", default="results/grid_sensitivity.csv")
    parser.add_argument("--markdown", default="results/grid_sensitivity.md")
    args = parser.parse_args()

    baseline = parse_case(args.baseline, "baseline")
    variants = []
    for values in args.variant:
        axis = values[0]
        if axis not in {"depth", "transverse", "angular"}:
            parser.error(f"unknown variant axis: {axis}")
        variants.append(parse_case(values[1:], axis))
    try:
        for variant in variants:
            validate_variant(baseline, variant)
    except ValueError as exc:
        parser.error(str(exc))

    base_metrics, base_x, base_norm = curve_metrics(baseline)
    rows = []
    for case in [baseline, *variants]:
        metrics, x, norm = curve_metrics(case)
        rows.append({
            "axis": case.axis,
            "label": case.label,
            "nx": case.nx,
            "ny": case.ny,
            "nz": case.nz,
            "nmu": case.nmu,
            "nom": case.nom,
            "entrance": metrics["entrance"],
            "dmax": metrics["Dmax"],
            "norm0": metrics["norm0"],
            "bp": metrics["BP"],
            "p90": metrics["P90"],
            "d90": metrics["D90"],
            "d20": metrics["D20"],
            "norm_l2_vs_baseline": 0.0 if case is baseline else relative_l2(
                x, norm, base_x, base_norm
            ),
            "norm_linf_vs_baseline": 0.0 if case is baseline else relative_linf(
                x, norm, base_x, base_norm
            ),
        })

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(args.markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| axis | label | grid | norm0 | BP cm | normalized L2 | normalized Linf |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        grid = f"{row['nx']} / {row['ny']}x{row['nz']} / {row['nmu']}x{row['nom']}"
        lines.append(
            f"| {row['axis']} | {row['label']} | {grid} | {row['norm0']:.9f} "
            f"| {row['bp']:.4f} | {row['norm_l2_vs_baseline']:.6g} "
            f"| {row['norm_linf_vs_baseline']:.6g} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
