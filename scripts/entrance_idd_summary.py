#!/usr/bin/env python3
"""Summarize normalized entrance IDD values for solver and paper figure reference."""

import argparse
import csv
from pathlib import Path

import numpy as np


CASES = [
    ("water", 100),
    ("water", 230),
    ("bone", 100),
    ("bone", 230),
]


def load_normalized(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    x = data[:, 0]
    y = data[:, 1]
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    ymax = float(np.max(y))
    if ymax > 0.0:
        y = y / ymax
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/table2_validation/entrance_idd_summary.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--positions",
        default="0,0.5,1.0",
        help="Comma-separated entrance depths in cm",
    )
    args = parser.parse_args()

    positions = [float(value) for value in args.positions.split(",")]
    rows = []
    for material, energy in CASES:
        sx, sy = load_normalized(Path(f"results/eq15_strict_fine_{material}_{energy}/idd_output.txt"))
        rx, ry = load_normalized(Path(f"results/fluka_fig3_reference/fluka_fig3_{material}_{energy}.txt"))
        for x_cm in positions:
            solver = float(np.interp(x_cm, sx, sy))
            reference = float(np.interp(x_cm, rx, ry))
            rows.append(
                {
                    "material": material,
                    "energy_MeV": energy,
                    "x_cm": x_cm,
                    "solver_norm_idd": solver,
                    "paper_ref_norm_idd": reference,
                    "diff": solver - reference,
                    "rel_diff_percent": (solver - reference) / reference * 100.0,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    for row in rows:
        print(
            f"{row['material']}{row['energy_MeV']} x={row['x_cm']:.1f} cm "
            f"solver={row['solver_norm_idd']:.6f} "
            f"paper_ref={row['paper_ref_norm_idd']:.6f} "
            f"diff={row['diff']:+.6f} "
            f"rel={row['rel_diff_percent']:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
