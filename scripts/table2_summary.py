#!/usr/bin/env python3
"""Summarize existing IDD outputs against paper Table 2 positions."""

import argparse
import csv
from pathlib import Path

from analyze_idd import TABLE2, analyze, load_idd


DEFAULT_CASES = [
    ("water", 100, "results/eq15_strict_fine_water_100/idd_output.txt"),
    ("water", 230, "results/eq15_strict_fine_water_230/idd_output.txt"),
    ("bone", 100, "results/eq15_strict_fine_bone_100/idd_output.txt"),
    ("bone", 230, "results/eq15_strict_fine_bone_230/idd_output.txt"),
]
TABLE2_CASES = [
    ("water", 50),
    ("water", 100),
    ("water", 230),
    ("bone", 50),
    ("bone", 100),
    ("bone", 230),
]


def case_rows(material: str, energy: int, path: Path) -> list[dict[str, object]]:
    x, dose = load_idd(path)
    metrics = analyze(x, dose)
    reference = TABLE2[(material, energy)]
    rows = []
    for metric in ("BP", "P90", "D90", "D20"):
        gpu_cm = metrics[metric]
        paper_cm = reference[metric]
        diff_cm = gpu_cm - paper_cm
        rows.append(
            {
                "material": material,
                "energy_MeV": energy,
                "metric": metric,
                "gpu_cm": gpu_cm,
                "paper_table2_cm": paper_cm,
                "diff_cm": diff_cm,
                "rel_diff_percent": diff_cm / paper_cm * 100.0,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| case | metric | GPU cm | Table 2 cm | diff cm | rel diff % |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            case = f"{row['material']} {row['energy_MeV']}"
            handle.write(
                f"| {case} | {row['metric']} | "
                f"{row['gpu_cm']:.4f} | {row['paper_table2_cm']:.4f} | "
                f"{row['diff_cm']:+.4f} | {row['rel_diff_percent']:+.4f} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="results/table2_validation/table2_summary.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--markdown",
        default="results/table2_validation/table2_summary.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--results-root",
        help="read standard case outputs from ROOT/<material>_<energy>/idd_output.txt",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("MATERIAL", "ENERGY", "IDD"),
        help="explicit case to include; may be passed multiple times",
    )
    args = parser.parse_args()

    if args.results_root and args.case:
        parser.error("--results-root and --case are mutually exclusive")

    if args.results_root:
        root = Path(args.results_root)
        cases = [
            (material, energy, root / f"{material}_{energy}" / "idd_output.txt")
            for material, energy in TABLE2_CASES
        ]
    elif args.case:
        cases = [
            (material, int(energy), Path(path))
            for material, energy, path in args.case
        ]
    else:
        cases = DEFAULT_CASES

    rows = []
    for material, energy, path in cases:
        rows.extend(case_rows(material, energy, Path(path)))

    write_csv(rows, Path(args.csv))
    write_markdown(rows, Path(args.markdown))
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
