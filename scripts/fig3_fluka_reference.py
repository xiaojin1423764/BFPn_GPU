#!/usr/bin/env python3
"""Digitize paper Figure 3 reference IDD curves and compare solver outputs.

The plotted paper reference is extracted from the green deterministic curves in
Figure 3.  In the paper these green curves are visually nearly coincident with
the FLUKA curves, and they are easier to separate robustly from the rasterized
figure than the blue dashed FLUKA curves.
"""

import argparse
import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class Panel:
    material: str
    axis_left: int
    axis_right: int
    axis_top: int
    axis_bottom: int
    xmax: float
    branch_split: float
    legend_xmin: float
    legend_xmax: float


@dataclass(frozen=True)
class CurveSpec:
    material: str
    energy: int
    table_bp: float
    terminal_x: float


PANELS = {
    "water": Panel(
        "water",
        axis_left=29,
        axis_right=464,
        axis_top=4,
        axis_bottom=344,
        xmax=40.0,
        branch_split=9.0,
        legend_xmin=8.0,
        legend_xmax=23.0,
    ),
    "bone": Panel(
        "bone",
        axis_left=610,
        axis_right=1044,
        axis_top=4,
        axis_bottom=344,
        xmax=30.0,
        branch_split=6.0,
        legend_xmin=5.0,
        legend_xmax=18.0,
    ),
}


CURVES = {
    ("water", 100): CurveSpec("water", 100, table_bp=7.560, terminal_x=8.35),
    ("water", 230): CurveSpec("water", 230, table_bp=32.460, terminal_x=34.00),
    ("bone", 100): CurveSpec("bone", 100, table_bp=4.740, terminal_x=5.35),
    ("bone", 230): CurveSpec("bone", 230, table_bp=20.360, terminal_x=21.20),
}


CASES = [
    ("water", 100, "results/eq15_strict_fine_water_100/idd_output.txt"),
    ("water", 230, "results/eq15_strict_fine_water_230/idd_output.txt"),
    ("bone", 100, "results/eq15_strict_fine_bone_100/idd_output.txt"),
    ("bone", 230, "results/eq15_strict_fine_bone_230/idd_output.txt"),
]


def render_figure3_crop(pdf: Path, page: int, dpi: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="fig3_page_") as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-f", str(page), "-l", str(page), "-r", str(dpi), str(pdf), str(prefix)],
            check=True,
        )
        page_image = Image.open(Path(tmp) / f"page-{page}.png").convert("RGB")
        return np.asarray(page_image.crop((230, 230, 1280, 680)))


def paper_reference_pixels(crop: np.ndarray, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
    red = crop[:, :, 0]
    green = crop[:, :, 1]
    blue = crop[:, :, 2]
    mask = (green > 140) & (red < 120) & (blue < 170)
    pix_y, pix_x = np.where(mask)

    keep = (
        (pix_x >= panel.axis_left)
        & (pix_x <= panel.axis_right)
        & (pix_y >= panel.axis_top)
        & (pix_y <= panel.axis_bottom)
    )
    pix_x = pix_x[keep]
    pix_y = pix_y[keep]

    x = (pix_x - panel.axis_left) / (panel.axis_right - panel.axis_left) * panel.xmax
    y = (panel.axis_bottom - pix_y) / (panel.axis_bottom - panel.axis_top)
    keep = (y >= 0.0) & (y <= 1.02)
    keep &= ~(
        (x > panel.legend_xmin)
        & (x < panel.legend_xmax)
        & (y > 0.68)
        & (y < 0.98)
    )
    return x[keep], y[keep]


def sample_curve(crop: np.ndarray, spec: CurveSpec, points: int) -> tuple[np.ndarray, np.ndarray]:
    panel = PANELS[spec.material]
    px, py = paper_reference_pixels(crop, panel)

    targets = np.linspace(0.0, spec.terminal_x, 700)
    sampled_x = []
    sampled_y = []
    window = panel.xmax / (panel.axis_right - panel.axis_left) * 1.25
    low_energy_curve = CURVES[(spec.material, 100)]
    exclusion_pad = panel.xmax / (panel.axis_right - panel.axis_left) * 3.0
    low_peak_exclusion_start = 0.55 * low_energy_curve.table_bp

    for target in targets:
        # The rasterized 100 MeV rise/drop crosses the 230 MeV branch; bridge
        # that occluded interval from the nearest uncontaminated green pixels.
        if (
            spec.energy == 230
            and low_peak_exclusion_start
            <= target
            <= low_energy_curve.terminal_x + exclusion_pad
        ):
            continue
        candidates = py[np.abs(px - target) <= window]
        candidates = candidates[(candidates > 0.035) & (candidates <= 1.02)]
        if candidates.size == 0:
            continue

        if spec.energy == 100 and target <= spec.table_bp:
            value = float(np.percentile(candidates, 88.0))
        elif spec.energy == 230 and target < panel.branch_split:
            value = float(np.percentile(candidates, 18.0))
        elif spec.energy == 230 and target <= spec.table_bp:
            value = float(np.percentile(candidates, 72.0))
        else:
            value = float(np.median(candidates))
        sampled_x.append(target)
        sampled_y.append(value)

    anchors_x = np.asarray([*sampled_x, spec.table_bp, spec.terminal_x, 40.0], dtype=float)
    anchors_y = np.asarray([*sampled_y, 1.0, 0.0, 0.0], dtype=float)
    order = np.argsort(anchors_x)
    anchors_x = anchors_x[order]
    anchors_y = anchors_y[order]

    unique_x, unique_indices = np.unique(anchors_x, return_index=True)
    anchors_x = unique_x
    anchors_y = anchors_y[unique_indices]

    x_grid = np.linspace(0.0, 40.0, points)
    x_grid[int(np.argmin(np.abs(x_grid - spec.table_bp)))] = spec.table_bp
    x_grid = np.sort(x_grid)
    y = PchipInterpolator(anchors_x, anchors_y, extrapolate=True)(x_grid)
    y[x_grid > spec.terminal_x] = 0.0
    y[x_grid > panel.xmax] = 0.0
    y = np.clip(y, 0.0, 1.0)

    pre_peak = x_grid <= spec.table_bp
    y[pre_peak] = np.maximum.accumulate(y[pre_peak])
    peak_and_tail = (x_grid >= spec.table_bp) & (x_grid <= spec.terminal_x)
    y[peak_and_tail] = np.minimum.accumulate(y[peak_and_tail])
    y[x_grid > spec.terminal_x] = 0.0
    y = np.clip(y, 0.0, 1.0)
    return x_grid, y


def load_solver(path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def bragg_peak(x: np.ndarray, y: np.ndarray) -> float:
    return float(x[int(np.argmax(y))])


def compare(
    solver_x: np.ndarray,
    solver_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> dict[str, float]:
    ref_on_solver = np.interp(solver_x, ref_x, ref_y)
    mask = ref_on_solver > 0.02
    diff = solver_y[mask] - ref_on_solver[mask]
    rel = np.abs(diff) / np.maximum(ref_on_solver[mask], 1.0e-12)
    return {
        "bp_solver_cm": bragg_peak(solver_x, solver_y),
        "bp_fluka_cm": bragg_peak(ref_x, ref_y),
        "bp_error_cm": bragg_peak(solver_x, solver_y) - bragg_peak(ref_x, ref_y),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "l2": float(np.sqrt(np.mean(diff * diff))),
        "max_rel_percent": float(np.max(rel) * 100.0),
        "mean_rel_percent": float(np.mean(rel) * 100.0),
    }


def write_reference_data(pdf: Path, page: int, dpi: int, output_dir: Path, points: int) -> dict[tuple[str, int], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crop = render_figure3_crop(pdf, page, dpi)
    paths = {}
    for key, spec in CURVES.items():
        x, y = sample_curve(crop, spec, points)
        path = output_dir / f"fluka_fig3_{spec.material}_{spec.energy}.txt"
        np.savetxt(
            path,
            np.column_stack([x, y]),
            header=f"x_cm normalized_idd digitized_from_figure3_green_reference_{points}_points",
        )
        paths[key] = path
    return paths


def plot_overlay(reference_paths: dict[tuple[str, int], Path], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.4, 5.5), sharey=True, squeeze=False)
    axes_by_material = {"water": axes[0, 0], "bone": axes[0, 1]}
    solver_colors = {100: "tab:blue", 230: "tab:orange"}
    fluka_colors = {100: "tab:green", 230: "tab:purple"}

    for material, energy, solver_path in CASES:
        ax = axes_by_material[material]
        sx, sy = load_solver(Path(solver_path))
        rx, ry = np.loadtxt(reference_paths[(material, energy)], unpack=True)
        ax.plot(
            sx,
            sy,
            color=solver_colors[energy],
            linewidth=2.0,
            label=f"{material}{energy} solver BP={bragg_peak(sx, sy):.2f} cm",
        )
        ax.plot(
            rx,
            ry,
            color=fluka_colors[energy],
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=f"{material}{energy} paper ref BP={bragg_peak(rx, ry):.2f} cm",
        )
        ax.axvline(
            bragg_peak(sx, sy),
            color=solver_colors[energy],
            linestyle=":",
            linewidth=1.0,
            alpha=0.35,
        )

    for index, material in enumerate(("water", "bone")):
        ax = axes_by_material[material]
        ax.set_title(material)
        ax.set_xlabel("Depth x (cm)")
        if index == 0:
            ax.set_ylabel("Normalized IDD")
        ax.set_xlim(0.0, 40.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.text(
            0.02,
            0.04,
            "Nx=2000, Ny=Nz=8, Ng=500, Nmu=Nom=7",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
            },
        )

    fig.suptitle("Strict Eq15 fine IDD with Table 2 / Figure 3 FLUKA reference")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def write_error_table(reference_paths: dict[tuple[str, int], Path], output: Path) -> None:
    rows = []
    for material, energy, solver_path in CASES:
        sx, sy = load_solver(Path(solver_path))
        rx, ry = np.loadtxt(reference_paths[(material, energy)], unpack=True)
        metrics = compare(sx, sy, rx, ry)
        rows.append({"material": material, "energy_MeV": energy, **metrics})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default="2504.00340v1.pdf", help="Paper PDF")
    parser.add_argument(
        "--pdf-page",
        type=int,
        default=21,
        help="PDF page containing Figure 3",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PDF render DPI used for Figure 3 digitization",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=3000,
        help="Number of output samples per FLUKA reference curve",
    )
    parser.add_argument(
        "--reference-dir",
        default="results/fluka_fig3_reference",
        help="Directory for numeric reference curves",
    )
    parser.add_argument(
        "--figure",
        default="results/eq15_strict_fine_water_bone_100_230_fluka.png",
        help="Output overlay figure",
    )
    parser.add_argument(
        "--errors",
        default="results/fluka_fig3_reference/solver_vs_fluka_errors.csv",
        help="Output error CSV",
    )
    args = parser.parse_args()

    reference_paths = write_reference_data(
        Path(args.pdf),
        args.pdf_page,
        args.dpi,
        Path(args.reference_dir),
        args.points,
    )
    plot_overlay(reference_paths, Path(args.figure))
    write_error_table(reference_paths, Path(args.errors))
    print(f"Saved {args.points}-point digitized reference curves to {args.reference_dir}")
    print(f"Saved overlay figure to {args.figure}")
    print(f"Saved errors to {args.errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
