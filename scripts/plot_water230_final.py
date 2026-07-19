#!/usr/bin/env python3
"""Plot the final water 230 MeV IDD grid comparison and Figure 3 reference."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig3_fluka_reference import CURVES, render_figure3_crop, sample_curve


def load_normalized(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    x = data[:, 0]
    y = data[:, 1]
    peak = float(np.max(y))
    if peak <= 0.0:
        raise ValueError(f"{path} has no positive IDD values")
    return x, y / peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coarse",
        default="results/stage6_final_water230/8_7_nx1000/idd_output.txt",
    )
    parser.add_argument(
        "--fine",
        default="results/stage6_final_water230/12_9_nx1000/idd_output.txt",
    )
    parser.add_argument("--pdf", default="2504.00340v1.pdf")
    parser.add_argument("--pdf-page", type=int, default=21)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--output",
        default="results/stage6_final_water230/water230_final_idd.png",
    )
    args = parser.parse_args()

    coarse_x, coarse_y = load_normalized(Path(args.coarse))
    fine_x, fine_y = load_normalized(Path(args.fine))
    crop = render_figure3_crop(Path(args.pdf), args.pdf_page, args.dpi)
    ref_x, ref_y = sample_curve(crop, CURVES[("water", 230)], 3000)

    curves = [
        (coarse_x, coarse_y, "8x8 / 7x7", "#1677b8", "-"),
        (fine_x, fine_y, "12x12 / 9x9", "#d95f02", "-"),
        (ref_x, ref_y, "Figure 3 pixel reference", "#252525", "--"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for ax in axes:
        for x, y, label, color, linestyle in curves:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.0)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
        ax.set_xlabel("Depth x (cm)")
        ax.set_ylabel("Normalized IDD")

    axes[0].set_title("Full depth-dose curve")
    axes[0].set_xlim(0.0, 35.0)
    axes[0].set_ylim(0.0, 1.04)
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].set_title("Entrance region")
    axes[1].set_xlim(0.0, 5.0)
    axes[1].set_ylim(0.20, 0.32)
    axes[1].legend(loc="lower right", frameon=False)

    fig.suptitle("Water 230 MeV IDD, Nx=1000")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
