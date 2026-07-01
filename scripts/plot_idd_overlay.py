#!/usr/bin/env python3
"""Plot multiple IDD curves on one figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path):
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns: depth and IDD")
    x = data[:, 0]
    y = data[:, 1]
    if x.size > 0 and x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Overlay multiple IDD curves")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input IDD files",
    )
    parser.add_argument(
        "-l",
        "--label",
        nargs="+",
        help="Curve labels; defaults to input file names",
    )
    parser.add_argument("-o", "--output", default="idd_overlay.png", help="Output image")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each curve by its own maximum",
    )
    parser.add_argument("--title", default="IDD comparison", help="Plot title")
    parser.add_argument(
        "--params",
        default=None,
        help="Discretization/run parameters to write inside the figure",
    )
    parser.add_argument(
        "--panel",
        nargs="+",
        help="Panel title for each curve; curves with the same panel are plotted together",
    )
    args = parser.parse_args()

    labels = args.label or [Path(path).parent.name or Path(path).name for path in args.input]
    if len(labels) != len(args.input):
        raise ValueError("--label count must match --input count")
    if args.panel and len(args.panel) != len(args.input):
        raise ValueError("--panel count must match --input count")

    panels = args.panel or [args.title] * len(args.input)
    panel_order = list(dict.fromkeys(panels))
    fig, axes = plt.subplots(
        1,
        len(panel_order),
        figsize=(8.2 * len(panel_order), 5.5),
        sharey=True,
        squeeze=False,
    )
    axes_by_panel = {panel: axes[0, idx] for idx, panel in enumerate(panel_order)}

    for path, label, panel in zip(args.input, labels, panels):
        x, y = load_idd(path)
        if args.normalize:
            ymax = float(np.max(y))
            if ymax > 0.0:
                y = y / ymax
        bp_idx = int(np.argmax(y))
        ax = axes_by_panel[panel]
        ax.plot(x, y, linewidth=2.0, label=f"{label} BP={x[bp_idx]:.2f} cm")
        ax.axvline(x[bp_idx], linestyle=":", linewidth=1.0, alpha=0.5)

    for idx, panel in enumerate(panel_order):
        ax = axes_by_panel[panel]
        ax.set_xlabel("Depth x (cm)")
        if idx == 0:
            ax.set_ylabel("Normalized IDD" if args.normalize else "IDD")
        ax.set_title(panel)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.0)
        ax.set_xlim(left=0.0)
        ax.legend()
        if args.params:
            ax.text(
                0.02,
                0.04,
                args.params,
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
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
