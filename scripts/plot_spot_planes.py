#!/usr/bin/env python3
"""Plot YZ spot dose planes saved by bfp_solver."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_plane(path):
    requested = None
    actual = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# requested_x_cm"):
                requested = float(line.split()[2])
            elif line.startswith("# actual_x_cm"):
                actual = float(line.split()[2])

    data = np.loadtxt(path, comments="#")
    y_values = np.unique(data[:, 0])
    z_values = np.unique(data[:, 1])
    dose = data[:, 2].reshape(len(z_values), len(y_values))
    return requested, actual, y_values, z_values, dose


def sigma_yz(y, z, dose):
    weights = np.maximum(dose, 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    yy, zz = np.meshgrid(y, z)
    y_mean = float(np.sum(yy * weights) / total)
    z_mean = float(np.sum(zz * weights) / total)
    var = float(np.sum(((yy - y_mean) ** 2 + (zz - z_mean) ** 2) * weights) / (2.0 * total))
    return float(np.sqrt(max(var, 0.0)))


def main():
    parser = argparse.ArgumentParser(description="Plot BFPn YZ spot dose planes")
    parser.add_argument("-i", "--input", nargs="+", required=True)
    parser.add_argument("-o", "--output", default="spot_planes.png")
    parser.add_argument("--title", default="Spot distribution")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--paper-contour", action="store_true", help="Use Figure 4/5 style contour panels")
    parser.add_argument("--energy", type=float, help="Beam energy for panel titles")
    args = parser.parse_args()

    planes = [load_plane(path) for path in args.input]
    fig, axes = plt.subplots(
        1,
        len(planes),
        figsize=((3.15 if args.paper_contour else 4.5) * len(planes), 3.2 if args.paper_contour else 4.2),
        squeeze=False,
    )

    images = []
    for ax, path, plane in zip(axes[0], args.input, planes):
        requested, actual, y, z, dose = plane
        values = dose.copy()
        if args.normalize and np.max(values) > 0:
            values /= np.max(values)
        if args.paper_contour:
            yy, zz = np.meshgrid(y, z)
            levels = [0.2, 0.4, 0.6, 0.8]
            image = ax.contour(yy, zz, values, levels=levels, cmap="turbo", linewidths=1.2)
            ax.clabel(image, inline=True, fontsize=7, fmt="%.1f")
            ax.plot([], [], color="tab:blue", linewidth=1.2, label="deterministic")
            ax.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False, edgecolor="0.3")
        else:
            image = ax.imshow(
                values,
                origin="lower",
                extent=[float(y[0]), float(y[-1]), float(z[0]), float(z[-1])],
                aspect="equal",
                cmap="viridis",
            )
            images.append(image)
        sigma = sigma_yz(y, z, dose)
        depth = actual if actual is not None else requested
        if args.paper_contour:
            energy = f", E0={args.energy:g} MeV" if args.energy else ""
            ax.set_title(f"x={depth:g} cm{energy}", fontsize=9, fontweight="bold")
        else:
            ax.set_title(f"x={depth:.2f} cm, sigma={sigma:.4f} cm")
        ax.set_xlabel("y (cm)")
        ax.set_ylabel("z (cm)")
        ax.set_xlim(float(y[0]), float(y[-1]))
        ax.set_ylim(float(z[0]), float(z[-1]))
        if args.paper_contour:
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8, direction="in", top=True, right=True)
        else:
            ax.text(
                0.02,
                0.02,
                Path(path).name,
                transform=ax.transAxes,
                fontsize=7,
                color="white",
                va="bottom",
                ha="left",
            )

    if not args.paper_contour and images:
        fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.82, label="Normalized dose" if args.normalize else "Dose")
        fig.suptitle(args.title)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
