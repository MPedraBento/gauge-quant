"""
Plot rotation loss convergence.
Usage: python scripts/plot_convergence.py --csv path/to/rot_loss.csv
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output", default="gauge_convergence.pdf")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    labels = args.labels or [f"Run {i+1}" for i in range(len(args.csv))]
    colors = ["#2c3e50", "#e74c3c", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))

    for i, (csv_path, label) in enumerate(zip(args.csv, labels)):
        df = pd.read_csv(csv_path)
        smooth = df["rot_loss"].rolling(window=args.window, min_periods=1).mean()
        ax.plot(df["step"], smooth, label=label, linewidth=1.5, color=colors[i % len(colors)])

    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"Rotation Loss ($\mathcal{L}_{\mathrm{rot}}$)")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
