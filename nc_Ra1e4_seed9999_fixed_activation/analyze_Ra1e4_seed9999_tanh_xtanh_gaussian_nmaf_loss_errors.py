from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


from nc_fixed_activation_common import FIXED_ROOT
ROOT = FIXED_ROOT

SEED = 9999
IMAGE_DPI = 800
SHOW_FIGURE = True

OUTPUT_PNG = ROOT / "Ra1e4_seed9999_tanh_xtanh_gaussian_nmaf_loss.png"


METHODS = [
    {
        "key": "tanh",
        "label": "tanh",
        "color": "#1f77b4",
        "zorder": 5,
    },
    {
        "key": "xtanh",
        "label": r"$x\tanh(x)$",
        "color": "#2ca02c",
        "zorder": 6,
    },
    {
        "key": "gaussian",
        "label": "Hermite-Gaussian",
        "color": "#9467bd",
        "zorder": 4,
    },
    {
        "key": "nmaf",
        "label": "NMAF",
        "color": "tab:red",
        "zorder": 7,
    },
]


BENCHMARK = {
    "psi_mid_abs": 5.071,
    "u_max_vertical_midplane": 16.178,
    "w_max_horizontal_midplane": 19.617,
    "Nu_max_hot_wall": 3.528,
    "Nu_min_hot_wall": 0.586,
    "Nu_mean_hot_wall": 2.243,
}

QUANTITY_LABELS = {
    "psi_mid_abs": "|psi_mid|",
    "u_max_vertical_midplane": "u_max",
    "w_max_horizontal_midplane": "w_max",
    "Nu_max_hot_wall": "Nu_max",
    "Nu_min_hot_wall": "Nu_min",
    "Nu_mean_hot_wall": "Nu_mean",
}


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 12})


def method_dir(method_key):
    return ROOT / method_key / f"seed_{SEED}"


def history_path(method_key):
    return method_dir(method_key) / "training_history.csv"


def summary_path(method_key):
    return method_dir(method_key) / "training_summary.json"


def load_history(method_key):
    path = history_path(method_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find history file:\n{path}"
        )

    df = pd.read_csv(path)

    if "total_loss" not in df.columns:
        raise KeyError(
            f"'total_loss' is missing in:\n{path}\n"
            f"Columns = {list(df.columns)}"
        )


    x_candidates = [
        "combined_step",
        "iteration",
        "iterations",
        "step",
        "global_step",
        "stage_step",
    ]

    x_col = None
    for name in x_candidates:
        if name in df.columns:
            x_col = name
            break

    if x_col is None:
        x = np.arange(1, len(df) + 1, dtype=float)
    else:
        x = pd.to_numeric(
            df[x_col],
            errors="coerce",
        ).to_numpy(dtype=float)

    y = pd.to_numeric(
        df["total_loss"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (y > 0.0)
    )

    if not np.any(valid):
        raise RuntimeError(
            f"No valid positive total-loss values in:\n{path}"
        )

    return x[valid], y[valid]


def load_summary(method_key):
    path = summary_path(method_key)

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find summary file:\n{path}"
        )

    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    if "physical_quantities" not in obj:
        raise KeyError(
            f"'physical_quantities' is missing in:\n{path}"
        )

    return obj


def plot_loss():
    configure_plot_style()

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    final_losses = {}

    for spec in METHODS:
        method = spec["key"]

        x, y = load_history(method)

        ax.semilogy(
            x,
            y,
            linewidth=1.5,
            color=spec["color"],
            label=spec["label"],
            zorder=spec["zorder"],
        )

        final_losses[method] = float(y[-1])


    ax.yaxis.set_major_locator(
        LogLocator(
            base=10.0,
            numticks=20,
        )
    )

    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100,
        )
    )

    ax.yaxis.set_minor_formatter(
        NullFormatter()
    )

    ax.set_xlabel(
        "Iteration",
        fontsize=14,
    )

    ax.set_ylabel(
        "Total loss",
        fontsize=14,
    )

    ax.legend(
        loc="upper right",
        fontsize=10,
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        alpha=0.6,
    )

    ax.grid(
        True,
        which="minor",
        linestyle="--",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PNG,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    if SHOW_FIGURE:
        plt.show()

    plt.close(fig)

    return final_losses


def relative_error_percent(pred, ref):
    return abs(
        float(pred) - float(ref)
    ) / abs(float(ref)) * 100.0


def print_errors(final_losses):
    print()
    print("=" * 100)
    print("Ra = 1e4 | seed = 9999 | fixed-activation ablation")
    print("=" * 100)

    for spec in METHODS:
        method = spec["key"]
        label = spec["label"]

        summary = load_summary(method)
        quantities = summary["physical_quantities"]

        print()
        print("-" * 100)
        print(f"{label}")
        print("-" * 100)

        print(
            f"Final total loss: "
            f"{final_losses[method]:.6e}"
        )

        errors = []

        print(
            f"{'Quantity':<14s}"
            f"{'Prediction':>16s}"
            f"{'Benchmark':>16s}"
            f"{'Relative error (%)':>22s}"
        )

        for key, benchmark_value in BENCHMARK.items():
            if key not in quantities:
                raise KeyError(
                    f"{key!r} is missing from "
                    f"{summary_path(method)}"
                )

            pred = float(
                quantities[key]
            )

            err = relative_error_percent(
                pred,
                benchmark_value,
            )

            errors.append(err)

            print(
                f"{QUANTITY_LABELS[key]:<14s}"
                f"{pred:>16.6f}"
                f"{benchmark_value:>16.6f}"
                f"{err:>22.6f}"
            )

        print(
            f"Mean relative error (%): "
            f"{np.mean(errors):.6f}"
        )

        print(
            f"Maximum relative error (%): "
            f"{np.max(errors):.6f}"
        )

    print()
    print("=" * 100)
    print(f"Loss figure saved to:\n{OUTPUT_PNG}")
    print("=" * 100)


def main():
    final_losses = plot_loss()
    print_errors(final_losses)


if __name__ == "__main__":
    main()
