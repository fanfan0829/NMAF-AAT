from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


ROOT = Path(
    r"C:\Users\FANFAN\Desktop\review\自然对流"
    r"\nc_Ra1e4_seed9999_fixed_activation_extended_v3_process_isolated"
    r"\results\fixed_budget"
)

SEED = 9999
TARGET_LENGTH = 10000
IMAGE_DPI = 800
SHOW_FIGURES = True


FONT_FAMILY = "Times New Roman"
GLOBAL_FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 10
LINE_WIDTH = 1.5
FIGSIZE = (7.0, 5.0)


Y_LIMIT_COMMON = None
Y_LIMIT_HYBRID = None


LEGEND_LOC_COMMON = "lower left"
LEGEND_LOC_HYBRID = "upper right"


OUTPUT_COMMON = ROOT / "NC_Ra1e4_seed9999_common_activations_loss.png"
OUTPUT_HYBRID = ROOT / "NC_Ra1e4_seed9999_hybrid_activations_loss.png"


COMMON_ACTIVATIONS = [
    {
        "label": "tanh",
        "folder_aliases": ["tanh"],
        "color": "#1f77b4",
        "zorder": 5,
    },
    {
        "label": "ReLU",
        "folder_aliases": ["relu"],
        "color": "#ff7f0e",
        "zorder": 6,
    },
    {
        "label": "sigmoid",
        "folder_aliases": ["sigmoid"],
        "color": "#2ca02c",
        "zorder": 6,
    },
    {
        "label": "Swish",
        "folder_aliases": ["swish"],
        "color": "#8c564b",
        "zorder": 6,
    },
    {
        "label": "sine",
        "folder_aliases": ["sine", "sin"],
        "color": "#bcbd22",
        "zorder": 6,
    },
    {
        "label": "Hermite-Gaussian",
        "folder_aliases": ["gaussian", "gaussion", "gauss"],
        "color": "#9467bd",
        "zorder": 4,
    },
    {
        "label": "GELU",
        "folder_aliases": ["gelu"],
        "color": "#17becf",
        "zorder": 6,
    },
    {
        "label": "Mish",
        "folder_aliases": ["mish"],
        "color": "#e377c2",
        "zorder": 6,
    },
    {
        "label": "NMAF",
        "folder_aliases": ["nmaf", "ihaf"],
        "color": "tab:red",
        "zorder": 7,
    },
]

HYBRID_ACTIVATIONS = [
    {
        "label": "TanSwish",
        "folder_aliases": ["tanswish", "tan_swish"],
        "color": "#1f77b4",
        "zorder": 5,
    },
    {
        "label": "SigSwish",
        "folder_aliases": ["sigswish", "sig_swish"],
        "color": "#e377c2",
        "zorder": 6,
    },
    {
        "label": "SigTanh",
        "folder_aliases": ["sigtanh", "sig_tanh"],
        "color": "#2ca02c",
        "zorder": 6,
    },
    {
        "label": "GaussTanh",
        "folder_aliases": ["gausstanh", "gauss_tanh"],
        "color": "#9467bd",
        "zorder": 4,
    },
    {
        "label": "GaussSigmoid",
        "folder_aliases": ["gausssigmoid", "gauss_sigmoid"],
        "color": "#17becf",
        "zorder": 6,
    },
    {
        "label": "GaussSwish",
        "folder_aliases": ["gaussswish", "gauss_swish"],
        "color": "#ff7f0e",
        "zorder": 6,
    },
    {
        "label": "NMAF",
        "folder_aliases": ["nmaf", "ihaf"],
        "color": "tab:red",
        "zorder": 7,
    },
]


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = FONT_FAMILY
    plt.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams.update({
        "font.size": GLOBAL_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.unicode_minus": False,
    })


def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_method_folder(folder_aliases):
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT does not exist:\n{ROOT}")

    subdirs = [p for p in ROOT.iterdir() if p.is_dir()]
    normalized = {normalize_name(p.name): p for p in subdirs}

    for alias in folder_aliases:
        key = normalize_name(alias)
        if key in normalized:
            return normalized[key]

    return None


def history_file_from_aliases(folder_aliases):
    method_folder = find_method_folder(folder_aliases)

    if method_folder is None:
        return None

    path = method_folder / f"seed_{SEED}" / "training_history.csv"

    if path.exists():
        return path

    return None


def pick_x_column(df):
    candidates = [
        "combined_step",
        "iteration",
        "iterations",
        "iter",
        "step",
        "steps",
        "global_step",
        "stage_step",
    ]

    for col in df.columns:
        key = normalize_name(col)
        for cand in candidates:
            if key == normalize_name(cand):
                return col

    return None


def load_and_pad_curve(history_path):
    df = pd.read_csv(history_path)

    if "total_loss" not in df.columns:
        raise KeyError(
            f"'total_loss' is missing in:\n{history_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    x_col = pick_x_column(df)

    if x_col is None:
        y = pd.to_numeric(df["total_loss"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y) & (y > 0.0)
        y = y[valid]
    else:
        x_raw = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(df["total_loss"], errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(x_raw) & np.isfinite(y_raw) & (y_raw > 0.0)

        if not np.any(valid):
            raise RuntimeError(
                f"No valid positive total-loss values in:\n{history_path}"
            )

        y = y_raw[valid]

    if len(y) == 0:
        raise RuntimeError(f"No valid loss values in:\n{history_path}")

    if len(y) >= TARGET_LENGTH:
        y_plot = y[:TARGET_LENGTH]
    else:
        pad_len = TARGET_LENGTH - len(y)
        y_plot = np.concatenate([y, np.full(pad_len, y[-1], dtype=float)])

    x_plot = np.arange(1, TARGET_LENGTH + 1, dtype=float)

    return x_plot, y_plot, len(y)


def apply_axis_style(ax, y_limit, legend_loc):
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Total loss", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(1, TARGET_LENGTH)

    if y_limit is not None:
        if len(y_limit) != 2:
            raise ValueError("y_limit must be None or (YMIN, YMAX).")

        ymin, ymax = y_limit

        if ymin <= 0 or ymax <= 0:
            raise ValueError("For logarithmic y-axis, limits must be > 0.")

        if ymin >= ymax:
            raise ValueError("YMIN must be smaller than YMAX.")

        ax.set_ylim(ymin, ymax)

    ax.legend(loc=legend_loc, fontsize=LEGEND_SIZE, frameon=True)

    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)

    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_SIZE)


def plot_group(group_specs, output_path, y_limit, legend_loc, group_title):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    plotted_count = 0

    print()
    print("=" * 100)
    print(group_title)
    print("=" * 100)

    for spec in group_specs:
        history_path = history_file_from_aliases(spec["folder_aliases"])

        if history_path is None:
            print(f"SKIPPED {spec['label']:<15s} | history file not found")
            continue

        x_plot, y_plot, original_len = load_and_pad_curve(history_path)

        ax.semilogy(
            x_plot,
            y_plot,
            linewidth=LINE_WIDTH,
            color=spec["color"],
            label=spec["label"],
            zorder=spec["zorder"],
        )

        plotted_count += 1

        if original_len < TARGET_LENGTH:
            print(
                f"PLOTTED {spec['label']:<15s} | "
                f"original length = {original_len}, padded to {TARGET_LENGTH}"
            )
        else:
            print(
                f"PLOTTED {spec['label']:<15s} | "
                f"original length = {original_len}"
            )

    if plotted_count == 0:
        raise RuntimeError(f"No valid methods were plotted for:\n{group_title}")

    apply_axis_style(ax=ax, y_limit=y_limit, legend_loc=legend_loc)

    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI, format="png", bbox_inches="tight")

    print(f"\nSaved: {output_path}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def main():
    configure_plot_style()

    plot_group(
        group_specs=COMMON_ACTIVATIONS,
        output_path=OUTPUT_COMMON,
        y_limit=Y_LIMIT_COMMON,
        legend_loc=LEGEND_LOC_COMMON,
        group_title="NC common activations",
    )

    plot_group(
        group_specs=HYBRID_ACTIVATIONS,
        output_path=OUTPUT_HYBRID,
        y_limit=Y_LIMIT_HYBRID,
        legend_loc=LEGEND_LOC_HYBRID,
        group_title="NC hybrid activations",
    )


if __name__ == "__main__":
    main()
