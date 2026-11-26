from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DPI = 800
SHOW_FIGURES = True
FONT_FAMILY = "Times New Roman"
GLOBAL_FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 10
LINE_WIDTH = 1.5
INDIVIDUAL_FIGSIZE = (7.0, 5.0)
COMBINED_FIGSIZE = (12.0, 8.5)


CASES = {
    "Burgers": {
        "folder": "Burgers",
        "panel": "(a)",
        "panel_name": "Burgers equation",
    },
    "NLSE": {
        "folder": "NLSE",
        "panel": "(b)",
        "panel_name": "Nonlinear Schrödinger equation",
    },
    "Allen_Cahn": {
        "folder": "Allen_Cahn",
        "panel": "(c)",
        "panel_name": "Allen–Cahn equation",
    },
    "Wave": {
        "folder": "Wave",
        "panel": "(d)",
        "panel_name": "Wave equation",
    },
}


Y_LIMITS = {
    "Burgers": (5e-7, 1e3),
    "NLSE": (1e-5, 1e4),
    "Allen_Cahn": (1e-6, 1e4),
    "Wave": None,
}


LEGEND_NCOL = 2

LEGEND_LOCS = {
    "Burgers": "upper right",
    "NLSE": "upper right",
    "Allen_Cahn": "upper right",
    "Wave": "lower left",
}


ACTIVATION_RULES = [
    {
        "key": "tanh",
        "aliases": ["tanh"],
        "label": "tanh",
        "color": "#1f77b4",
        "zorder": 5,
    },
    {
        "key": "relu",
        "aliases": ["relu"],
        "label": "ReLU",
        "color": "#ff7f0e",
        "zorder": 6,
    },
    {
        "key": "sigmoid",
        "aliases": ["sigmoid"],
        "label": "sigmoid",
        "color": "#2ca02c",
        "zorder": 6,
    },
    {
        "key": "swish",
        "aliases": ["swish"],
        "label": "Swish",
        "color": "#8c564b",
        "zorder": 6,
    },
    {
        "key": "sine",
        "aliases": ["sine", "sin"],
        "label": "sine",
        "color": "#bcbd22",
        "zorder": 6,
    },
    {
        "key": "gaussian",
        "aliases": ["gaussian", "gaussion", "gauss"],
        "label": "Hermite-Gaussian",
        "color": "#9467bd",
        "zorder": 4,
    },
    {
        "key": "gelu",
        "aliases": ["gelu"],
        "label": "GELU",
        "color": "#17becf",
        "zorder": 6,
    },
    {
        "key": "mish",
        "aliases": ["mish"],
        "label": "Mish",
        "color": "#e377c2",
        "zorder": 6,
    },
    {
        "key": "nmaf",
        "aliases": ["nmaf", "ihaf"],
        "label": "NMAF",
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
        .replace("*", "")
        .replace("(", "")
        .replace(")", "")
    )
def find_column(df, aliases):
    normalized_columns = {
        normalize_name(col): col
        for col in df.columns
    }
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized_columns:
            return normalized_columns[key]
    return None
def find_iteration_column(df):
    aliases = [
        "iteration",
        "iterations",
        "iter",
        "step",
        "steps",
        "epoch",
        "epochs",
    ]
    return find_column(df, aliases)
def load_case(case_key):
    case = CASES[case_key]
    excel_path = (
        SCRIPT_DIR
        / case["folder"]
        / "1loss.xlsx"
    )
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Cannot find:\n{excel_path}"
        )
    df = pd.read_excel(excel_path)
    df = (
        df
        .dropna(axis=1, how="all")
        .dropna(axis=0, how="all")
    )
    if df.empty:
        raise RuntimeError(
            f"{excel_path} contains no valid data."
        )
    iteration_col = find_iteration_column(df)
    if iteration_col is None:
        x = np.arange(
            1,
            len(df) + 1,
            dtype=float,
        )
    else:
        x = pd.to_numeric(
            df[iteration_col],
            errors="coerce",
        ).to_numpy(dtype=float)
    curves = {}
    for rule in ACTIVATION_RULES:
        col = find_column(
            df,
            rule["aliases"],
        )
        if col is None:
            print(
                f"[{case_key}] skipped: "
                f"column not found for {rule['label']}"
            )
            continue
        y = pd.to_numeric(
            df[col],
            errors="coerce",
        ).to_numpy(dtype=float)
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (y > 0.0)
        )
        if not np.any(valid):
            print(
                f"[{case_key}] skipped: "
                f"no positive finite values for {rule['label']}"
            )
            continue
        curves[rule["key"]] = {
            "x": x[valid],
            "y": y[valid],
            "rule": rule,
        }
    if not curves:
        raise RuntimeError(
            f"No valid activation loss columns were found in:\n{excel_path}\n"
            f"Excel columns: {list(df.columns)}"
        )
    return curves


def format_loss_axis(ax, case_key):
    ax.set_yscale("log")
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
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.set_ylabel(
        "Total loss",
        fontsize=AXIS_LABEL_SIZE,
    )
    y_lim = Y_LIMITS.get(case_key, None)
    if y_lim is not None:
        if len(y_lim) != 2:
            raise ValueError(
                f"Y_LIMITS['{case_key}'] must be None or (YMIN, YMAX)."
            )
        ymin, ymax = y_lim
        if ymin <= 0 or ymax <= 0:
            raise ValueError(
                f"Y_LIMITS['{case_key}'] must be > 0 for a log axis."
            )
        if ymin >= ymax:
            raise ValueError(
                f"Y_LIMITS['{case_key}']: YMIN must be smaller than YMAX."
            )
        ax.set_ylim(
            ymin,
            ymax,
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
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=TICK_LABEL_SIZE,
    )
def draw_case(ax, case_key, curves, show_panel_label=False):
    for rule in ACTIVATION_RULES:
        key = rule["key"]
        if key not in curves:
            continue
        data = curves[key]
        ax.semilogy(
            data["x"],
            data["y"],
            linewidth=LINE_WIDTH,
            color=rule["color"],
            label=rule["label"],
            zorder=rule["zorder"],
        )
    format_loss_axis(
        ax,
        case_key,
    )
    ax.legend(
        loc=LEGEND_LOCS[case_key],
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_SIZE,
        frameon=True,
        columnspacing=1.0,
        handlelength=2.0,
        borderaxespad=0.6,
    )
    if show_panel_label:
        case = CASES[case_key]
        ax.text(
            0.5,
            -0.22,
            f"{case['panel']} {case['panel_name']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )


def save_individual_figures(all_curves):
    outputs = []
    for case_key, curves in all_curves.items():
        fig, ax = plt.subplots(
            figsize=INDIVIDUAL_FIGSIZE
        )
        draw_case(
            ax,
            case_key,
            curves,
            show_panel_label=False,
        )
        plt.tight_layout()
        output_path = (
            SCRIPT_DIR
            / CASES[case_key]["folder"]
            / f"{case_key}_activation_ablation_loss.png"
        )
        fig.savefig(
            output_path,
            dpi=IMAGE_DPI,
            format="png",
            bbox_inches="tight",
        )
        outputs.append(output_path)
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)
    return outputs


def save_combined_figure(all_curves):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=COMBINED_FIGSIZE,
    )
    axes = axes.ravel()
    case_order = [
        "Burgers",
        "NLSE",
        "Allen_Cahn",
        "Wave",
    ]
    for ax, case_key in zip(
        axes,
        case_order,
    ):
        draw_case(
            ax,
            case_key,
            all_curves[case_key],
            show_panel_label=True,
        )
    fig.tight_layout(
        h_pad=3.0,
        w_pad=2.0,
    )
    output_path = (
        SCRIPT_DIR
        / "Four_time_benchmarks_activation_ablation_loss.png"
    )
    fig.savefig(
        output_path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    configure_plot_style()
    all_curves = {}
    for case_key in CASES:
        all_curves[case_key] = load_case(case_key)
    individual_outputs = save_individual_figures(
        all_curves
    )
    combined_output = save_combined_figure(
        all_curves
    )
    print("\nSaved individual figures:")
    for path in individual_outputs:
        print(" ", path)
    print("\nSaved combined paper figure:")
    print(" ", combined_output)
if __name__ == "__main__":
    main()
