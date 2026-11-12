from __future__ import annotations

import json
from pathlib import Path
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import torch

from ac_beta_ablation_common import (
    ANALYSIS_DIR, SEEDS, DTYPE,
    build_alpha_only_model as build_a_only_model,
    build_full_aat_model,
    alpha_history_path as a_history_path,
    full_history_path,
    alpha_checkpoint_path as a_checkpoint_path,
    full_checkpoint_path,
    alpha_summary_path as a_summary_path,
    full_summary_path,
    evaluate_model,
)


FINAL_EVAL = 20000
IMAGE_DPI = 800
SHOW_FIGURES = True
CPU_DEVICE = torch.device("cpu")


FONT_FAMILY = "Times New Roman"
GLOBAL_FONT_SIZE = 14
GLOBAL_AXES_LABEL_SIZE = 14
GLOBAL_AXES_TITLE_SIZE = 14
GLOBAL_LEGEND_SIZE = 12
GLOBAL_XTICK_SIZE = 12
GLOBAL_YTICK_SIZE = 12
GLOBAL_LINE_WIDTH = 1.0


LOSS_FIGSIZE = (7.0, 5.0)
LOSS_BAND_ALPHA = 0.14
LOSS_MAJOR_Y_GRID_WIDTH = 0.8
LOSS_MAJOR_Y_GRID_ALPHA = 0.55
LOSS_MINOR_Y_GRID_WIDTH = 0.45
LOSS_MINOR_Y_GRID_ALPHA = 0.30
LOSS_X_GRID_WIDTH = 0.6
LOSS_X_GRID_ALPHA = 0.22
LOSS_LEGEND_FRAME = True


COMBINED_FIGSIZE = (13.2, 6.20)
SUBPLOT_W_PAD = 0.75
LAYOUT_LEFT = 0.00
LAYOUT_BOTTOM = 0.07
LAYOUT_RIGHT = 1.00
LAYOUT_TOP = 0.90


METHOD_CENTER_DISTANCE = 0.68
BAR_WIDTH = 0.38
PANEL_X_MARGIN_LEFT = 0.4
PANEL_X_MARGIN_RIGHT = 0.4
BAR_COLOR = "#C0B3D0"
BAR_ALPHA = 0.78
BAR_EDGE_WIDTH = 0.75
ERRORBAR_CAPSIZE = 4.5
ERRORBAR_LINEWIDTH = 1.05
ERRORBAR_CAPTHICK = 1.05


SEED_ORDER = [2026, 99, 1234]
SEED_OFFSETS = {
    2026: -0.10,
    99: 0.00,
    1234: 0.10,
}
SEED_MARKERS = {2026: "o", 99: "s", 1234: "^"}
SEED_COLORS = {
    2026: "#1f77b4",
    99: "#ff7f0e",
    1234: "#2ca02c",
}
SEED_POINT_SIZE = 42
SEED_POINT_EDGE_WIDTH = 0.60


PANEL_XTICK_FONT_SIZE = 12
PANEL_XTICK_PAD = 5
PANEL_YTICK_FONT_SIZE = 12
PANEL_YLABEL_FONT_SIZE = 13
PANEL_YLABEL_PAD = 5
PANEL_GRID_ALPHA = 0.35
PANEL_GRID_WIDTH = 0.65


ANNOTATION_X_FACTOR = 0.77
ANNOTATION_Y_FACTOR_DEFAULT = 0.80
ANNOTATION_Y_FACTOR_LOSS = 0.95
ANNOTATION_FONT_SIZE = 9.5
ANNOTATION_BOX_PAD = 0.14
ANNOTATION_BOX_ALPHA = 0.88


PANEL_LABEL_X = 0.50
PANEL_LABEL_Y = -0.20
PANEL_LABEL_FONT_SIZE = 14


LEGEND_MARKER_SIZE = 7.2
LEGEND_MARKER_EDGE_WIDTH = 0.60
LEGEND_FONT_SIZE = 12.5
LEGEND_NCOL = 3
LEGEND_Y = 0.982
LEGEND_BORDER_PAD = 0.50
LEGEND_HANDLE_TEXT_PAD = 0.50
LEGEND_COLUMN_SPACING = 1.40
LEGEND_FRAME = True


METHODS = ["a_only", "full_aat"]
LABELS = {
    "a_only": r"NMAF-AAT ($a$ only)",
    "full_aat": r"NMAF-AAT ($a+c$)",
}
LINESTYLES = {"a_only": "--", "full_aat": "-"}
MARKERS = {"a_only": "o", "full_aat": "s"}

assert CPU_DEVICE.type == "cpu"


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = FONT_FAMILY
    plt.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams.update({
        "font.size": GLOBAL_FONT_SIZE,
        "axes.labelsize": GLOBAL_AXES_LABEL_SIZE,
        "axes.titlesize": GLOBAL_AXES_TITLE_SIZE,
        "legend.fontsize": GLOBAL_LEGEND_SIZE,
        "xtick.labelsize": GLOBAL_XTICK_SIZE,
        "ytick.labelsize": GLOBAL_YTICK_SIZE,
        "lines.linewidth": GLOBAL_LINE_WIDTH,
    })


def history_path(method, seed):
    return a_history_path(seed) if method == "a_only" else full_history_path(seed)


def checkpoint_path(method, seed):
    return a_checkpoint_path(seed, FINAL_EVAL) if method == "a_only" else full_checkpoint_path(seed, FINAL_EVAL)


def summary_path(method, seed):
    return a_summary_path(seed) if method == "a_only" else full_summary_path(seed)


def load_history(method, seed):
    path = history_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    needed = {"lbfgs_eval", "total_loss"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Unexpected history columns in {path}: {list(df.columns)}")
    df = df[["lbfgs_eval", "total_loss"]].copy()
    df["lbfgs_eval"] = pd.to_numeric(df["lbfgs_eval"], errors="coerce")
    df["total_loss"] = pd.to_numeric(df["total_loss"], errors="coerce")
    df = df.dropna()
    df = df[(df["lbfgs_eval"] >= 1) & (df["lbfgs_eval"] <= FINAL_EVAL)]
    df["lbfgs_eval"] = df["lbfgs_eval"].astype(int)
    return df.drop_duplicates("lbfgs_eval", keep="last")


def inspect_actual_evals():
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            path = summary_path(method, seed)
            obj = json.loads(path.read_text(encoding="utf-8"))
            actual = int(obj["final_actual_lbfgs_eval"])
            rows.append({"method": method, "seed": seed, "final_actual_lbfgs_eval": actual})
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_DIR / "actual_lbfgs_evaluations.csv", index=False)
    return out


def curve_stats(method):
    frames = []
    for seed in SEEDS:
        f = load_history(method, seed).rename(columns={"total_loss": f"loss_seed_{seed}"})
        frames.append(f)
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="lbfgs_eval", how="inner")
    if merged.empty:
        raise RuntimeError(f"No common evaluations across seeds for {method}")
    vals = merged[[f"loss_seed_{s}" for s in SEEDS]].to_numpy(float)
    if np.any(vals <= 0) or np.any(~np.isfinite(vals)):
        raise ValueError(f"Invalid loss values for {method}")
    logs = np.log10(vals)
    mean_log = logs.mean(axis=1)
    std_log = logs.std(axis=1, ddof=1)
    out = pd.DataFrame({
        "lbfgs_eval": merged["lbfgs_eval"].to_numpy(int),
        "geometric_mean_total_loss": 10.0 ** mean_log,
        "log10_sample_std": std_log,
        "lower_band": 10.0 ** (mean_log - std_log),
        "upper_band": 10.0 ** (mean_log + std_log),
    })
    for seed in SEEDS:
        out[f"total_loss_seed_{seed}"] = merged[f"loss_seed_{seed}"].to_numpy(float)
    out.to_csv(ANALYSIS_DIR / f"{method}_loss_log_statistics.csv", index=False)
    return out


def plot_loss_curves(stats_map):
    fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)

    for method in METHODS:
        frame = stats_map[method]

        line, = ax.plot(
            frame["lbfgs_eval"],
            frame["geometric_mean_total_loss"],
            linestyle=LINESTYLES[method],
            label=LABELS[method],
            zorder=3,
        )

        ax.fill_between(
            frame["lbfgs_eval"],
            frame["lower_band"],
            frame["upper_band"],
            color=line.get_color(),
            alpha=LOSS_BAND_ALPHA,
            linewidth=0.0,
            zorder=1,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")

    common_max = min(
        int(stats_map[m]["lbfgs_eval"].max())
        for m in METHODS
    )
    ax.set_xlim(
        1,
        min(FINAL_EVAL, common_max),
    )

    ax.yaxis.set_major_locator(
        LogLocator(
            base=10.0,
            numticks=100,
        )
    )
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=1000,
        )
    )

    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=LOSS_MAJOR_Y_GRID_WIDTH,
        alpha=LOSS_MAJOR_Y_GRID_ALPHA,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=LOSS_MINOR_Y_GRID_WIDTH,
        alpha=LOSS_MINOR_Y_GRID_ALPHA,
    )
    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="-",
        linewidth=LOSS_X_GRID_WIDTH,
        alpha=LOSS_X_GRID_ALPHA,
    )

    ax.legend(
        frameon=LOSS_LEGEND_FRAME,
    )

    fig.tight_layout()

    out = ANALYSIS_DIR / "allen_cahn_c_ablation_total_loss.png"
    fig.savefig(
        out,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def load_model_at_20000(method, seed):
    path = checkpoint_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing true eval20000 checkpoint: {path}\n"
            "If the a-only run stopped early, inspect actual_lbfgs_evaluations.csv; "
            "the script deliberately does not create a fake eval20000 checkpoint."
        )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    actual = int(ck.get("actual_lbfgs_eval", ck.get("lbfgs_eval", -1)))
    if actual < FINAL_EVAL:
        raise RuntimeError(f"{path} is labeled eval20000 but actual_lbfgs_eval={actual}")
    model = (build_a_only_model(device=CPU_DEVICE)
             if method == "a_only"
             else build_full_aat_model(device=CPU_DEVICE))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model, ck


def final_loss_from_history(method, seed):
    h = load_history(method, seed)
    exact = h[h["lbfgs_eval"] == FINAL_EVAL]
    if not exact.empty:
        return float(exact.iloc[-1]["total_loss"]), FINAL_EVAL


    le = h[h["lbfgs_eval"] <= FINAL_EVAL]
    if le.empty:
        raise RuntimeError(f"No loss at/before {FINAL_EVAL} for {method}, seed={seed}")
    return float(le.iloc[-1]["total_loss"]), int(le.iloc[-1]["lbfgs_eval"])


def build_final_tables():
    rows = []
    adaptive_rows = []
    for method in METHODS:
        for seed in SEEDS:
            model, ck = load_model_at_20000(method, seed)
            metrics = evaluate_model(model, device=CPU_DEVICE)
            loss, loss_eval = final_loss_from_history(method, seed)
            rows.append({
                "method": method,
                "method_label": LABELS[method],
                "seed": int(seed),
                "loss_eval": int(loss_eval),
                "checkpoint_actual_eval": int(ck.get("actual_lbfgs_eval", FINAL_EVAL)),
                "total_loss": loss,
                "relative_l2_error": metrics["relative_l2_error"],
                "maximum_absolute_error": metrics["maximum_absolute_error"],
            })

            state = ck["model_state_dict"]


            a = state["activation.alpha"].detach().cpu().numpy()
            item = {
                "method": method,
                "seed": int(seed),
                "a_mean": float(a.mean()),
                "a_std": float(a.std(ddof=1)),
                "a_min": float(a.min()),
                "a_max": float(a.max()),
                "delta_a_l2": float(np.linalg.norm(a - 1.0)),
            }

            if method == "full_aat":
                c = state["activation.beta"].detach().cpu().numpy()
                item.update({
                    "c_mean": float(c.mean()),
                    "c_std": float(c.std(ddof=1)),
                    "c_min": float(c.min()),
                    "c_max": float(c.max()),
                    "delta_c_l2": float(np.linalg.norm(c)),
                })
            else:
                item.update({
                    "c_mean": np.nan,
                    "c_std": np.nan,
                    "c_min": np.nan,
                    "c_max": np.nan,
                    "delta_c_l2": np.nan,
                })

            adaptive_rows.append(item)

    individual = pd.DataFrame(rows)
    individual.to_csv(ANALYSIS_DIR / "individual_seed_final_metrics.csv", index=False)
    pd.DataFrame(adaptive_rows).to_csv(ANALYSIS_DIR / "adaptive_parameter_summary.csv", index=False)

    stat_rows = []
    for method in METHODS:
        g = individual[individual["method"] == method]
        row = {"method": method, "method_label": LABELS[method], "number_of_seeds": len(g)}
        for metric in ["total_loss", "relative_l2_error", "maximum_absolute_error"]:
            vals = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sample_std"] = float(vals.std(ddof=1))
        stat_rows.append(row)
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(ANALYSIS_DIR / "mean_std_final_metrics.csv", index=False)

    paper = []
    for _, r in stats.iterrows():
        paper.append({
            "Method": r["method_label"],
            "Final total loss": f"{r['total_loss_mean']:.6e} ± {r['total_loss_sample_std']:.6e}",
            "Relative L2 error": f"{r['relative_l2_error_mean']:.6e} ± {r['relative_l2_error_sample_std']:.6e}",
            "Maximum absolute error": f"{r['maximum_absolute_error_mean']:.6e} ± {r['maximum_absolute_error_sample_std']:.6e}",
        })
    pd.DataFrame(paper).to_csv(ANALYSIS_DIR / "paper_ready_c_ablation_table.csv", index=False)


    paired = []
    for seed in SEEDS:
        a = individual[(individual.method == "a_only") & (individual.seed == seed)].iloc[0]
        f = individual[(individual.method == "full_aat") & (individual.seed == seed)].iloc[0]
        row = {"seed": int(seed)}
        for metric in ["total_loss", "relative_l2_error", "maximum_absolute_error"]:
            av = float(a[metric]); fv = float(f[metric])
            row[f"a_only_{metric}"] = av
            row[f"full_aat_{metric}"] = fv
            row[f"full_aat_reduction_percent_{metric}"] = 100.0 * (av - fv) / av
        paired.append(row)
    pd.DataFrame(paired).to_csv(ANALYSIS_DIR / "paired_c_effect_by_seed.csv", index=False)
    return individual, stats


def _log_safe_yerr(means, stds):
    means = np.asarray(means, float)
    stds = np.asarray(stds, float)
    return np.vstack([np.minimum(stds, 0.95 * means), stds])


def draw_metric_bar_panel(ax, individual, stats, metric, ylabel, panel_label):


    x = np.array(
        [0.0, METHOD_CENTER_DISTANCE],
        dtype=float,
    )

    means = np.array(
        [
            stats.loc[
                stats.method == m,
                f"{metric}_mean",
            ].iloc[0]
            for m in METHODS
        ],
        dtype=float,
    )

    stds = np.array(
        [
            stats.loc[
                stats.method == m,
                f"{metric}_sample_std",
            ].iloc[0]
            for m in METHODS
        ],
        dtype=float,
    )

    bars = ax.bar(
        x,
        means,
        width=BAR_WIDTH,
        yerr=_log_safe_yerr(means, stds),
        capsize=ERRORBAR_CAPSIZE,
        alpha=BAR_ALPHA,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        error_kw={
            "elinewidth": ERRORBAR_LINEWIDTH,
            "capthick": ERRORBAR_CAPTHICK,
        },
        zorder=2,
    )

    for seed in SEED_ORDER:
        xs, ys = [], []

        for i, method in enumerate(METHODS):
            g = individual[
                (individual["method"] == method)
                & (individual["seed"] == seed)
            ]

            if g.empty:
                continue

            xs.append(
                x[i] + SEED_OFFSETS[seed]
            )
            ys.append(
                float(g.iloc[0][metric])
            )

        if xs:
            ax.scatter(
                xs,
                ys,
                marker=SEED_MARKERS[seed],
                s=SEED_POINT_SIZE,
                facecolors=SEED_COLORS[seed],
                edgecolors="black",
                linewidths=SEED_POINT_EDGE_WIDTH,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABELS[m] for m in METHODS],
        fontsize=PANEL_XTICK_FONT_SIZE,
    )

    ax.tick_params(
        axis="x",
        pad=PANEL_XTICK_PAD,
    )
    ax.tick_params(
        axis="y",
        labelsize=PANEL_YTICK_FONT_SIZE,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=PANEL_YLABEL_FONT_SIZE,
        labelpad=PANEL_YLABEL_PAD,
    )

    ax.set_yscale("log")

    ax.grid(
        True,
        axis="y",
        which="both",
        alpha=PANEL_GRID_ALPHA,
        linewidth=PANEL_GRID_WIDTH,
    )

    ax.set_axisbelow(True)

    ax.set_xlim(
        x[0] - PANEL_X_MARGIN_LEFT,
        x[-1] + PANEL_X_MARGIN_RIGHT,
    )


    y_factor = (
        ANNOTATION_Y_FACTOR_LOSS
        if metric == "total_loss"
        else ANNOTATION_Y_FACTOR_DEFAULT
    )

    for rect, value in zip(bars, means):
        x_text = (
            rect.get_x()
            + rect.get_width() * ANNOTATION_X_FACTOR
        )
        y_text = value * y_factor

        ax.annotate(
            f"{value:.2e}",
            xy=(x_text, y_text),
            xytext=(0, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox=dict(
                boxstyle=f"round,pad={ANNOTATION_BOX_PAD}",
                facecolor="white",
                edgecolor="none",
                alpha=ANNOTATION_BOX_ALPHA,
            ),
            zorder=5,
        )


def plot_combined_final_metric_bars(individual, stats):


    fig, axes = plt.subplots(
        1,
        3,
        figsize=COMBINED_FIGSIZE,
    )

    panels = [
        (
            "total_loss",
            "Total loss",
            "(a) Final total loss",
        ),
        (
            "relative_l2_error",
            r"Relative $L_2$ error",
            r"(b) Relative $L_2$ error",
        ),
        (
            "maximum_absolute_error",
            "Maximum absolute error",
            "(c) Maximum absolute error",
        ),
    ]

    for ax, (metric, ylabel, label_below) in zip(
        axes,
        panels,
    ):
        draw_metric_bar_panel(
            ax,
            individual,
            stats,
            metric,
            ylabel,
            label_below,
        )

        ax.text(
            PANEL_LABEL_X,
            PANEL_LABEL_Y,
            label_below,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONT_SIZE,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=SEED_MARKERS[seed],
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=SEED_COLORS[seed],
            markeredgecolor="black",
            markeredgewidth=LEGEND_MARKER_EDGE_WIDTH,
            label=f"seed {seed}",
        )
        for seed in SEED_ORDER
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=LEGEND_NCOL,
        frameon=LEGEND_FRAME,
        fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, LEGEND_Y),
        borderpad=LEGEND_BORDER_PAD,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        columnspacing=LEGEND_COLUMN_SPACING,
    )

    fig.tight_layout(
        rect=[
            LAYOUT_LEFT,
            LAYOUT_BOTTOM,
            LAYOUT_RIGHT,
            LAYOUT_TOP,
        ],
        w_pad=SUBPLOT_W_PAD,
    )


    mid_bbox = axes[1].get_position()
    mid_center_x = 0.5 * (
        mid_bbox.x0 + mid_bbox.x1
    )

    legend.set_bbox_to_anchor(
        (mid_center_x, LEGEND_Y),
        transform=fig.transFigure,
    )

    out = ANALYSIS_DIR / "allen_cahn_c_ablation_final_metrics_combined.png"

    fig.savefig(
        out,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)
    return out


def main():
    print("Analysis device: CPU only (CUDA disabled for this script)")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    actual = inspect_actual_evals()
    print("\nActual L-BFGS evaluations:")
    print(actual.to_string(index=False))

    stats_map = {m: curve_stats(m) for m in METHODS}
    plot_loss_curves(stats_map)
    individual, stats = build_final_tables()

    combined_bar_figure = plot_combined_final_metric_bars(individual, stats)
    print(f"Combined final-metric figure: {combined_bar_figure}")

    manifest = {
        "experiment": "Allen-Cahn shift-coefficient c ablation in shared neuron-wise AAT",
        "a_only": "X_j = a_j*z_j",
        "full_aat": "X_j = a_j*z_j + c_j",
        "base_activation": "NMAF",
        "seeds": SEEDS,
        "fixed_max_function_evaluations": FINAL_EVAL,
        "loss_curve_center": "geometric mean over seeds",
        "loss_curve_band": "+/- 1 sample std in log10(loss)",
        "final_metric_statistics": "arithmetic mean +/- sample std",
        "only_intended_difference": "presence/absence of shift coefficient c",
        "figures": "800-dpi lossless PNG; loss curve separate; three final metrics combined into one 1x3 figure",
        "analysis_device": "CPU only; CUDA disabled before importing torch",
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nFinal statistics:")
    print(stats.to_string(index=False))
    print(f"\nOutputs: {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
