import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from ac_adaptive5_common import (
    EXPERIMENT_DIR, NEW_TIME_RESULT_ROOT, SEEDS, method_label,
    locate_standard_matched_summary, load_original_nmaf_reference,
)

METHOD_ORDER = ["tanh", "gaaf", "llaaf", "nlaaf", "nmaf_aat"]
NEW_METHODS = ["gaaf", "llaaf", "nlaaf"]


DISPLAY_LABELS = {
    "tanh": "Standard PINN",
    "gaaf": "GAAF",
    "llaaf": "L-LAAF",
    "nlaaf": "N-LAAF",
    "nmaf_aat": "AAT",
}

def display_label(method):
    return DISPLAY_LABELS.get(method, method_label(method))
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis_seedwise_nmaf_time_five_methods"
IMAGE_DPI = 800
SHOW_FIGURES = True
BAR_COLOR = "#C0B3D0"


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.0,
    })


def local_summary(method, seed):
    return NEW_TIME_RESULT_ROOT / method / f"seed_{int(seed)}" / "seedwise_matched_summary.json"


def normalize_summary(obj, source):
    return {
        "method": obj["method"],
        "method_label": display_label(obj["method"]),
        "seed": int(obj["seed"]),
        "source": str(source),
        "target_wall_time_seconds": float(obj["target_wall_time_seconds"]),
        "actual_training_time_seconds": float(obj.get("equivalent_end_to_end_time_seconds", obj.get("actual_stage_training_time_seconds", np.nan))),
        "time_overshoot_seconds": float(obj.get("time_overshoot_seconds", np.nan)),
        "loss": float(obj["monitor_total_loss"]),
        "relative_l2": float(obj["relative_l2_error"]),
        "maximum_absolute_error": float(obj["maximum_absolute_error"]),
        "lbfgs_outer_steps_completed": float(obj.get("lbfgs_outer_steps_completed", np.nan)),
        "lbfgs_closure_evals_completed": float(obj.get("lbfgs_closure_evals_completed", np.nan)),
    }


def load_standard_row(seed):
    path = locate_standard_matched_summary(seed)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("method") != "tanh":
        raise RuntimeError(f"Not a Standard-PINN summary: {path}")
    if int(obj.get("seed")) != int(seed):
        raise RuntimeError(f"Seed mismatch: {path}")
    return normalize_summary(obj, path)


def load_new_row(method, seed):
    path = local_summary(method, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if bool(obj.get("average_target_used", False)):
        raise RuntimeError(f"Averaged target detected in {path}")
    if obj.get("reference_method") != "nmaf_aat" or int(obj.get("reference_seed")) != int(seed):
        raise RuntimeError(f"Invalid same-seed NMAF reference in {path}")
    return normalize_summary(obj, path)


def load_nmaf_row(seed):
    ref = load_original_nmaf_reference(seed)
    return {
        "method": "nmaf_aat",
        "method_label": "AAT",
        "seed": int(seed),
        "source": ref["metric_source"],
        "target_wall_time_seconds": float(ref["target_wall_time_seconds"]),
        "actual_training_time_seconds": float(ref["target_wall_time_seconds"]),
        "time_overshoot_seconds": 0.0,
        "loss": float(ref["nmaf_total_loss"]),
        "relative_l2": float(ref["nmaf_relative_l2_error"]),
        "maximum_absolute_error": float(ref["nmaf_maximum_absolute_error"]),
        "lbfgs_outer_steps_completed": np.nan,
        "lbfgs_closure_evals_completed": 20000.0,
    }


def build_individual_table():
    rows = []
    for seed in SEEDS:
        rows.append(load_standard_row(seed))
    for method in NEW_METHODS:
        for seed in SEEDS:
            rows.append(load_new_row(method, seed))
    for seed in SEEDS:
        rows.append(load_nmaf_row(seed))
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["_order"] = df["method"].map(order)
    df = df.sort_values(["_order", "seed"]).drop(columns="_order").reset_index(drop=True)
    df.to_csv(ANALYSIS_DIR / "seedwise_individual_runs.csv", index=False)
    return df


def build_statistics(df):
    rows = []
    for method in METHOD_ORDER:
        g = df[df["method"] == method]
        row = {"method": method, "method_label": display_label(method), "number_of_seeds": int(g["seed"].nunique())}
        for metric in ["target_wall_time_seconds", "actual_training_time_seconds", "loss", "relative_l2", "maximum_absolute_error"]:
            vals = g[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_sample_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats.to_csv(ANALYSIS_DIR / "seedwise_mean_std_statistics.csv", index=False)

    paper = []
    for _, r in stats.iterrows():
        def fmt(metric):
            return f"{r[f'{metric}_mean']:.6e} ± {r[f'{metric}_sample_std']:.6e}"
        paper.append({
            "Method": r["method_label"],
            "Total loss": fmt("loss"),
            "Relative L2 error": fmt("relative_l2"),
            "Maximum absolute error": fmt("maximum_absolute_error"),
            "Actual training time (s)": fmt("actual_training_time_seconds"),
        })
    pd.DataFrame(paper).to_csv(ANALYSIS_DIR / "paper_ready_seedwise_nmaf_time_table.csv", index=False)
    return stats


def build_paired_seed_table(df):
    rows = []
    for seed in SEEDS:
        nmaf = df[(df["method"] == "nmaf_aat") & (df["seed"] == seed)].iloc[0]
        row = {"seed": seed, "nmaf_target_wall_time_seconds": nmaf["target_wall_time_seconds"]}
        for method in METHOD_ORDER:
            r = df[(df["method"] == method) & (df["seed"] == seed)].iloc[0]
            row[f"{method}_actual_time_seconds"] = r["actual_training_time_seconds"]
            row[f"{method}_loss"] = r["loss"]
            row[f"{method}_relative_l2"] = r["relative_l2"]
            row[f"{method}_max_abs_error"] = r["maximum_absolute_error"]
            if method != "nmaf_aat":
                row[f"{method}_loss_ratio_to_nmaf"] = r["loss"] / nmaf["loss"]
                row[f"{method}_l2_ratio_to_nmaf"] = r["relative_l2"] / nmaf["relative_l2"]
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(ANALYSIS_DIR / "paired_seedwise_results.csv", index=False)
    return paired


def _log_safe_yerr(means, stds):
    means = np.asarray(means, float)
    stds = np.asarray(stds, float)
    return np.vstack([np.minimum(stds, 0.95 * means), stds])


def bar_with_seed_points(df, stats, metric, ylabel, filename, log_y=True):
    methods = METHOD_ORDER
    labels = [display_label(m) for m in methods]
    means = np.array([stats.loc[stats["method"] == m, f"{metric}_mean"].iloc[0] for m in methods])
    stds = np.array([stats.loc[stats["method"] == m, f"{metric}_sample_std"].iloc[0] for m in methods])
    x = np.arange(len(methods), dtype=float)


    fig, ax = plt.subplots(figsize=(7, 5) if metric == "loss" else (8.7, 5.5))
    yerr = _log_safe_yerr(means, stds) if log_y else stds
    bars = ax.bar(
        x, means, yerr=yerr, capsize=5, alpha=0.78,
        color=BAR_COLOR, edgecolor="black", linewidth=0.7,
    )
    seed_offsets = np.linspace(-0.12, 0.12, len(SEEDS))
    for i, method in enumerate(methods):
        g = df[df["method"] == method].set_index("seed")
        for off, seed in zip(seed_offsets, SEEDS):
            ax.scatter(i + off, float(g.loc[seed, metric]), s=28, alpha=0.75, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=14 if metric == "loss" else None)
    if log_y:
        ax.set_yscale("log")

    if metric == "loss":

        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="major", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    else:
        ax.grid(True, axis="y", which="both", alpha=0.35)

    ax.set_axisbelow(True)

    for rect, value in zip(bars, means):
        ax.annotate(
            f"{value:.2e}",
            xy=(rect.get_x() + 0.90 * rect.get_width(), 0.90 * value),
            ha="right", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.7),
        )

    fig.tight_layout()
    out = ANALYSIS_DIR / filename
    fig.savefig(out, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    df = build_individual_table()


    for seed in SEEDS:
        g = df[df["seed"] == seed]
        targets = g["target_wall_time_seconds"].to_numpy(float)
        if not np.allclose(targets, targets[0], rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"Target-time mismatch for seed={seed}: {targets}")

    stats = build_statistics(df)
    build_paired_seed_table(df)
    bar_with_seed_points(df, stats, "loss", "Total loss", "seedwise_nmaf_time_total_loss_bar.png", log_y=True)
    bar_with_seed_points(df, stats, "relative_l2", r"Relative $L_2$ error", "seedwise_nmaf_time_relative_l2_bar.png", log_y=True)
    bar_with_seed_points(df, stats, "maximum_absolute_error", "Maximum absolute error", "seedwise_nmaf_time_max_abs_error_bar.png", log_y=True)

    manifest = {
        "problem": "Allen-Cahn",
        "methods": [display_label(m) for m in METHOD_ORDER],
        "protocol": "same-seed AAT matched wall-clock",
        "average_target_used": False,
        "nmaf_retrained": False,
        "standard_result": "reused from previous AC seedwise_NMAF_matched_time_efficiency when available",
        "newly_trained": ["GAAF", "L-LAAF", "N-LAAF"],
        "base_activation_for_new_adaptive_methods": "NMAF",
        "slope_recovery": False,
        "statistics": "arithmetic mean +/- sample std with three seed points",
        "plot_format": "lossless PNG, 800 dpi",
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(stats.to_string(index=False))
    print(f"\nOutputs: {ANALYSIS_DIR.resolve()}")

if __name__ == "__main__":
    main()
