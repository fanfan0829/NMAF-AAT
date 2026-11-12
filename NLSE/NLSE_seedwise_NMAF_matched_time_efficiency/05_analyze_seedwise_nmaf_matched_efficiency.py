from __future__ import annotations

import json
from pathlib import Path
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlse_seedwise_common import (
    EXPERIMENT_DIR,
    NEW_RESULT_ROOT,
    PACKAGE_VERSION,
    SEEDS,
    export_nmaf_seedwise_targets,
    load_original_nmaf_seed_reference,
    method_label,
)

ANALYSIS_DIR = EXPERIMENT_DIR / "analysis_seedwise_matched_to_nmaf"
IMAGE_DPI = 800
SHOW_FIGURES = True
METHOD_ORDER = ["tanh", "rar", "stan", "nmaf_aat"]
COMPETING_METHODS = ["tanh", "rar", "stan"]


BAR_COLOR = "#C0B3D0"


SEED_PLOT_ORDER = [2026, 99, 1234]
SEED_OFFSETS = {2026: -0.12, 99: 0.0, 1234: 0.12}
SEED_MARKERS = {2026: "o", 99: "s", 1234: "^"}
SEED_COLORS = {
    2026: "#1f77b4",
    99: "#ff7f0e",
    1234: "#2ca02c",
}
PLOT_LABELS = {
    "tanh": "Standard PINN",
    "rar": "RAR-PINN",
    "stan": "Stan-PINN",
    "nmaf_aat": "NMAF-AAT",
}


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


def summary_path(method: str, seed: int) -> Path:
    return NEW_RESULT_ROOT / method / f"seed_{seed}" / "seedwise_matched_summary.json"


def available_competing_methods():
    available = []
    print("\nAvailable NLSE seed-wise matched results")
    print("-" * 100)
    for method in COMPETING_METHODS:
        missing = [seed for seed in SEEDS if not summary_path(method, seed).exists()]
        if missing:
            print(f"SKIPPED {method_label(method):20s} | missing seeds: {missing}")
        else:
            print(f"USING   {method_label(method):20s} | all 3 seeds available")
            available.append(method)
    print("USING   NMAF-AAT             | original formal 3-seed results (not retrained)")
    print("-" * 100)
    return available


def load_competing_row(method: str, seed: int) -> dict:
    path = summary_path(method, seed)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("reference_method") != "nmaf_aat":
        raise RuntimeError(f"Invalid reference method in {path}")
    if int(obj.get("reference_seed")) != int(seed):
        raise RuntimeError(f"Seed pairing mismatch in {path}")
    if bool(obj.get("average_target_used", False)):
        raise RuntimeError(f"Average-time result is forbidden in this analysis: {path}")
    return {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "source": "new same-seed NMAF-matched wall-clock run",
        "target_wall_time_seconds": float(obj["target_wall_time_seconds"]),
        "actual_training_time_seconds": float(obj["equivalent_end_to_end_time_seconds"]),
        "time_overshoot_seconds": float(obj["time_overshoot_seconds"]),
        "loss": float(obj["monitor_total_loss"]),
        "optimization_loss": float(obj["optimization_total_loss"]),
        "relative_l2": float(obj["relative_l2_error"]),
        "maximum_absolute_error": float(obj["maximum_absolute_error"]),
        "lbfgs_outer_steps_completed": float(obj["lbfgs_outer_steps_completed"]),
        "lbfgs_closure_evals_completed": float(obj["lbfgs_closure_evals_completed"]),
        "final_n_res_training": float(obj["final_n_res_training"]),
        "rar_selection_overhead_seconds": (
            float(obj["rar_selection_overhead_seconds_applied"])
            if obj.get("rar_selection_overhead_seconds_applied") is not None else np.nan
        ),
        "rar_selection_overhead_known": obj.get("rar_selection_overhead_known"),
        "rar_timing_scope": obj.get("rar_timing_scope"),
    }


def _old_nmaf_final_eval(seed: int):
    path = EXPERIMENT_DIR.parent / "nlse_multiseed_results" / "nmaf_aat" / f"seed_{seed}" / "training_summary.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if "final_actual_lbfgs_eval" in obj:
                return float(obj["final_actual_lbfgs_eval"])
        except Exception:
            pass
    return 10000.0


def load_nmaf_row(seed: int) -> dict:
    ref = load_original_nmaf_seed_reference(seed)
    return {
        "method": "nmaf_aat",
        "method_label": "NMAF-AAT",
        "seed": int(seed),
        "source": "original formal NMAF-AAT run",
        "target_wall_time_seconds": float(ref["target_wall_time_seconds"]),
        "actual_training_time_seconds": float(ref["target_wall_time_seconds"]),
        "time_overshoot_seconds": 0.0,
        "loss": float(ref["nmaf_total_loss"]),
        "optimization_loss": float(ref["nmaf_total_loss"]),
        "relative_l2": float(ref["nmaf_relative_l2_error"]),
        "maximum_absolute_error": float(ref["nmaf_maximum_absolute_error"]),
        "lbfgs_outer_steps_completed": np.nan,
        "lbfgs_closure_evals_completed": _old_nmaf_final_eval(seed),
        "final_n_res_training": 10000.0,
        "rar_selection_overhead_seconds": np.nan,
        "rar_selection_overhead_known": np.nan,
        "rar_timing_scope": None,
    }


def build_individual_table(available):
    rows = [load_nmaf_row(seed) for seed in SEEDS]
    for method in available:
        rows.extend(load_competing_row(method, seed) for seed in SEEDS)
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["_order"] = df["method"].map(order)
    df = df.sort_values(["_order", "seed"]).drop(columns="_order").reset_index(drop=True)
    df.to_csv(ANALYSIS_DIR / "seedwise_individual_runs.csv", index=False)
    return df


def mean_std(values):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(v) == 0:
        return np.nan, np.nan, 0
    if len(v) == 1:
        return float(v.iloc[0]), np.nan, 1
    return float(v.mean()), float(v.std(ddof=1)), int(len(v))


def build_statistics(df):
    metrics = [
        "target_wall_time_seconds",
        "actual_training_time_seconds",
        "time_overshoot_seconds",
        "loss",
        "optimization_loss",
        "relative_l2",
        "maximum_absolute_error",
        "lbfgs_outer_steps_completed",
        "lbfgs_closure_evals_completed",
    ]
    rows = []
    for method, group in df.groupby("method", sort=False):
        row = {
            "method": method,
            "method_label": method_label(method),
            "number_of_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            m, s, n = mean_std(group[metric])
            row[f"{metric}_mean"] = m
            row[f"{metric}_sample_std"] = s
            row[f"{metric}_valid_n"] = n
        if method == "rar":
            known = group["rar_selection_overhead_known"].fillna(False).astype(bool)
            row["rar_selection_overhead_known_for_all_seeds"] = bool(known.all())
        rows.append(row)

    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    out["_order"] = out["method"].map(order)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out.to_csv(ANALYSIS_DIR / "seedwise_mean_std_statistics.csv", index=False)

    paper = []
    for _, r in out.iterrows():
        def fmt(metric):
            m = r[f"{metric}_mean"]
            s = r[f"{metric}_sample_std"]
            if pd.isna(m):
                return "NA"
            return f"{m:.6e} ± {s:.6e}" if pd.notna(s) else f"{m:.6e}"
        row = {
            "Method": r["method_label"],
            "Seeds": int(r["number_of_seeds"]),
            "NMAF-matched target time (s), descriptive mean ± std": fmt("target_wall_time_seconds"),
            "Common total loss": fmt("loss"),
            "Relative L2 error": fmt("relative_l2"),
            "Maximum absolute error": fmt("maximum_absolute_error"),
            "Actual/equivalent end-to-end time (s)": fmt("actual_training_time_seconds"),
        }
        if r["method"] == "rar":
            row["RAR timing note"] = (
                "selection overhead included/deducted for all seeds"
                if bool(r.get("rar_selection_overhead_known_for_all_seeds", False))
                else "selection overhead unavailable for >=1 seed; RAR timing is optimistic Stage-B-only for those seeds"
            )
        paper.append(row)
    pd.DataFrame(paper).to_csv(
        ANALYSIS_DIR / "seedwise_paper_ready_mean_std_table.csv", index=False
    )
    return out


def build_paired_seed_table(df):
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    rows = []
    for seed in SEEDS:
        nmaf = df[(df["method"] == "nmaf_aat") & (df["seed"] == seed)].iloc[0]
        row = {
            "seed": seed,
            "nmaf_target_wall_time_seconds": nmaf["target_wall_time_seconds"],
        }
        for method in methods:
            g = df[(df["method"] == method) & (df["seed"] == seed)]
            if g.empty:
                continue
            r = g.iloc[0]
            row[f"{method}_actual_time_seconds"] = r["actual_training_time_seconds"]
            row[f"{method}_common_loss"] = r["loss"]
            row[f"{method}_relative_l2"] = r["relative_l2"]
            row[f"{method}_max_abs_error"] = r["maximum_absolute_error"]
            row[f"{method}_lbfgs_closure_evals"] = r["lbfgs_closure_evals_completed"]
            if method != "nmaf_aat":
                row[f"{method}_loss_ratio_to_nmaf"] = r["loss"] / nmaf["loss"]
                row[f"{method}_l2_ratio_to_nmaf"] = r["relative_l2"] / nmaf["relative_l2"]
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(ANALYSIS_DIR / "paired_seedwise_results.csv", index=False)
    return paired


def export_target_table():
    rows = []
    for seed in SEEDS:
        ref = load_original_nmaf_seed_reference(seed)
        rows.append({
            "seed": seed,
            "target_wall_time_seconds": ref["target_wall_time_seconds"],
            "reference_method": "NMAF-AAT",
            "nmaf_total_loss": ref["nmaf_total_loss"],
            "nmaf_relative_l2_error": ref["nmaf_relative_l2_error"],
            "nmaf_maximum_absolute_error": ref["nmaf_maximum_absolute_error"],
            "time_source": ref["time_source"],
            "metric_source": ref["metric_source"],
        })
    pd.DataFrame(rows).to_csv(ANALYSIS_DIR / "nmaf_seedwise_target_times.csv", index=False)


def _log_safe_yerr(means, stds):
    means = np.asarray(means, float)
    stds = np.asarray(stds, float)
    lower = np.minimum(stds, 0.95 * means)
    return np.vstack([lower, stds])


def bar_with_seed_points(df, stats, metric, ylabel, filename, log_y=True):
    methods = stats["method"].tolist()
    labels = [PLOT_LABELS.get(m, method_label(m)) for m in methods]
    means = stats[f"{metric}_mean"].to_numpy(float)
    stds = stats[f"{metric}_sample_std"].to_numpy(float)
    x = np.arange(len(methods), dtype=float)

    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    yerr = _log_safe_yerr(means, stds) if log_y else stds
    bars = ax.bar(
        x, means, yerr=yerr, capsize=5, alpha=0.78,
        color=BAR_COLOR, edgecolor="black", linewidth=0.7,
    )


    for seed in SEED_PLOT_ORDER:
        if seed not in set(pd.to_numeric(df["seed"], errors="coerce").dropna().astype(int)):
            continue

        xs, ys = [], []
        for i, method in enumerate(methods):
            g = df[(df["method"] == method) & (df["seed"] == seed)]
            if not g.empty:
                xs.append(i + SEED_OFFSETS[seed])
                ys.append(float(g.iloc[0][metric]))

        if xs:
            ax.scatter(
                xs,
                ys,
                marker=SEED_MARKERS[seed],
                s=40,
                facecolors=SEED_COLORS[seed],
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.35)
    ax.legend(frameon=True, ncol=3)

    for rect, value in zip(bars, means):
        x_text = rect.get_x() + rect.get_width() * 0.90
        y_text = value * 0.90
        ax.annotate(
            f"{value:.2e}",
            xy=(x_text, y_text),
            xytext=(0, 0), textcoords="offset points",
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=5,
        )

    fig.tight_layout()
    out = ANALYSIS_DIR / filename
    fig.savefig(out, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    print(f"Saved: {out.resolve()}")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def print_rar_timing_warning(df):
    rar = df[df["method"] == "rar"]
    if rar.empty:
        return
    known = rar["rar_selection_overhead_known"].fillna(False).astype(bool)
    if not known.all():
        bad = rar.loc[~known, "seed"].astype(int).tolist()
        print("\nWARNING ABOUT NLSE RAR EFFICIENCY TIMING")
        print("-" * 100)
        print(
            "The original formal NLSE RAR protocol uses a preselected 12000-point Stage-B set. "
            "No recorded Stage-A/selection wall time was found for seeds: " + str(bad)
        )
        print(
            "For those seeds this package gives RAR the FULL NMAF target as Stage-B training time. "
            "That is optimistic in favor of RAR, not in favor of NMAF-AAT."
        )
        print(
            "For a strict end-to-end RAR timing claim, enter the measured selection times in "
            "RAR_SELECTION_OVERHEAD_SECONDS_OVERRIDE in nlse_seedwise_common.py and rerun RAR."
        )
        print("-" * 100)


def main():
    print(f"Package version: {PACKAGE_VERSION}")
    print(f"Analysis file actually executed: {Path(__file__).resolve()}")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    targets = export_nmaf_seedwise_targets()
    export_target_table()
    available = available_competing_methods()
    df = build_individual_table(available)
    stats = build_statistics(df)
    build_paired_seed_table(df)
    print_rar_timing_warning(df)

    print("\n" + "=" * 126)
    print("NLSE: NMAF-AAT SEED-WISE MATCHED WALL-CLOCK EFFICIENCY")
    print("No averaged target is used. Each seed uses the final time of the original NMAF-AAT run with the SAME seed.")
    print("-" * 126)
    for seed in SEEDS:
        print(
            f"seed {seed:4d}: target = "
            f"{targets['target_wall_time_seconds_by_seed'][str(seed)]:.6f} s"
        )
    print("-" * 126)
    print(stats[[
        "method_label", "number_of_seeds",
        "target_wall_time_seconds_mean", "target_wall_time_seconds_sample_std",
        "loss_mean", "loss_sample_std",
        "relative_l2_mean", "relative_l2_sample_std",
    ]].to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("=" * 126)

    bar_with_seed_points(
        df, stats, "loss", "Total loss",
        "nlse_seedwise_nmaf_matched_total_loss_bar_points.png", log_y=True,
    )
    bar_with_seed_points(
        df, stats, "relative_l2", r"Relative $L_2$ error",
        "nlse_seedwise_nmaf_matched_relative_l2_bar_points.png", log_y=True,
    )
    bar_with_seed_points(
        df, stats, "maximum_absolute_error", "Maximum absolute error",
        "nlse_seedwise_nmaf_matched_max_abs_error_bar_points.png", log_y=True,
    )

    manifest = {
        "package_version": PACKAGE_VERSION,
        "problem": "Nonlinear Schrodinger equation",
        "protocol": "seed-wise matched wall-clock budget",
        "reference_method": "NMAF-AAT",
        "training_target_definition": {
            str(seed): f"original NMAF-AAT seed {seed} final elapsed time" for seed in SEEDS
        },
        "average_target_used_for_training": False,
        "nmaf_retrained": False,
        "nmaf_results": "reused from original formal NLSE runs",
        "competing_methods_available": available,
        "formal_optimizer": "pure L-BFGS (actual executable history_size=50)",
        "rar_training_set": "formal preselected 12000-point set (10000 original + 2000 selected)",
        "rar_primary_comparison_loss": "monitor total loss on original shared 10000 PDE points",
        "rar_optimization_loss": "12000-point objective, diagnostic only",
        "rar_selection_overhead_policy": (
            "deduct detected/overridden Stage-A selection wall time from same-seed NMAF target; "
            "if unavailable, flag RAR as optimistic Stage-B-only"
        ),
        "metrics_evaluated_after_training_timer_stops": True,
        "primary_figures": "bar + sample-std error bar + 3 individual seed points",
        "seed_plot_style": "AC reference: seed 2026 blue circle; seed 99 orange square; seed 1234 green triangle; legend order 2026,99,1234",
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
