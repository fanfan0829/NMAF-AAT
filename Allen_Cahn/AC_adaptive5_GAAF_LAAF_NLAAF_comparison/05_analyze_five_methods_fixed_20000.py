from pathlib import Path
import json
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
import torch
import importlib

import ac_adaptive5_common as ac_common


ac_common = importlib.reload(ac_common)

from ac_adaptive5_common import (
    EXPERIMENT_DIR, SEEDS, method_label, build_model,
    fixed_history_path, fixed_final_checkpoint, evaluate_model,
)


CPU_DEVICE = torch.device("cpu")
ac_common.DEVICE = CPU_DEVICE
DEVICE = CPU_DEVICE

def _move_to_cpu(obj):


    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, torch.device) and obj.type == "cuda":
        return CPU_DEVICE
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj


def _force_common_globals_to_cpu():


    ac_common.DEVICE = CPU_DEVICE
    for name, value in list(ac_common.__dict__.items()):
        if name.startswith('__'):
            continue
        try:
            new_value = _move_to_cpu(value)
            if new_value is not value:
                ac_common.__dict__[name] = new_value


            if callable(value) and hasattr(value, "__defaults__"):
                if value.__defaults__ is not None:
                    value.__defaults__ = tuple(_move_to_cpu(v) for v in value.__defaults__)
                if getattr(value, "__kwdefaults__", None):
                    value.__kwdefaults__ = {
                        k: _move_to_cpu(v) for k, v in value.__kwdefaults__.items()
                    }
        except Exception:
            pass


def _force_model_cpu(model):

    model = model.to(CPU_DEVICE)
    for module in model.modules():
        for name, value in list(vars(module).items()):
            if name in {'_parameters', '_buffers', '_modules'}:
                continue
            try:
                new_value = _move_to_cpu(value)
                if new_value is not value:
                    setattr(module, name, new_value)
            except Exception:
                pass
    return model


_force_common_globals_to_cpu()

METHOD_ORDER = ["tanh", "gaaf", "llaaf", "nlaaf", "nmaf_aat"]


DISPLAY_LABELS = {
    "tanh": "Standard PINN",
    "gaaf": "GAAF",
    "llaaf": "L-LAAF",
    "nlaaf": "N-LAAF",
    "nmaf_aat": "AAT",
}

def display_label(method):
    return DISPLAY_LABELS.get(method, method_label(method))
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis_fixed_20000_five_methods"
IMAGE_DPI = 800
SHOW_FIGURES = True
BAR_COLOR = "#C0B3D0"
FINAL_EVAL = 20000
LOSS_YMIN = 1.0e-6
LOSS_YMAX = 1.0e2


LOSS_COLORS = {
    "tanh": "#1f77b4",
    "gaaf": "#ff7f0e",
    "llaaf": "#2ca02c",
    "nlaaf": "#9467bd",
    "nmaf_aat": "#d62728",
}

LINESTYLES = {
    "tanh": "-",
    "gaaf": "--",
    "llaaf": "-.",
    "nlaaf": ":",
    "nmaf_aat": (0, (5, 2)),
}
MARKERS = {"tanh": "o", "gaaf": "s", "llaaf": "^", "nlaaf": "D", "nmaf_aat": "P"}


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


def load_history(method, seed):
    path = fixed_history_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "lbfgs_eval" not in df.columns or "total_loss" not in df.columns:
        raise ValueError(f"Unexpected history columns in {path}: {list(df.columns)}")
    df = df[["lbfgs_eval", "total_loss"]].copy()
    df["lbfgs_eval"] = pd.to_numeric(df["lbfgs_eval"], errors="coerce")
    df["total_loss"] = pd.to_numeric(df["total_loss"], errors="coerce")
    df = df.dropna()
    df = df[(df["lbfgs_eval"] >= 1) & (df["lbfgs_eval"] <= FINAL_EVAL)]
    df["lbfgs_eval"] = df["lbfgs_eval"].astype(int)
    return df.drop_duplicates("lbfgs_eval", keep="last")


def method_curve_stats(method):
    frames = []
    for seed in SEEDS:
        f = load_history(method, seed).rename(columns={"total_loss": f"loss_seed_{seed}"})
        frames.append(f)
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="lbfgs_eval", how="inner")
    if merged.empty:
        raise RuntimeError(f"No common evaluations for {method}")
    values = merged[[f"loss_seed_{s}" for s in SEEDS]].to_numpy(float)
    if np.any(values <= 0) or np.any(~np.isfinite(values)):
        raise ValueError(f"Invalid loss values for {method}")
    logs = np.log10(values)
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
    out.to_csv(ANALYSIS_DIR / f"{method}_total_loss_log_statistics.csv", index=False)
    return out


def plot_loss_curves(stats_map):


    with plt.rc_context({
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "font.family": "Times New Roman",
        "font.size": 12,
    }):
        fig, ax = plt.subplots(figsize=(7, 5))


        for method in METHOD_ORDER:
            frame = stats_map[method]
            line, = ax.semilogy(
                frame["lbfgs_eval"],
                frame["geometric_mean_total_loss"],
                linewidth=1.5,
                color=LOSS_COLORS[method],
                label=display_label(method),
                zorder=3,
            )
            ax.fill_between(
                frame["lbfgs_eval"],
                frame["lower_band"],
                frame["upper_band"],
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
                zorder=1,
            )


        ax.yaxis.set_major_locator(
            LogLocator(base=10.0, numticks=20)
        )
        ax.yaxis.set_minor_locator(
            LogLocator(
                base=10.0,
                subs=np.arange(2, 10) * 0.1,
                numticks=100,
            )
        )
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.set_xlabel("Iteration", fontsize=14)
        ax.set_ylabel("Total loss", fontsize=14)
        ax.set_xlim(1, FINAL_EVAL)


        if LOSS_YMIN is not None and LOSS_YMAX is not None:
            if LOSS_YMIN <= 0 or LOSS_YMAX <= 0:
                raise ValueError("For a logarithmic y-axis, LOSS_YMIN and LOSS_YMAX must be > 0.")
            if LOSS_YMIN >= LOSS_YMAX:
                raise ValueError("LOSS_YMIN must be smaller than LOSS_YMAX.")
            ax.set_ylim(LOSS_YMIN, LOSS_YMAX)

        ax.legend(loc="upper right", fontsize=10)


        ax.grid(True, which="major", linestyle="-", alpha=0.6)
        ax.grid(True, which="minor", linestyle="--", alpha=0.3)

        plt.tight_layout()

        out = ANALYSIS_DIR / "allen_cahn_adaptive5_total_loss_mean_std.png"
        fig.savefig(out, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)
        return out


def load_final_model_metrics(method, seed):
    path = fixed_final_checkpoint(method, seed)
    ck = torch.load(path, map_location=CPU_DEVICE, weights_only=False)
    _force_common_globals_to_cpu()
    model = build_model(method)
    model = _force_model_cpu(model)
    cpu_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(cpu_state, strict=True)
    model = _force_model_cpu(model)
    _force_common_globals_to_cpu()
    metrics = ac_common.evaluate_model(model)
    hist = load_history(method, seed)
    final_row = hist.loc[hist["lbfgs_eval"] <= FINAL_EVAL].iloc[-1]
    return {
        "method": method,
        "method_label": display_label(method),
        "seed": int(seed),
        "final_eval": int(final_row["lbfgs_eval"]),
        "total_loss": float(final_row["total_loss"]),
        "relative_l2_error": float(metrics["relative_l2_error"]),
        "maximum_absolute_error": float(metrics["maximum_absolute_error"]),
        "checkpoint": str(path),
    }


def build_final_tables():
    rows = [load_final_model_metrics(m, s) for m in METHOD_ORDER for s in SEEDS]
    individual = pd.DataFrame(rows)
    individual.to_csv(ANALYSIS_DIR / "individual_seed_final_metrics.csv", index=False)

    stats_rows = []
    for method in METHOD_ORDER:
        g = individual[individual["method"] == method]
        row = {"method": method, "method_label": display_label(method), "number_of_seeds": len(g)}
        for metric in ["total_loss", "relative_l2_error", "maximum_absolute_error"]:
            vals = g[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sample_std"] = float(vals.std(ddof=1))
        stats_rows.append(row)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(ANALYSIS_DIR / "mean_std_final_metrics.csv", index=False)

    paper = []
    for _, r in stats.iterrows():
        paper.append({
            "Method": r["method_label"],
            "Final total loss": f"{r['total_loss_mean']:.6e} ± {r['total_loss_sample_std']:.6e}",
            "Relative L2 error": f"{r['relative_l2_error_mean']:.6e} ± {r['relative_l2_error_sample_std']:.6e}",
            "Maximum absolute error": f"{r['maximum_absolute_error_mean']:.6e} ± {r['maximum_absolute_error_sample_std']:.6e}",
        })
    pd.DataFrame(paper).to_csv(ANALYSIS_DIR / "paper_ready_fixed_20000_table.csv", index=False)
    return individual, stats


def _log_safe_yerr(means, stds):
    means = np.asarray(means, float)
    stds = np.asarray(stds, float)
    return np.vstack([np.minimum(stds, 0.95 * means), stds])


def bar_with_seed_points(individual, stats, metric, ylabel, filename):
    methods = METHOD_ORDER
    labels = [display_label(m) for m in methods]
    x = np.arange(len(methods), dtype=float)
    means = np.array([
        stats.loc[stats["method"] == m, f"{metric}_mean"].iloc[0]
        for m in methods
    ])
    stds = np.array([
        stats.loc[stats["method"] == m, f"{metric}_sample_std"].iloc[0]
        for m in methods
    ])


    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    bars = ax.bar(
        x,
        means,
        yerr=_log_safe_yerr(means, stds),
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
    )


    offsets = {2026: -0.12, 99: 0.0, 1234: 0.12}
    markers = {2026: "o", 99: "s", 1234: "^"}
    fallback_offsets = np.linspace(-0.12, 0.12, len(SEEDS))
    fallback_markers = ["o", "s", "^"]

    for seed_index, seed in enumerate(SEEDS):
        xs, ys = [], []
        for i, method in enumerate(methods):
            g = individual[
                (individual["method"] == method)
                & (individual["seed"] == seed)
            ]
            if not g.empty:
                offset = offsets.get(seed, fallback_offsets[seed_index])
                xs.append(i + offset)
                ys.append(float(g.iloc[0][metric]))

        if xs:
            ax.scatter(
                xs,
                ys,
                marker=markers.get(seed, fallback_markers[seed_index % 3]),
                s=40,
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.35)
    ax.legend(frameon=True, ncol=3)


    for rect, value in zip(bars, means):
        x_text = rect.get_x() + rect.get_width() * 0.90
        y_text = value * 0.90
        ax.annotate(
            f"{value:.2e}",
            xy=(x_text, y_text),
            xytext=(0, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=5,
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
    stats_map = {m: method_curve_stats(m) for m in METHOD_ORDER}
    loss_path = plot_loss_curves(stats_map)
    individual, stats = build_final_tables()
    bar_with_seed_points(individual, stats, "total_loss", "Total loss", "fixed20000_total_loss_bar.png")
    bar_with_seed_points(individual, stats, "relative_l2_error", r"Relative $L_2$ error", "fixed20000_relative_l2_bar.png")
    bar_with_seed_points(individual, stats, "maximum_absolute_error", "Maximum absolute error", "fixed20000_max_abs_error_bar.png")
    manifest = {
        "protocol": "fixed 20000 L-BFGS closure/function evaluations",
        "methods": [display_label(m) for m in METHOD_ORDER],
        "seeds": SEEDS,
        "loss_curve_center": "geometric mean across seeds",
        "loss_curve_band": "+/- 1 sample std in log10(loss)",
        "final_table_statistics": "arithmetic mean +/- sample std",
        "slope_recovery_for_gaaf_llaaf_nlaaf": False,
        "base_activation_for_gaaf_llaaf_nlaaf": "NMAF",
        "loss_figure": str(loss_path),
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(stats.to_string(index=False))
    print(f"\nOutputs: {ANALYSIS_DIR.resolve()}")

if __name__ == "__main__":
    main()
