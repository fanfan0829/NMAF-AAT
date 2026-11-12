from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "wave_common_fixed_points_adam_lbfgs.py"
_spec = importlib.util.spec_from_file_location("wave_common_local_v2", COMMON_PATH)
W = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = W
_spec.loader.exec_module(W)

CPU = torch.device("cpu")
W.DEVICE = CPU
RESULT_ROOT = SCRIPT_DIR / "wave_multiseed_results"
OUT = RESULT_ROOT / "wave_cpu_postprocessing" / "01_frequency_resolved_error_paper_style"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [int(s) for s in W.SEEDS]
METHODS = ["tanh", "nmaf_aat", "rar", "stan"]
METHOD_LABELS = {
    "tanh": "Standard PINN",
    "nmaf_aat": "NMAF-AAT",
    "rar": "RAR-PINN",
    "stan": "Stan-PINN",
}
CHECKPOINTS = [spec["name"] for spec in W.CHECKPOINT_SPECS]
CHECKPOINT_LABELS = {
    "adam0500000": "Adam 500k",
    "total0700000": "Adam 500k + L-BFGS",
}
RAR_METHOD_DIR_NAME = "rar_adam_integrated_base6400_plus1280"
IMAGE_DPI = 800
SHOW_FIGURES = True
N_X = 401
N_T = 401

def configure_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.7,
    })


BAR_COLOR_LOW = "#C0B3D0"
BAR_COLOR_HIGH = "#C97B7B"

SEED_ORDER = [2026, 99, 1234]
SEED_OFFSETS = {2026: -0.12, 99: 0.0, 1234: 0.12}
SEED_MARKERS = {2026: "o", 99: "s", 1234: "^"}
SEED_COLORS = {2026: "#1f77b4", 99: "#ff7f0e", 1234: "#2ca02c"}

def _log_safe_yerr(means, stds):
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    lower = np.minimum(stds, 0.95 * np.maximum(means, 1.0e-300))
    return np.vstack([lower, stds])

def _ordered_unique_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    keep = {}
    for h, lab in zip(handles, labels):
        if lab not in keep:
            keep[lab] = h
    return list(keep.values()), list(keep.keys())

def _annotate_bar_tops(ax, bars, values, fmt="{:.2e}", log_y=False):
    for rect, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        if log_y:
            y_text = value * 0.90
            va = "top"
        else:
            y_text = value * 0.95 if value >= 0 else value * 1.05
            va = "top" if value >= 0 else "bottom"
        x_text = rect.get_x() + rect.get_width() * 0.90
        ax.annotate(
            fmt.format(value),
            xy=(x_text, y_text),
            xytext=(0, 0),
            textcoords="offset points",
            ha="right",
            va=va,
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=6,
        )

def grouped_two_bar_with_seed_points(
    ax,
    categories,
    mean_low,
    std_low,
    mean_high,
    std_high,
    seedwise_low,
    seedwise_high,
    ylabel,
    low_label,
    high_label,
    log_y=True,
):
    x = np.arange(len(categories), dtype=float)
    width = 0.34

    yerr_low = _log_safe_yerr(mean_low, std_low) if log_y else std_low
    yerr_high = _log_safe_yerr(mean_high, std_high) if log_y else std_high

    bars_low = ax.bar(
        x - width / 2,
        mean_low,
        width,
        yerr=yerr_low,
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR_LOW,
        edgecolor="black",
        linewidth=0.7,
        label=low_label,
        zorder=2,
    )
    bars_high = ax.bar(
        x + width / 2,
        mean_high,
        width,
        yerr=yerr_high,
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR_HIGH,
        edgecolor="black",
        linewidth=0.7,
        label=high_label,
        zorder=2,
    )

    for seed in SEED_ORDER:
        xs, ys = [], []
        for i in range(len(categories)):
            v = seedwise_low.get((i, seed), np.nan)
            if np.isfinite(v):
                xs.append((x - width / 2)[i] + SEED_OFFSETS[seed] * 0.40)
                ys.append(v)
        if xs:
            ax.scatter(
                xs, ys,
                marker=SEED_MARKERS[seed],
                s=40,
                facecolors=SEED_COLORS[seed],
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )

    for seed in SEED_ORDER:
        xs, ys = [], []
        for i in range(len(categories)):
            v = seedwise_high.get((i, seed), np.nan)
            if np.isfinite(v):
                xs.append((x + width / 2)[i] + SEED_OFFSETS[seed] * 0.40)
                ys.append(v)
        if xs:
            ax.scatter(
                xs, ys,
                marker=SEED_MARKERS[seed],
                s=40,
                facecolors=SEED_COLORS[seed],
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.35)

    _annotate_bar_tops(ax, bars_low, mean_low, fmt="{:.2e}", log_y=log_y)
    _annotate_bar_tops(ax, bars_high, mean_high, fmt="{:.2e}", log_y=log_y)

    handles, labels = _ordered_unique_legend(ax)
    preferred = [low_label, high_label] + [f"seed {s}" for s in SEED_ORDER]
    ordered_h, ordered_l = [], []
    for lab in preferred:
        if lab in labels:
            ordered_h.append(handles[labels.index(lab)])
            ordered_l.append(lab)
    ax.legend(ordered_h, ordered_l, frameon=True, ncol=2)

def single_bar_with_seed_points(
    ax,
    categories,
    means,
    stds,
    seedwise_map,
    ylabel,
    legend_title=None,
    log_y=True,
):
    x = np.arange(len(categories), dtype=float)
    yerr = _log_safe_yerr(means, stds) if log_y else stds

    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR_LOW,
        edgecolor="black",
        linewidth=0.7,
        zorder=2,
        label=legend_title if legend_title is not None else None,
    )

    for seed in SEED_ORDER:
        xs, ys = [], []
        for i in range(len(categories)):
            v = seedwise_map.get((i, seed), np.nan)
            if np.isfinite(v):
                xs.append(x[i] + SEED_OFFSETS[seed])
                ys.append(v)
        if xs:
            ax.scatter(
                xs, ys,
                marker=SEED_MARKERS[seed],
                s=40,
                facecolors=SEED_COLORS[seed],
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.35)

    _annotate_bar_tops(ax, bars, means, fmt="{:.2e}", log_y=log_y)

    handles, labels = _ordered_unique_legend(ax)
    preferred = ([legend_title] if legend_title is not None else []) + [f"seed {s}" for s in SEED_ORDER]
    ordered_h, ordered_l = [], []
    for lab in preferred:
        if lab in labels:
            ordered_h.append(handles[labels.index(lab)])
            ordered_l.append(lab)
    if ordered_h:
        ax.legend(ordered_h, ordered_l, frameon=True, ncol=min(2, len(ordered_h)))


def mean_std(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, 0
    if len(s) == 1:
        return float(s.iloc[0]), np.nan, 1
    return float(s.mean()), float(s.std(ddof=1)), int(len(s))

def rar_dir():
    preferred = RESULT_ROOT / RAR_METHOD_DIR_NAME
    if preferred.is_dir():
        return preferred
    candidates = []
    if RESULT_ROOT.exists():
        for p in RESULT_ROOT.iterdir():
            if p.is_dir() and "rar" in p.name.lower():
                ok = all((p / f"seed_{s}").is_dir() for s in SEEDS)
                if ok:
                    candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError("Cannot uniquely locate Wave RAR directory.")

def checkpoint_path(method, seed, checkpoint):
    if method == "rar":
        sd = rar_dir() / f"seed_{seed}"
        exact = sd / f"wave_{RAR_METHOD_DIR_NAME}_seed{seed}_{checkpoint}.pth"
        if exact.exists():
            return exact
        matches = sorted(sd.glob(f"wave_*_seed{seed}_{checkpoint}.pth"))
        rar_matches = [p for p in matches if "rar" in p.name.lower()]
        if len(rar_matches) == 1:
            return rar_matches[0]
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"Ambiguous RAR checkpoint: {matches}")
    return RESULT_ROOT / method / f"seed_{seed}" / f"wave_{method}_seed{seed}_{checkpoint}.pth"

def build_model(method):
    if method in ("tanh", "rar"):
        model = W.TanhPINN(W.HIDDEN_LAYERS, W.HIDDEN_NEURONS)
    elif method == "nmaf_aat":
        model = W.NMAFAATPINN(W.HIDDEN_LAYERS, W.HIDDEN_NEURONS, dtype=W.DTYPE_LBFGS)
    elif method == "stan":
        model = W.StanPINN(W.HIDDEN_LAYERS, W.HIDDEN_NEURONS, dtype=W.DTYPE_LBFGS)
    else:
        raise ValueError(method)
    return model.to(device=CPU, dtype=W.DTYPE_LBFGS)

def load_model(method, seed, checkpoint):
    path = checkpoint_path(method, seed, checkpoint)
    ck = torch.load(path, map_location=CPU, weights_only=False)
    model = build_model(method)
    state = {
        k: (v.to(device=CPU, dtype=W.DTYPE_LBFGS) if torch.is_tensor(v) else v)
        for k, v in ck["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ck, path

@torch.no_grad()
def predict_field(model, X, T):
    xf = torch.as_tensor(X.reshape(-1,1), dtype=W.DTYPE_LBFGS, device=CPU)
    tf = torch.as_tensor(T.reshape(-1,1), dtype=W.DTYPE_LBFGS, device=CPU)
    parts = []
    bs = 20000
    for start in range(0, len(xf), bs):
        end = min(start + bs, len(xf))
        parts.append(model(xf[start:end], tf[start:end]).detach().cpu().numpy())
    return np.vstack(parts).reshape(X.shape)

def trapz(y, x, axis=0):
    fn = getattr(np, "trapezoid", np.trapz)
    return fn(y, x=x, axis=axis)

def mode_metrics(x, t, u):
    low_basis = np.sin(2.0 * np.pi * x)[:, None]
    high_basis = np.sin(10.0 * np.pi * x)[:, None]
    low_pred = 2.0 * trapz(u * low_basis, x=x, axis=0)
    high_pred = 2.0 * trapz(u * high_basis, x=x, axis=0)
    low_exact = np.sin(2.0 * np.pi * t)
    high_exact = np.sin(10.0 * np.pi * t)
    low_rel = np.linalg.norm(low_pred - low_exact) / np.linalg.norm(low_exact)
    high_rel = np.linalg.norm(high_pred - high_exact) / np.linalg.norm(high_exact)
    return float(low_rel), float(high_rel)

def main():
    configure_style()

    x = np.linspace(0.0, 1.0, N_X)
    t = np.linspace(0.0, 1.0, N_T)
    X, T = np.meshgrid(x, t, indexing="ij")

    rows = []
    for checkpoint in CHECKPOINTS:
        for method in METHODS:
            for seed in SEEDS:
                model, ck, path = load_model(method, seed, checkpoint)
                pred = predict_field(model, X, T)
                low_rel, high_rel = mode_metrics(x, t, pred)
                rows.append({
                    "checkpoint": checkpoint,
                    "checkpoint_label": CHECKPOINT_LABELS.get(checkpoint, checkpoint),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "low_mode_relative_l2": low_rel,
                    "high_mode_relative_l2": high_rel,
                    "spectral_bias_index_high_over_low": (
                        high_rel / low_rel if low_rel > 0 else np.nan
                    ),
                    "checkpoint_file": str(path),
                })
                del model
                gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "frequency_resolved_seedwise.csv", index=False)


    summary_rows = []
    for (checkpoint, method), g in df.groupby(["checkpoint", "method"], sort=False):
        row = {
            "checkpoint": checkpoint,
            "checkpoint_label": CHECKPOINT_LABELS.get(checkpoint, checkpoint),
            "method": method,
            "method_label": METHOD_LABELS[method],
        }
        for metric in [
            "low_mode_relative_l2",
            "high_mode_relative_l2",
            "spectral_bias_index_high_over_low",
        ]:
            mu, sd, n = mean_std(g[metric])
            row[f"{metric}_mean"] = mu
            row[f"{metric}_sample_std"] = sd
            row[f"{metric}_n"] = n
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "frequency_resolved_mean_std.csv", index=False)


    final = summary[summary["checkpoint"] == "total0700000"].copy()
    final["order"] = final["method"].map({m: i for i, m in enumerate(METHODS)})
    final = final.sort_values("order").reset_index(drop=True)

    seedwise_low = {}
    seedwise_high = {}
    final_df = df[df["checkpoint"] == "total0700000"]
    for i, method in enumerate(final["method"]):
        g = final_df[final_df["method"] == method]
        for _, r in g.iterrows():
            seedwise_low[(i, int(r["seed"]))] = float(r["low_mode_relative_l2"])
            seedwise_high[(i, int(r["seed"]))] = float(r["high_mode_relative_l2"])

    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    grouped_two_bar_with_seed_points(
        ax=ax,
        categories=final["method_label"].tolist(),
        mean_low=final["low_mode_relative_l2_mean"].to_numpy(float),
        std_low=final["low_mode_relative_l2_sample_std"].to_numpy(float),
        mean_high=final["high_mode_relative_l2_mean"].to_numpy(float),
        std_high=final["high_mode_relative_l2_sample_std"].to_numpy(float),
        seedwise_low=seedwise_low,
        seedwise_high=seedwise_high,
        ylabel=r"Modal coefficient relative $L_2$ error",
        low_label=r"Low frequency: $\sin(2\pi x)$",
        high_label=r"High frequency: $\sin(10\pi x)$",
        log_y=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / "wave_low_high_mode_error_final_paper_style.png",
                dpi=IMAGE_DPI, bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    xloc = np.arange(len(METHODS), dtype=float)
    width = 0.34

    adam = summary[summary["checkpoint"] == "adam0500000"].copy()
    final = summary[summary["checkpoint"] == "total0700000"].copy()
    for dat in (adam, final):
        dat["order"] = dat["method"].map({m: i for i, m in enumerate(METHODS)})
        dat.sort_values("order", inplace=True)

    bars1 = ax.bar(
        xloc - width / 2,
        adam["spectral_bias_index_high_over_low_mean"].to_numpy(float),
        width,
        yerr=adam["spectral_bias_index_high_over_low_sample_std"].to_numpy(float),
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR_LOW,
        edgecolor="black",
        linewidth=0.7,
        label=CHECKPOINT_LABELS["adam0500000"],
        zorder=2,
    )
    bars2 = ax.bar(
        xloc + width / 2,
        final["spectral_bias_index_high_over_low_mean"].to_numpy(float),
        width,
        yerr=final["spectral_bias_index_high_over_low_sample_std"].to_numpy(float),
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR_HIGH,
        edgecolor="black",
        linewidth=0.7,
        label=CHECKPOINT_LABELS["total0700000"],
        zorder=2,
    )

    for seed in SEED_ORDER:
        xs, ys = [], []
        for i, method in enumerate(METHODS):
            g = df[(df["checkpoint"] == "adam0500000") & (df["method"] == method) & (df["seed"] == seed)]
            if not g.empty:
                xs.append((xloc - width / 2)[i] + SEED_OFFSETS[seed] * 0.40)
                ys.append(float(g.iloc[0]["spectral_bias_index_high_over_low"]))
        if xs:
            ax.scatter(xs, ys, marker=SEED_MARKERS[seed], s=40,
                       facecolors=SEED_COLORS[seed], edgecolors="black",
                       linewidths=0.6, label=f"seed {seed}", zorder=4)
    for seed in SEED_ORDER:
        xs, ys = [], []
        for i, method in enumerate(METHODS):
            g = df[(df["checkpoint"] == "total0700000") & (df["method"] == method) & (df["seed"] == seed)]
            if not g.empty:
                xs.append((xloc + width / 2)[i] + SEED_OFFSETS[seed] * 0.40)
                ys.append(float(g.iloc[0]["spectral_bias_index_high_over_low"]))
        if xs:
            ax.scatter(xs, ys, marker=SEED_MARKERS[seed], s=40,
                       facecolors=SEED_COLORS[seed], edgecolors="black",
                       linewidths=0.6, label=f"seed {seed}", zorder=4)

    ax.set_xticks(xloc)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_ylabel("High-/low-frequency error ratio")
    ax.grid(True, axis="y", alpha=0.35)
    _annotate_bar_tops(ax, bars1, adam["spectral_bias_index_high_over_low_mean"].to_numpy(float), fmt="{:.2e}", log_y=False)
    _annotate_bar_tops(ax, bars2, final["spectral_bias_index_high_over_low_mean"].to_numpy(float), fmt="{:.2e}", log_y=False)
    handles, labels = _ordered_unique_legend(ax)
    preferred = [CHECKPOINT_LABELS["adam0500000"], CHECKPOINT_LABELS["total0700000"]] + [f"seed {s}" for s in SEED_ORDER]
    ordered_h = [handles[labels.index(lab)] for lab in preferred if lab in labels]
    ordered_l = [lab for lab in preferred if lab in labels]
    ax.legend(ordered_h, ordered_l, frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "wave_spectral_bias_index_checkpoints_paper_style.png",
                dpi=IMAGE_DPI, bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)

    print("Saved:", OUT.resolve())

if __name__ == "__main__":
    main()
