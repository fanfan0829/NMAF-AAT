from __future__ import annotations

from pathlib import Path
import os


FORCE_CPU = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import sys
import gc

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "nc_common.py"

if not COMMON_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find local nc_common.py:\n{COMMON_PATH}\n"
        "Put this analysis script in the same Ra folder as nc_common.py."
    )

spec = importlib.util.spec_from_file_location(
    "nc_common_local_ra6_repredict",
    COMMON_PATH,
)
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)


ANALYSIS_DEVICE = torch.device("cpu")
F.DEVICE = ANALYSIS_DEVICE

TARGET_RA_EXPONENT = 6
if int(F.RA_EXPONENT) != TARGET_RA_EXPONENT:
    raise RuntimeError(
        f"This is the Ra1e6 analysis script, but the local "
        f"nc_common.py has RA_EXPONENT={F.RA_EXPONENT}.\n"
        f"Common file loaded: {COMMON_PATH.resolve()}"
    )


EXPLICIT_BENCHMARK = {
    "psi_mid_abs": 16.3864,
    "psi_max_abs": 16.8111,
    "u_max_vertical_midplane": 64.8344,
    "w_max_horizontal_midplane": 220.559,
    "Nu_max_hot_wall": 17.536,
    "Nu_min_hot_wall": 0.9795,
    "Nu_mean_hot_wall": 8.825,
}

METHODS = ["tanh", "rar", "stan", "nmaf_aat"]
OUT = F.RESULT_ROOT / "four_method_analysis_repredict"
OUT.mkdir(parents=True, exist_ok=True)


RECOMPUTE_TOTAL_LOSS = True

BAR_COLOR = "#C0B3D0"
IMAGE_DPI = 800
SHOW_FIGURES = True


LOSS_FIG_HEIGHT = 5.0
LOSS_FIG_WIDTH = 7.0
LOSS_FONT_SIZE = 12
LOSS_AXIS_LABEL_SIZE = 14
LOSS_TITLE_SIZE = 13
LOSS_LEGEND_SIZE = 11
LOSS_TICK_LABEL_SIZE = 12
LOSS_LINEWIDTH = 1.9
LOSS_YLIM = (1.0e-1, 2.0)

LOSS_LINESTYLES = {
    "tanh": "-",
    "rar": "-",
    "stan": "-",
    "nmaf_aat": "-",
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


def configure_burgers_loss_style():

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": LOSS_FONT_SIZE,
        "axes.labelsize": LOSS_AXIS_LABEL_SIZE,
        "axes.titlesize": LOSS_TITLE_SIZE,
        "legend.fontsize": LOSS_LEGEND_SIZE,
        "xtick.labelsize": LOSS_TICK_LABEL_SIZE,
        "ytick.labelsize": LOSS_TICK_LABEL_SIZE,
        "lines.linewidth": 1.7,
    })


def method_label(method: str) -> str:
    return F.METHOD_LABELS.get(method, method)


def quantity_label(name: str) -> str:
    return F.QUANTITY_LABELS.get(name, name)


def finite_or_nan(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def relerr_percent(value, benchmark):
    value = finite_or_nan(value)
    benchmark = finite_or_nan(benchmark)
    if not np.isfinite(value) or not np.isfinite(benchmark) or benchmark == 0.0:
        return np.nan
    return abs(value - benchmark) / abs(benchmark) * 100.0


def detached_losses(d):
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu())
        else:
            out[k] = finite_or_nan(v)
    return out


def sync():

    return None


def checkpoint_path(method, seed):
    return (
        F.fixed_out_dir(method, seed)
        / f"Ra1e{F.RA_EXPONENT}_{method}_seed{int(seed)}_final.pth"
    )


def load_checkpoint(path):
    try:
        return torch.load(path, map_location=ANALYSIS_DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=ANALYSIS_DEVICE)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        tensor_like = [torch.is_tensor(v) for v in checkpoint.values()]
        if tensor_like and all(tensor_like):
            return checkpoint
    raise TypeError("Cannot find model_state_dict in checkpoint.")


def build_eval_model_from_checkpoint(method, seed):
    path = checkpoint_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(path)

    model = F.build_model(method, F.DTYPE_EVAL)
    checkpoint = load_checkpoint(path)
    state = extract_state_dict(checkpoint)

    if hasattr(F, "cast_state_dict"):
        state = F.cast_state_dict(state, F.DTYPE_EVAL)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model, path, checkpoint


def recompute_total_loss(model, seed):
    base = F.load_points_np(seed)
    monitor = F.make_tensors(base, base["interior"], F.DTYPE_EVAL)

    with torch.enable_grad():
        d = F.loss_components(model, monitor)
        losses = detached_losses(d)

    del monitor, d
    gc.collect()
    return losses


def evaluate_one(method, seed):
    model, ckpt_path, checkpoint = build_eval_model_from_checkpoint(method, seed)


    checkpoint_summary = (
        checkpoint.get("summary", {})
        if isinstance(checkpoint, dict)
        else {}
    )

    sync()
    with torch.enable_grad():
        physical = F.physical_quantities(model)
    sync()

    losses = {}
    if RECOMPUTE_TOTAL_LOSS:
        losses = recompute_total_loss(model, seed)

    row = {
        "Ra_exponent": int(F.RA_EXPONENT),
        "Ra": float(F.RA),
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "checkpoint_path": str(ckpt_path),


        "formal_adam_time_seconds": finite_or_nan(
            checkpoint_summary.get("formal_adam_time_seconds", np.nan)
        ),
        "formal_lbfgs_time_seconds": finite_or_nan(
            checkpoint_summary.get("formal_lbfgs_time_seconds", np.nan)
        ),
        "formal_training_time_seconds": finite_or_nan(
            checkpoint_summary.get("formal_training_time_seconds", np.nan)
        ),


        "rar_selection_seconds_total": finite_or_nan(
            checkpoint_summary.get("rar_selection_seconds_total", np.nan)
        ),
        "lbfgs_closure_evaluations": finite_or_nan(
            checkpoint_summary.get("lbfgs_closure_evaluations", np.nan)
        ),
    }

    if losses:
        row["total_loss"] = finite_or_nan(losses.get("total_loss", np.nan))
        row["pde_loss"] = finite_or_nan(losses.get("pde_loss", np.nan))
        row["bc_loss"] = finite_or_nan(losses.get("bc_loss", np.nan))
        row["f1_loss"] = finite_or_nan(losses.get("f1_loss", np.nan))
        row["f2_loss"] = finite_or_nan(losses.get("f2_loss", np.nan))

    scalar_errors = []
    for key, benchmark in EXPLICIT_BENCHMARK.items():
        pred = finite_or_nan(physical.get(key, np.nan))
        err = relerr_percent(pred, benchmark)
        row[key] = pred
        row[key + "_benchmark"] = benchmark
        row[key + "_relative_error_percent"] = err
        if np.isfinite(err):
            scalar_errors.append(float(err))


    all_reference_mre = float(np.mean(scalar_errors)) if scalar_errors else np.nan
    row["all_reference_physical_quantities_mean_relative_error_percent"] = all_reference_mre


    row["mean_scalar_physical_relative_error_percent"] = all_reference_mre

    row["number_of_reference_physical_quantities_used"] = int(len(scalar_errors))

    del model
    gc.collect()

    return row


def write_benchmark_table():
    rows = []
    for key, value in EXPLICIT_BENCHMARK.items():
        rows.append({
            "Ra_exponent": int(F.RA_EXPONENT),
            "Ra": float(F.RA),
            "quantity": key,
            "quantity_label": quantity_label(key),
            "benchmark_value_used": value,
            "included_in_relative_error": value is not None and np.isfinite(finite_or_nan(value)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "explicit_benchmark_used.csv", index=False)
    return df


def build_individual_table():
    rows = []
    skipped = []

    print("\nDirect model-checkpoint re-prediction analysis")
    print("=" * 112)
    print(f"Analysis script      : {Path(__file__).resolve()}")
    print(f"Local common module  : {COMMON_PATH.resolve()}")
    print(f"Result root          : {Path(F.RESULT_ROOT).resolve()}")
    print(f"Output dir           : {OUT.resolve()}")
    print(f"RA_EXPONENT          : {F.RA_EXPONENT}")
    print(f"Analysis device      : {ANALYSIS_DEVICE}")
    print(f"Methods requested    : {METHODS}")
    print(f"Seeds requested      : {list(F.SEEDS)}")
    print("=" * 112)

    for method in METHODS:
        for seed in F.SEEDS:
            ckpt = checkpoint_path(method, seed)
            try:
                row = evaluate_one(method, seed)
                rows.append(row)

                loss_msg = ""
                if "total_loss" in row:
                    loss_msg = f" | recomputed loss={row['total_loss']:.6e}"
                print(
                    f"USING   {method_label(method):18s} | seed={int(seed):5d}"
                    f"{loss_msg} | {ckpt}"
                )

            except Exception as exc:
                skipped.append({
                    "method": method,
                    "method_label": method_label(method),
                    "seed": int(seed),
                    "checkpoint_path": str(ckpt),
                    "reason": repr(exc),
                })
                print(
                    f"SKIPPED {method_label(method):18s} | seed={int(seed):5d} | {exc}"
                )

    pd.DataFrame(skipped).to_csv(OUT / "skipped_model_checkpoint_results.csv", index=False)

    if not rows:
        raise RuntimeError("No usable final model checkpoints were found.")

    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    df["_method_order"] = df["method"].map(order)
    df = (
        df.sort_values(["_method_order", "seed"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )
    df.to_csv(OUT / "individual_repredicted_results.csv", index=False)
    return df, pd.DataFrame(skipped)


def build_stats_table(df, filename="mean_std_repredicted_results.csv"):
    exclude = {"method", "method_label", "seed", "checkpoint_path"}
    numeric = [c for c in df.columns if c not in exclude]

    rows = []
    for method in METHODS:
        d = df[df.method == method]
        if d.empty:
            continue

        r = {
            "method": method,
            "method_label": method_label(method),
            "number_of_seeds": int(d["seed"].nunique()),
            "available_seeds": ",".join(str(int(s)) for s in sorted(d["seed"].unique())),
        }

        for c in numeric:
            vals = pd.to_numeric(d[c], errors="coerce").dropna()
            r[c + "_mean"] = float(vals.mean()) if len(vals) else np.nan
            r[c + "_sample_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            r[c + "_valid_n"] = int(len(vals))

        rows.append(r)

    st = pd.DataFrame(rows)
    st.to_csv(OUT / filename, index=False)
    return st


def common_completed_seeds(df):
    sets = []
    for method in METHODS:
        d = df[df["method"] == method]
        if d.empty:
            continue
        sets.append(set(int(s) for s in d["seed"].unique()))
    return sorted(set.intersection(*sets)) if sets else []


def build_paper_ready_table(stats):
    rows = []

    for _, r in stats.iterrows():
        def fmt(metric):
            m = r.get(metric + "_mean", np.nan)
            s = r.get(metric + "_sample_std", np.nan)
            if pd.isna(m):
                return "NA"
            return f"{m:.6e} ± {s:.6e}" if pd.notna(s) else f"{m:.6e}"

        rows.append({
            "Method": r["method_label"],
            "Available seeds": r["available_seeds"],
            "Number of seeds": int(r["number_of_seeds"]),
            "Recomputed total loss": fmt("total_loss"),
            "Mean scalar physical relative error (%)": fmt(
                "mean_scalar_physical_relative_error_percent"
            ),
            "Formal Adam time (s)": fmt("formal_adam_time_seconds"),
            "Formal L-BFGS time (s)": fmt("formal_lbfgs_time_seconds"),
            "Formal training time (s)": fmt("formal_training_time_seconds"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paper_ready_repredicted_results.csv", index=False)
    return out


def build_all_reference_physical_error_summary(df):


    metric = "all_reference_physical_quantities_mean_relative_error_percent"
    rows = []

    for method in METHODS:
        d = df[df["method"] == method].copy()
        if d.empty or metric not in d.columns:
            continue

        vals = pd.to_numeric(d[metric], errors="coerce").dropna()

        n_quantities = pd.to_numeric(
            d["number_of_reference_physical_quantities_used"],
            errors="coerce",
        ).dropna()

        rows.append({
            "method": method,
            "method_label": method_label(method),
            "available_seeds": ",".join(
                str(int(s)) for s in sorted(d["seed"].unique())
            ),
            "number_of_seeds": int(len(vals)),
            "number_of_reference_physical_quantities_used_per_seed": (
                int(n_quantities.iloc[0]) if len(n_quantities) else 0
            ),
            "all_reference_physical_quantities_mean_relative_error_percent_mean": (
                float(vals.mean()) if len(vals) else np.nan
            ),
            "all_reference_physical_quantities_mean_relative_error_percent_sample_std": (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            ),
            "mean_plus_minus_sample_std": (
                (
                    f"{float(vals.mean()):.6f} ± {float(vals.std(ddof=1)):.6f}"
                    if len(vals) > 1
                    else f"{float(vals.mean()):.6f}"
                )
                if len(vals)
                else "NA"
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT / "all_reference_physical_quantities_mean_relative_error_summary.csv",
        index=False,
    )


    seed_cols = [
        "method",
        "method_label",
        "seed",
        "number_of_reference_physical_quantities_used",
        metric,
    ]
    available_seed_cols = [c for c in seed_cols if c in df.columns]
    df[available_seed_cols].to_csv(
        OUT / "all_reference_physical_quantities_mean_relative_error_seedwise.csv",
        index=False,
    )

    return out


def build_training_time_summary(df):


    timing_cols = [
        "formal_adam_time_seconds",
        "formal_lbfgs_time_seconds",
        "formal_training_time_seconds",
    ]

    seedwise_cols = [
        "method",
        "method_label",
        "seed",
        *timing_cols,
        "rar_selection_seconds_total",
        "lbfgs_closure_evaluations",
        "checkpoint_path",
    ]
    seedwise_cols = [c for c in seedwise_cols if c in df.columns]
    seedwise = df[seedwise_cols].copy()
    seedwise.to_csv(
        OUT / "training_time_seedwise_from_checkpoint.csv",
        index=False,
    )

    rows = []
    for method in METHODS:
        d = df[df["method"] == method].copy()
        if d.empty:
            continue

        r = {
            "method": method,
            "method_label": method_label(method),
            "available_seeds": ",".join(
                str(int(s)) for s in sorted(d["seed"].unique())
            ),
            "number_of_seeds": int(d["seed"].nunique()),
        }

        for col in timing_cols:
            vals = pd.to_numeric(d[col], errors="coerce").dropna()
            r[col + "_mean"] = float(vals.mean()) if len(vals) else np.nan
            r[col + "_sample_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            )
            r[col + "_valid_n"] = int(len(vals))
            if len(vals) > 1:
                r[col + "_mean_plus_minus_sample_std"] = (
                    f"{float(vals.mean()):.6f} ± {float(vals.std(ddof=1)):.6f}"
                )
            elif len(vals) == 1:
                r[col + "_mean_plus_minus_sample_std"] = (
                    f"{float(vals.mean()):.6f}"
                )
            else:
                r[col + "_mean_plus_minus_sample_std"] = "NA"

        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(
        OUT / "training_time_mean_std_from_checkpoint.csv",
        index=False,
    )
    return out


def _log_safe_yerr(means, stds):
    means = np.asarray(means, float)
    stds = np.nan_to_num(np.asarray(stds, float), nan=0.0, posinf=0.0, neginf=0.0)
    lower = np.minimum(stds, 0.95 * means)
    return np.vstack([lower, stds])


def bar_with_seed_points(df, metric, ylabel, filename, log_y=False, scientific=False):
    if df is None or df.empty or metric not in df.columns:
        print(f"SKIPPED bar figure for {metric}: no available data")
        return

    methods = [m for m in METHODS if m in set(df["method"])]
    methods = [
        m for m in methods
        if pd.to_numeric(df.loc[df.method == m, metric], errors="coerce").notna().any()
    ]

    if not methods:
        print(f"SKIPPED bar figure for {metric}: no method has valid values")
        return

    x = np.arange(len(methods), dtype=float)
    means = np.array([
        pd.to_numeric(df[df.method == m][metric], errors="coerce").dropna().mean()
        for m in methods
    ], dtype=float)
    stds = np.array([
        pd.to_numeric(df[df.method == m][metric], errors="coerce").dropna().std(ddof=1)
        for m in methods
    ], dtype=float)
    stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)

    if log_y:
        valid = np.isfinite(means) & (means > 0.0)
    else:
        valid = np.isfinite(means)

    if not np.all(valid):
        methods = [m for m, ok in zip(methods, valid) if ok]
        means = means[valid]
        stds = stds[valid]
        x = np.arange(len(methods), dtype=float)

    if len(methods) == 0:
        print(f"SKIPPED bar figure for {metric}: no finite bar values")
        return

    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    bars = ax.bar(
        x,
        means,
        yerr=_log_safe_yerr(means, stds) if log_y else stds,
        capsize=5,
        alpha=0.78,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
    )

    offsets = {9999: -0.12, 1234: 0.0, 2026: 0.12}
    markers = {9999: "o", 1234: "s", 2026: "^"}
    colors = {9999: "#1f77b4", 1234: "#ff7f0e", 2026: "#2ca02c"}
    fallback_offsets = np.linspace(-0.12, 0.12, max(1, len(F.SEEDS)))
    fallback_markers = ["o", "s", "^", "D", "P"]

    for j, seed in enumerate(F.SEEDS):
        offset = offsets.get(seed, float(fallback_offsets[j]))
        marker = markers.get(seed, fallback_markers[j % len(fallback_markers)])
        color = colors.get(seed, None)

        xs, ys = [], []
        for i, method in enumerate(methods):
            g = df[(df.method == method) & (df.seed == seed)]
            if g.empty:
                continue

            value = pd.to_numeric(g.iloc[0][metric], errors="coerce")
            if pd.isna(value):
                continue
            value = float(value)
            if log_y and value <= 0.0:
                continue

            xs.append(i + offset)
            ys.append(value)

        if xs:
            kw = dict(
                marker=marker,
                s=40,
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )
            if color is not None:
                kw["facecolors"] = color
            ax.scatter(xs, ys, **kw)

    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods])
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.grid(True, axis="y", which="both", alpha=0.35)
    ax.legend(frameon=True, ncol=3)

    for rect, value in zip(bars, means):
        if not np.isfinite(value):
            continue
        label = f"{value:.2e}" if scientific else f"{value:.2f}"
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() * 0.90, value * 0.90),
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
    fig.savefig(OUT / filename, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def make_bar_figures(df):
    if "total_loss" in df.columns:
        bar_with_seed_points(
            df,
            "total_loss",
            "Recomputed total loss",
            f"Ra1e{F.RA_EXPONENT}_repredicted_total_loss.png",
            log_y=True,
            scientific=True,
        )

    bar_with_seed_points(
        df,
        "all_reference_physical_quantities_mean_relative_error_percent",
        "Mean relative error of all physical quantities (%)",
        f"Ra1e{F.RA_EXPONENT}_all_physical_quantities_mean_relative_error.png",
    )

    if "formal_training_time_seconds" in df.columns:
        bar_with_seed_points(
            df,
            "formal_training_time_seconds",
            "Formal training time (s)",
            f"Ra1e{F.RA_EXPONENT}_formal_training_time.png",
            log_y=False,
            scientific=False,
        )

    for key, benchmark in EXPLICIT_BENCHMARK.items():
        if benchmark is None:
            continue
        metric = key + "_relative_error_percent"
        if metric not in df.columns:
            continue

        bar_with_seed_points(
            df,
            metric,
            f"Relative error of {quantity_label(key)} (%)",
            f"Ra1e{F.RA_EXPONENT}_repredicted_{key}_relative_error.png",
        )


def build_loss_curve_statistics():


    outputs = []
    skipped = []

    for method in METHODS:
        frames = []
        used_seeds = []

        for seed in F.SEEDS:
            path = F.fixed_out_dir(method, seed) / "training_history.csv"
            try:
                if not path.exists():
                    raise FileNotFoundError(path)

                d = pd.read_csv(path)
                required = {"combined_step", "total_loss"}
                missing = required - set(d.columns)
                if missing:
                    raise KeyError(f"{path} missing columns: {sorted(missing)}")

                d = d[["combined_step", "total_loss"]].copy()
                d["combined_step"] = pd.to_numeric(d["combined_step"], errors="coerce")
                d["total_loss"] = pd.to_numeric(d["total_loss"], errors="coerce")
                d = d.dropna(subset=["combined_step", "total_loss"])
                d = d[np.isfinite(d["combined_step"]) & np.isfinite(d["total_loss"])]
                d = d[d["total_loss"] > 0.0]
                d["combined_step"] = d["combined_step"].astype(int)
                d = d.drop_duplicates("combined_step", keep="last")

                if d.empty:
                    raise RuntimeError("no finite positive loss rows")

                frames.append(
                    d.rename(columns={"total_loss": f"loss_seed_{int(seed)}"})
                )
                used_seeds.append(int(seed))

            except Exception as exc:
                skipped.append({
                    "method": method,
                    "method_label": method_label(method),
                    "seed": int(seed),
                    "history_path": str(path),
                    "reason": repr(exc),
                })

        if not frames:
            continue

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="combined_step", how="inner")

        if merged.empty:
            skipped.append({
                "method": method,
                "method_label": method_label(method),
                "seed": "all_available",
                "history_path": "",
                "reason": "No common combined_step across available seed histories.",
            })
            continue

        loss_cols = [f"loss_seed_{s}" for s in used_seeds]
        values = merged[loss_cols].to_numpy(dtype=float)

        valid = np.all(np.isfinite(values), axis=1) & np.all(values > 0.0, axis=1)
        merged = merged.loc[valid].copy()
        values = values[valid]

        if values.size == 0:
            continue

        log_values = np.log10(values)
        log_mean = np.mean(log_values, axis=1)
        if values.shape[1] > 1:
            log_std = np.std(log_values, axis=1, ddof=1)
        else:
            log_std = np.zeros_like(log_mean)

        stats = pd.DataFrame({
            "method": method,
            "method_label": method_label(method),
            "combined_step": merged["combined_step"].to_numpy(dtype=int),
            "geometric_mean_total_loss": 10.0 ** log_mean,
            "log10_sample_std": log_std,
            "lower_band": 10.0 ** (log_mean - log_std),
            "upper_band": 10.0 ** (log_mean + log_std),
            "number_of_seeds": int(len(used_seeds)),
            "available_seeds": ",".join(str(s) for s in used_seeds),
        })

        for seed in used_seeds:
            stats[f"total_loss_seed_{seed}"] = merged[
                f"loss_seed_{seed}"
            ].to_numpy(dtype=float)

        outputs.append(stats)

    pd.DataFrame(skipped).to_csv(
        OUT / "skipped_loss_histories.csv",
        index=False,
    )

    if not outputs:
        print("No usable training histories; Burgers-style loss figure skipped.")
        return pd.DataFrame()

    out = pd.concat(outputs, ignore_index=True)
    out.to_csv(
        OUT / "four_method_loss_log_statistics.csv",
        index=False,
    )
    return out


def plot_loss_burgers_style(loss_stats):

    if loss_stats is None or loss_stats.empty:
        return None

    configure_burgers_loss_style()

    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))

    plotted = False
    for method in METHODS:
        frame = loss_stats.loc[loss_stats["method"] == method].copy()
        if frame.empty:
            continue

        line, = ax.plot(
            frame["combined_step"],
            frame["geometric_mean_total_loss"],
            linestyle=LOSS_LINESTYLES[method],
            linewidth=LOSS_LINEWIDTH,
            label=method_label(method),
            zorder=3,
        )

        ax.fill_between(
            frame["combined_step"],
            frame["lower_band"],
            frame["upper_band"],
            color=line.get_color(),
            alpha=0.14,
            linewidth=0.0,
            zorder=1,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_yscale("log")

    xmax = int(loss_stats["combined_step"].max())
    ax.set_xlim(1, xmax)

    if LOSS_YLIM is not None:
        ax.set_ylim(*LOSS_YLIM)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")

    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, numticks=100)
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
        linewidth=0.8,
        alpha=0.55,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.32,
    )
    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )

    ax.legend(
        frameon=True,
        handlelength=2.5,
        handletextpad=0.8,
    )
    fig.tight_layout()

    path = OUT / f"Ra1e{F.RA_EXPONENT}_four_method_total_loss_burgers_style.png"
    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

    return path


def main():
    configure_plot_style()

    benchmark_df = write_benchmark_table()
    df, skipped = build_individual_table()
    stats = build_stats_table(df)
    paper = build_paper_ready_table(stats)
    all_reference_error_summary = build_all_reference_physical_error_summary(df)
    training_time_summary = build_training_time_summary(df)

    common_seeds = common_completed_seeds(df)
    if common_seeds:
        common_df = df[df["seed"].isin(common_seeds)].copy()
        common_df.to_csv(OUT / "individual_repredicted_results_common_seeds_only.csv", index=False)
        build_stats_table(
            common_df,
            "mean_std_repredicted_results_common_seeds_only.csv",
        )
    else:
        pd.DataFrame().to_csv(
            OUT / "mean_std_repredicted_results_common_seeds_only.csv",
            index=False,
        )

    loss_stats = build_loss_curve_statistics()
    loss_figure = plot_loss_burgers_style(loss_stats)


    configure_plot_style()
    make_bar_figures(df)

    manifest = {
        "version": "MODEL_CHECKPOINT_REPREDICT_EXPLICIT_BENCHMARK_RA6_CPU_V5_TRAINING_TIME",
        "analysis_mode": (
            "load final model checkpoints, recompute predictions and total loss, "
            "calculate errors from explicit benchmark values"
        ),
        "analysis_script": str(Path(__file__).resolve()),
        "local_common_module": str(COMMON_PATH.resolve()),
        "result_root": str(Path(F.RESULT_ROOT).resolve()),
        "output_dir": str(OUT.resolve()),
        "Ra_exponent": int(F.RA_EXPONENT),
        "Ra": float(F.RA),
        "device": str(ANALYSIS_DEVICE),
        "force_cpu": bool(FORCE_CPU),
        "recompute_total_loss": bool(RECOMPUTE_TOTAL_LOSS),
        "training_time_source": "checkpoint[summary] only; no training_summary.json",
        "training_time_statistic": "arithmetic mean +/- sample std across available seeds",
        "loss_plot_style": "finalized Burgers style",
        "loss_curve_statistics": "geometric mean +/- 1 sample std in log10(loss) space",
        "loss_figure": str(loss_figure) if loss_figure is not None else None,
        "methods_requested": METHODS,
        "seeds_requested": list(F.SEEDS),
        "available_rows": int(len(df)),
        "skipped_rows": int(len(skipped)),
        "common_completed_seeds_across_available_methods": common_seeds,
        "explicit_benchmark": EXPLICIT_BENCHMARK,
        "outputs": [
            "explicit_benchmark_used.csv",
            "individual_repredicted_results.csv",
            "mean_std_repredicted_results.csv",
            "paper_ready_repredicted_results.csv",
            "all_reference_physical_quantities_mean_relative_error_seedwise.csv",
            "all_reference_physical_quantities_mean_relative_error_summary.csv",
            "training_time_seedwise_from_checkpoint.csv",
            "training_time_mean_std_from_checkpoint.csv",
            "individual_repredicted_results_common_seeds_only.csv",
            "mean_std_repredicted_results_common_seeds_only.csv",
            "skipped_model_checkpoint_results.csv",
            "four_method_loss_log_statistics.csv",
            "skipped_loss_histories.csv",
            "Ra1e*_four_method_total_loss_burgers_style.png",
            "Ra1e*_repredicted_*.png",
        ],
    }
    (OUT / "model_reprediction_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nExplicit benchmark used:")
    print(benchmark_df.to_string(index=False))

    print("\nMean ± std from model-checkpoint re-prediction:")
    print(stats.to_string(index=False))

    if common_seeds:
        print("\nCommon completed seeds across available methods:", common_seeds)
        print("Common-seed statistics saved to:")
        print(OUT / "mean_std_repredicted_results_common_seeds_only.csv")

    print("\nPaper-ready table:")
    print(paper.to_string(index=False))

    print("\nFinal mean relative error across ALL reference physical quantities:")
    print(all_reference_error_summary.to_string(index=False))

    print("\nOriginal formal training time from final checkpoints:")
    print(training_time_summary.to_string(index=False))

    if len(skipped):
        print("\nSome checkpoints were skipped. See:")
        print(OUT / "skipped_model_checkpoint_results.csv")

    print("\nOutputs:", OUT.resolve())


if __name__ == "__main__":
    main()
