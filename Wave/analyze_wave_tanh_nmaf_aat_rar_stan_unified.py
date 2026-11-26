from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import torch

from wave_common_fixed_points_adam_lbfgs import (
    ADAM_EPOCHS,
    CHECKPOINT_SPECS,
    DTYPE_LBFGS,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    IMAGE_DPI,
    NMAFAATPINN,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    SEEDS,
    StanPINN,
    TanhPINN,
)


TOTAL_LOSS_YLIM = None


LOSS_FIG_WIDTH = 7.0
LOSS_FIG_HEIGHT = 5.0


FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TITLE_SIZE = 13
LEGEND_SIZE = 11
TICK_LABEL_SIZE = 12


LEGEND_HANDLE_LENGTH = 2.5
LEGEND_HANDLE_TEXT_PAD = 0.8


ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"
SHOW_FIGURES = True


DEVICE = torch.device("cpu")


N_TEST = 200


N_X_TIMEWISE = 1000
N_T_TIMEWISE = 101


LOSS_PLOT_MAX_POINTS = 20000


EXPORT_FULL_LOSS_STATS_GZIP = True


RAR_METHOD_DIR_NAME = "rar_adam_integrated_base6400_plus1280"
RAR_SELECTION_ADAM_STEP = 100000
RAR_BASE_N_RES = 6400
RAR_ADDED_N_RES = 1280
RAR_FINAL_N_RES = 7680


METHOD_ORDER = ["tanh", "rar", "stan", "nmaf_aat"]

METHOD_LABELS = {
    "tanh": "Standard PINN",
    "rar": "RAR-PINN",
    "stan": "Stan-PINN",
    "nmaf_aat": "NMAF-AAT",
}


LINESTYLES = {
    "tanh": "-",
    "rar": "-",
    "stan": "-",
    "nmaf_aat": "-",
}

MARKERS = {
    "tanh": "o",
    "rar": "^",
    "stan": "D",
    "nmaf_aat": "s",
}

CHECKPOINT_NAMES = [spec["name"] for spec in CHECKPOINT_SPECS]
CHECKPOINT_LABELS = {
    "adam0500000": "Adam 500k",
    "total0700000": "Adam 500k + L-BFGS",
}


def configure_plot_style():

    plt.rcParams["path.simplify"] = False

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "lines.linewidth": 1.5,
    })


def save_json(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def finish_figure(fig, path: Path):

    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    print(f"Saved PNG figure: {path.resolve()}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)
    return path


def mean_sample_std(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, 0
    if len(s) == 1:
        return float(s.iloc[0]), np.nan, 1
    return float(s.mean()), float(s.std(ddof=1)), int(len(s))


def fmt_mean_std(mean_value, std_value):
    if not np.isfinite(mean_value):
        return "NA"
    if not np.isfinite(std_value):
        return f"{mean_value:.6e}"
    return f"{mean_value:.6e} ± {std_value:.6e}"


def _safe_positive(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} contains non-positive or non-finite loss values.")
    return arr


def downsample_curve(frame: pd.DataFrame, max_points=LOSS_PLOT_MAX_POINTS):

    if len(frame) <= max_points:
        return frame.copy()

    idx = np.linspace(0, len(frame) - 1, max_points, dtype=np.int64)
    keep = set(idx.tolist())

    x = frame["global_iteration"].to_numpy(np.int64)
    for important in (1, RAR_SELECTION_ADAM_STEP, ADAM_EPOCHS, x[-1]):
        pos = int(np.argmin(np.abs(x - important)))
        keep.add(pos)

    return frame.iloc[sorted(keep)].reset_index(drop=True)


def build_model(method: str):
    if method in ("tanh", "rar"):
        model = TanhPINN(HIDDEN_LAYERS, HIDDEN_NEURONS)
    elif method == "nmaf_aat":
        model = NMAFAATPINN(
            HIDDEN_LAYERS,
            HIDDEN_NEURONS,
            dtype=DTYPE_LBFGS,
        )
    elif method == "stan":
        model = StanPINN(
            HIDDEN_LAYERS,
            HIDDEN_NEURONS,
            dtype=DTYPE_LBFGS,
        )
    else:
        raise ValueError(method)

    return model.to(dtype=DTYPE_LBFGS, device=DEVICE)


def _convert_state_dict_to_double(state_dict):
    return {
        key: (value.to(dtype=DTYPE_LBFGS) if torch.is_tensor(value) else value)
        for key, value in state_dict.items()
    }


def fixed_method_dir(method: str):
    return RESULT_ROOT / method


def rar_method_dir():
    path = RESULT_ROOT / RAR_METHOD_DIR_NAME
    if path.is_dir():
        return path


    candidates = []
    if RESULT_ROOT.exists():
        for p in RESULT_ROOT.iterdir():
            if not p.is_dir() or "rar" not in p.name.lower():
                continue
            ok = True
            for seed in SEEDS:
                hp = p / f"seed_{seed}" / "training_history.csv"
                if not hp.exists():
                    ok = False
                    break
                try:
                    cols = pd.read_csv(hp, nrows=2).columns
                    if "optimization_total_loss" not in cols:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok:
                candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Could not identify the formal Wave RAR result directory. Expected:\n"
        f"  {RESULT_ROOT / RAR_METHOD_DIR_NAME}\n"
        f"Fallback candidates: {[str(p) for p in candidates]}"
    )


def checkpoint_path(method: str, seed: int, checkpoint_name: str):
    if method == "rar":
        seed_dir = rar_method_dir() / f"seed_{seed}"
        exact = (
            seed_dir
            / f"wave_{RAR_METHOD_DIR_NAME}_seed{seed}_{checkpoint_name}.pth"
        )
        if exact.exists():
            return exact
        matches = sorted(seed_dir.glob(f"wave_*_seed{seed}_{checkpoint_name}.pth"))
        rar_matches = [p for p in matches if "rar" in p.name.lower()]
        if len(rar_matches) == 1:
            return rar_matches[0]
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Could not uniquely locate RAR checkpoint in {seed_dir}: {matches}"
        )

    return (
        fixed_method_dir(method)
        / f"seed_{seed}"
        / f"wave_{method}_seed{seed}_{checkpoint_name}.pth"
    )


def load_checkpoint_model(method: str, seed: int, checkpoint_name: str):
    path = checkpoint_path(method, seed, checkpoint_name)
    if not path.exists():
        raise FileNotFoundError(path)

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ck:
        raise KeyError(f"Checkpoint has no model_state_dict: {path}")

    model = build_model(method)
    model.load_state_dict(
        _convert_state_dict_to_double(ck["model_state_dict"]),
        strict=True,
    )
    model.eval()
    return ck, model, path


def make_reference_grid(n_test=N_TEST):
    x = np.linspace(0.0, 1.0, n_test, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_test, dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")
    u_exact = (
        np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * T)
        + np.sin(10.0 * np.pi * X) * np.sin(10.0 * np.pi * T)
    )
    return x, t, X, T, u_exact


@torch.no_grad()
def predict_field(model, X, T):
    xf = torch.as_tensor(
        X.reshape(-1, 1), dtype=DTYPE_LBFGS, device=DEVICE
    )
    tf = torch.as_tensor(
        T.reshape(-1, 1), dtype=DTYPE_LBFGS, device=DEVICE
    )

    parts = []
    batch_size = 20000
    for start in range(0, xf.shape[0], batch_size):
        pred = model(xf[start:start + batch_size], tf[start:start + batch_size])
        parts.append(pred.detach().cpu().numpy())

    return np.vstack(parts).reshape(X.shape)


def relative_l2(u_exact, u_pred):
    ef = np.asarray(u_exact, dtype=np.float64).reshape(-1)
    pf = np.asarray(u_pred, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(ef - pf, 2) / np.linalg.norm(ef, 2))


def frequency_mode_errors(x, t, u_pred):


    mode_low = np.sin(2.0 * np.pi * x)[:, None]
    mode_high = np.sin(10.0 * np.pi * x)[:, None]


    trap = getattr(np, "trapezoid", np.trapz)
    low_pred = 2.0 * trap(u_pred * mode_low, x=x, axis=0)
    high_pred = 2.0 * trap(u_pred * mode_high, x=x, axis=0)

    low_exact = np.sin(2.0 * np.pi * t)
    high_exact = np.sin(10.0 * np.pi * t)

    low_rel = float(
        np.linalg.norm(low_pred - low_exact, 2)
        / np.linalg.norm(low_exact, 2)
    )
    high_rel = float(
        np.linalg.norm(high_pred - high_exact, 2)
        / np.linalg.norm(high_exact, 2)
    )

    projected = (
        mode_low * low_pred[None, :]
        + mode_high * high_pred[None, :]
    )
    denom = np.linalg.norm(np.asarray(u_pred).reshape(-1), 2)
    leakage_rel = (
        float(np.linalg.norm((u_pred - projected).reshape(-1), 2) / denom)
        if denom > 0.0
        else np.nan
    )

    return {
        "low_mode_relative_l2": low_rel,
        "high_mode_relative_l2": high_rel,
        "two_mode_leakage_relative_l2": leakage_rel,
        "low_mode_coeff_pred": low_pred,
        "high_mode_coeff_pred": high_pred,
        "low_mode_coeff_exact": low_exact,
        "high_mode_coeff_exact": high_exact,
    }


def checkpoint_recorded_losses(ck):
    latest = ck.get("latest_metrics", {})
    return {
        "total_loss": float(latest.get("total_loss", np.nan)),
        "residual_loss": float(latest.get("residual_loss", np.nan)),
        "data_loss": float(latest.get("data_loss", np.nan)),
        "bc_loss": float(latest.get("bc_loss", np.nan)),
        "ic1_loss": float(latest.get("ic1_loss", np.nan)),
        "ic2_loss": float(latest.get("ic2_loss", np.nan)),
        "optimization_total_loss": float(
            latest.get("optimization_total_loss", np.nan)
        ),
    }


def rebuild_all_checkpoint_metrics():
    x, t, X, T, u_exact = make_reference_grid()
    rows = []
    prediction_dir = ANALYSIS_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for method in METHOD_ORDER:
        for seed in SEEDS:
            for checkpoint_name in CHECKPOINT_NAMES:
                ck, model, path = load_checkpoint_model(method, seed, checkpoint_name)
                u_pred = predict_field(model, X, T)
                abs_error = np.abs(u_exact - u_pred)
                mode = frequency_mode_errors(x, t, u_pred)
                losses = checkpoint_recorded_losses(ck)

                pred_name = (
                    f"wave_{method}_seed{seed}_{checkpoint_name}_prediction.npz"
                )
                np.savez_compressed(
                    prediction_dir / pred_name,
                    x=x,
                    t=t,
                    X=X,
                    T=T,
                    u_exact=u_exact,
                    u_pred=u_pred,
                    abs_error=abs_error,
                    low_mode_coeff_pred=mode["low_mode_coeff_pred"],
                    high_mode_coeff_pred=mode["high_mode_coeff_pred"],
                    low_mode_coeff_exact=mode["low_mode_coeff_exact"],
                    high_mode_coeff_exact=mode["high_mode_coeff_exact"],
                )

                total_iteration = int(
                    ck.get(
                        "total_iteration",
                        next(
                            spec["total_iteration"]
                            for spec in CHECKPOINT_SPECS
                            if spec["name"] == checkpoint_name
                        ),
                    )
                )

                row = {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": int(seed),
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_label": CHECKPOINT_LABELS.get(
                        checkpoint_name, checkpoint_name
                    ),
                    "phase": ck.get("phase", "NA"),
                    "total_iteration": total_iteration,
                    "adam_iteration": int(ck.get("adam_iteration", 0)),
                    "lbfgs_eval": int(ck.get("lbfgs_eval", 0)),
                    **losses,
                    "relative_l2_error": relative_l2(u_exact, u_pred),
                    "maximum_absolute_error": float(abs_error.max()),
                    "low_mode_relative_l2": mode["low_mode_relative_l2"],
                    "high_mode_relative_l2": mode["high_mode_relative_l2"],
                    "two_mode_leakage_relative_l2": mode[
                        "two_mode_leakage_relative_l2"
                    ],
                    "elapsed_seconds": float(ck.get("elapsed_seconds", np.nan)),
                    "checkpoint_file": str(path),
                    "prediction_file": str(prediction_dir / pred_name),
                }
                rows.append(row)

                print(
                    f"{METHOD_LABELS[method]:15s} | seed {seed:4d} | "
                    f"{checkpoint_name:12s} | "
                    f"L2={row['relative_l2_error']:.6e} | "
                    f"Max={row['maximum_absolute_error']:.6e} | "
                    f"LowMode={row['low_mode_relative_l2']:.6e} | "
                    f"HighMode={row['high_mode_relative_l2']:.6e}"
                )

                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    individual = pd.DataFrame(rows)
    individual["method_order"] = individual["method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    )
    individual["checkpoint_order"] = individual["checkpoint_name"].map(
        {n: i for i, n in enumerate(CHECKPOINT_NAMES)}
    )
    individual = individual.sort_values(
        ["checkpoint_order", "method_order", "seed"]
    ).drop(columns=["method_order", "checkpoint_order"]).reset_index(drop=True)

    individual.to_csv(
        ANALYSIS_DIR / "four_method_individual_checkpoint_results.csv",
        index=False,
    )
    return individual


def summarize_checkpoint_metrics(individual):
    metrics = [
        "total_loss",
        "residual_loss",
        "data_loss",
        "relative_l2_error",
        "maximum_absolute_error",
        "low_mode_relative_l2",
        "high_mode_relative_l2",
        "two_mode_leakage_relative_l2",
        "elapsed_seconds",
    ]

    rows = []
    for (checkpoint_name, method), group in individual.groupby(
        ["checkpoint_name", "method"], sort=False
    ):
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "checkpoint_name": checkpoint_name,
            "checkpoint_label": CHECKPOINT_LABELS.get(
                checkpoint_name, checkpoint_name
            ),
            "number_of_seeds": int(group["seed"].nunique()),
            "actual_total_iteration_min": int(group["total_iteration"].min()),
            "actual_total_iteration_max": int(group["total_iteration"].max()),
        }
        for metric in metrics:
            mean_v, std_v, n_v = mean_sample_std(group[metric])
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_sample_std"] = std_v
            row[f"{metric}_valid_n"] = n_v
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["method_order"] = summary["method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    )
    summary["checkpoint_order"] = summary["checkpoint_name"].map(
        {n: i for i, n in enumerate(CHECKPOINT_NAMES)}
    )
    summary = summary.sort_values(
        ["checkpoint_order", "method_order"]
    ).drop(columns=["method_order", "checkpoint_order"]).reset_index(drop=True)

    summary.to_csv(
        ANALYSIS_DIR / "four_method_mean_std_statistics.csv",
        index=False,
    )

    paper_metrics = [
        "total_loss",
        "relative_l2_error",
        "maximum_absolute_error",
        "low_mode_relative_l2",
        "high_mode_relative_l2",
    ]
    paper_rows = []
    for _, r in summary.iterrows():
        out = {
            "Method": r["method_label"],
            "Checkpoint": r["checkpoint_label"],
            "Seeds": int(r["number_of_seeds"]),
        }
        for metric in paper_metrics:
            out[metric] = fmt_mean_std(
                r[f"{metric}_mean"],
                r[f"{metric}_sample_std"],
            )
        paper_rows.append(out)

    pd.DataFrame(paper_rows).to_csv(
        ANALYSIS_DIR / "four_method_paper_ready_mean_std_table.csv",
        index=False,
    )
    return summary


def method_history_path(method, seed):
    if method == "rar":
        return rar_method_dir() / f"seed_{seed}" / "training_history.csv"
    return fixed_method_dir(method) / f"seed_{seed}" / "training_history.csv"


def load_history_column(method, seed, column):
    path = method_history_path(method, seed)
    if not path.exists():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0).columns
    if "global_iteration" not in header or column not in header:
        raise ValueError(
            f"{path} does not contain required columns global_iteration/{column}. "
            f"Available: {list(header)}"
        )

    frame = pd.read_csv(
        path,
        usecols=["global_iteration", column],
        dtype={"global_iteration": np.int64, column: np.float64},
    )
    frame = frame.drop_duplicates("global_iteration", keep="last")
    frame = frame.loc[frame["global_iteration"] >= 1].copy()
    return frame


def compute_log_curve(method, source_column="total_loss"):
    merged = None
    for seed in SEEDS:
        frame = load_history_column(method, seed, source_column).rename(
            columns={source_column: f"loss_seed_{seed}"}
        )
        merged = (
            frame
            if merged is None
            else merged.merge(frame, on="global_iteration", how="inner")
        )

    seed_cols = [f"loss_seed_{seed}" for seed in SEEDS]
    values = _safe_positive(
        merged[seed_cols].to_numpy(np.float64),
        f"{method}/{source_column}",
    )
    logs = np.log10(values)
    mu = logs.mean(axis=1)
    sd = logs.std(axis=1, ddof=1)

    out = pd.DataFrame({
        "global_iteration": merged["global_iteration"].to_numpy(np.int64),
        "geometric_mean_total_loss": 10.0 ** mu,
        "log10_sample_std": sd,
        "lower_band": 10.0 ** (mu - sd),
        "upper_band": 10.0 ** (mu + sd),
    })
    for seed in SEEDS:
        out[f"loss_seed_{seed}"] = merged[f"loss_seed_{seed}"].to_numpy(
            np.float64
        )

    if EXPORT_FULL_LOSS_STATS_GZIP:
        suffix = (
            "common6400"
            if source_column == "total_loss"
            else source_column
        )
        out.to_csv(
            ANALYSIS_DIR / f"{method}_{suffix}_loss_log_statistics.csv.gz",
            index=False,
            compression="gzip",
        )

    return out


def plot_four_method_common_loss():
    fig, ax = plt.subplots(
        figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT)
    )

    compact_exports = []

    for method in METHOD_ORDER:
        full = compute_log_curve(method, "total_loss")
        compact = downsample_curve(full)
        compact.insert(0, "method", method)
        compact.insert(1, "method_label", METHOD_LABELS[method])
        compact_exports.append(compact)

        line, = ax.plot(
            compact["global_iteration"],
            compact["geometric_mean_total_loss"],
            linestyle=LINESTYLES[method],
            linewidth=1.5,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=METHOD_LABELS[method],
            zorder=3,
        )

        ax.fill_between(
            compact["global_iteration"],
            compact["lower_band"],
            compact["upper_band"],
            color=line.get_color(),
            alpha=0.14,
            linewidth=0.0,
            zorder=1,
        )

        del full, compact
        gc.collect()

    pd.concat(
        compact_exports,
        ignore_index=True,
    ).to_csv(
        ANALYSIS_DIR / "four_method_common_loss_plot_statistics.csv",
        index=False,
    )

    ax.set_yscale("log")
    ax.set_xlim(left=1)

    if TOTAL_LOSS_YLIM is not None:
        ax.set_ylim(*TOTAL_LOSS_YLIM)

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
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )

    fig.tight_layout()

    return finish_figure(
        fig,
        ANALYSIS_DIR / "wave_four_method_common_total_loss.png",
    )


def plot_rar_monitor_vs_optimization():
    common = compute_log_curve("rar", "total_loss")
    opt = compute_log_curve("rar", "optimization_total_loss")
    common = downsample_curve(common)
    opt = downsample_curve(opt)

    compact = common[[
        "global_iteration",
        "geometric_mean_total_loss",
        "lower_band",
        "upper_band",
    ]].rename(columns={
        "geometric_mean_total_loss": "monitor_geometric_mean",
        "lower_band": "monitor_lower_band",
        "upper_band": "monitor_upper_band",
    }).merge(
        opt[[
            "global_iteration",
            "geometric_mean_total_loss",
            "lower_band",
            "upper_band",
        ]].rename(columns={
            "geometric_mean_total_loss": "optimization_geometric_mean",
            "lower_band": "optimization_lower_band",
            "upper_band": "optimization_upper_band",
        }),
        on="global_iteration",
        how="inner",
    )

    compact.to_csv(
        ANALYSIS_DIR
        / "rar_common6400_vs_optimization7680_plot_statistics.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT)
    )

    line1, = ax.plot(
        compact["global_iteration"],
        compact["monitor_geometric_mean"],
        linestyle="-",
        linewidth=1.5,
        label="RAR: original 6400-point monitor",
        zorder=3,
    )

    ax.fill_between(
        compact["global_iteration"],
        compact["monitor_lower_band"],
        compact["monitor_upper_band"],
        color=line1.get_color(),
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )

    line2, = ax.plot(
        compact["global_iteration"],
        compact["optimization_geometric_mean"],
        linestyle="--",
        linewidth=1.5,
        label="RAR: actual 7680-point objective",
        zorder=3,
    )

    ax.fill_between(
        compact["global_iteration"],
        compact["optimization_lower_band"],
        compact["optimization_upper_band"],
        color=line2.get_color(),
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )

    ax.set_yscale("log")
    ax.set_xlim(left=1)

    if TOTAL_LOSS_YLIM is not None:
        ax.set_ylim(*TOTAL_LOSS_YLIM)

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
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )

    fig.tight_layout()

    return finish_figure(
        fig,
        ANALYSIS_DIR
        / "wave_rar_common6400_vs_optimization7680_loss.png",
    )


def plot_grouped_metric(summary, metric, ylabel, filename, log_scale=True):
    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))
    checkpoints = CHECKPOINT_NAMES
    n_methods = len(METHOD_ORDER)
    group_x = np.arange(len(checkpoints), dtype=np.float64)
    width = 0.18

    for i, method in enumerate(METHOD_ORDER):
        means = []
        stds = []
        for checkpoint_name in checkpoints:
            row = summary.loc[
                (summary["method"] == method)
                & (summary["checkpoint_name"] == checkpoint_name)
            ]
            if len(row) != 1:
                raise RuntimeError(
                    f"Missing summary row for {method}/{checkpoint_name}"
                )
            means.append(float(row.iloc[0][f"{metric}_mean"]))
            stds.append(float(row.iloc[0][f"{metric}_sample_std"]))

        offset = (i - (n_methods - 1) / 2.0) * width
        ax.bar(
            group_x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=METHOD_LABELS[method],
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(group_x)
    ax.set_xticklabels([CHECKPOINT_LABELS[c] for c in checkpoints])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", which="both", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = ANALYSIS_DIR / Path(filename).with_suffix(".png")
    finish_figure(fig, path)


def plot_frequency_mode_error(summary):
    final_name = "total0700000"
    data = summary.loc[summary["checkpoint_name"] == final_name].copy()
    data["method_order"] = data["method"].map(
        {m: i for i, m in enumerate(METHOD_ORDER)}
    )
    data = data.sort_values("method_order")

    x = np.arange(len(data), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))
    ax.bar(
        x - width / 2,
        data["low_mode_relative_l2_mean"],
        width,
        yerr=data["low_mode_relative_l2_sample_std"],
        capsize=3,
        label=r"Low mode: $\sin(2\pi x)$",
    )
    ax.bar(
        x + width / 2,
        data["high_mode_relative_l2_mean"],
        width,
        yerr=data["high_mode_relative_l2_sample_std"],
        capsize=3,
        label=r"High mode: $\sin(10\pi x)$",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(data["method_label"])
    ax.set_ylabel("Spatial-mode coefficient relative $L_2$ error")
    ax.grid(True, axis="y", which="both", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = ANALYSIS_DIR / "wave_four_method_frequency_resolved_error_final.png"
    finish_figure(fig, path)


def prediction_file(method, seed, checkpoint_name):
    return (
        ANALYSIS_DIR
        / "predictions"
        / f"wave_{method}_seed{seed}_{checkpoint_name}_prediction.npz"
    )


def plot_representative_profiles(checkpoint_name):
    times = [0.25, 0.50, 0.75]
    fig, axes = plt.subplots(1, len(times), figsize=(15, 4.4), sharey=True)

    first = np.load(
        prediction_file("tanh", REPRESENTATIVE_SEED, checkpoint_name)
    )
    x = first["x"]
    t = first["t"]
    exact = first["u_exact"]

    for ax, target_t in zip(axes, times):
        j = int(np.argmin(np.abs(t - target_t)))
        actual_t = float(t[j])
        ax.plot(x, exact[:, j], linewidth=2.0, label="Exact")
        for method in METHOD_ORDER:
            data = np.load(
                prediction_file(method, REPRESENTATIVE_SEED, checkpoint_name)
            )
            ax.plot(
                x,
                data["u_pred"][:, j],
                linestyle=LINESTYLES[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"t = {actual_t:.3f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("u(x,t)")
    axes[-1].legend(frameon=True, fontsize=9)
    fig.tight_layout()
    path = ANALYSIS_DIR / f"wave_four_method_profiles_{checkpoint_name}.png"
    finish_figure(fig, path)


def plot_representative_error_fields(checkpoint_name):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.3), sharex=True, sharey=True)
    axes = axes.ravel()
    vmax = 0.0
    loaded = {}
    for method in METHOD_ORDER:
        data = np.load(
            prediction_file(method, REPRESENTATIVE_SEED, checkpoint_name)
        )
        loaded[method] = data
        vmax = max(vmax, float(np.max(data["abs_error"])))

    for ax, method in zip(axes, METHOD_ORDER):
        data = loaded[method]
        im = ax.pcolormesh(
            data["t"],
            data["x"],
            data["abs_error"],
            shading="auto",
            cmap="Reds",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    path = ANALYSIS_DIR / f"wave_four_method_abs_error_fields_{checkpoint_name}.png"
    finish_figure(fig, path)


def timewise_l2_for_model(model):
    x = torch.linspace(
        0.0, 1.0, N_X_TIMEWISE, dtype=DTYPE_LBFGS, device=DEVICE
    ).reshape(-1, 1)
    t_values = np.linspace(0.0, 1.0, N_T_TIMEWISE)
    rows = []
    model.eval()

    with torch.no_grad():
        for tv in t_values:
            tt = torch.full_like(x, float(tv))
            pred = model(x, tt)
            exact = (
                torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * tt)
                + torch.sin(10.0 * np.pi * x) * torch.sin(10.0 * np.pi * tt)
            )
            denom = torch.linalg.vector_norm(exact.reshape(-1), ord=2)
            if float(denom.detach().cpu()) < 1.0e-14:
                rel = np.nan
            else:
                rel = float(
                    (
                        torch.linalg.vector_norm(
                            (exact - pred).reshape(-1), ord=2
                        )
                        / denom
                    ).detach().cpu()
                )
            rows.append({"t": float(tv), "relative_l2_error": rel})
    return pd.DataFrame(rows)


def plot_representative_timewise_final():
    checkpoint_name = "total0700000"
    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))
    exports = []

    for method in METHOD_ORDER:
        _, model, _ = load_checkpoint_model(
            method, REPRESENTATIVE_SEED, checkpoint_name
        )
        frame = timewise_l2_for_model(model)
        frame.insert(0, "method", method)
        frame.insert(1, "method_label", METHOD_LABELS[method])
        exports.append(frame)
        ax.plot(
            frame["t"],
            frame["relative_l2_error"],
            linestyle=LINESTYLES[method],
            label=METHOD_LABELS[method],
        )
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    pd.concat(exports, ignore_index=True).to_csv(
        ANALYSIS_DIR / "wave_four_method_timewise_l2_seed99_final.csv",
        index=False,
    )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Time t")
    ax.set_ylabel(r"Relative $L_2$ error over x")
    ax.grid(True, which="both", alpha=0.45)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = ANALYSIS_DIR / "wave_four_method_timewise_l2_seed99_final.png"
    finish_figure(fig, path)


def make_manifest(summary):
    manifest = {
        "methods": METHOD_ORDER,
        "method_labels": METHOD_LABELS,
        "seeds": list(SEEDS),
        "representative_seed": int(REPRESENTATIVE_SEED),
        "test_grid": [N_TEST, N_TEST],
        "training": {
            "adam_steps": int(ADAM_EPOCHS),
            "lbfgs_target_max_eval": 200000,
            "global_index_definition": "Adam step + L-BFGS closure/function-evaluation count",
        },
        "rar": {
            "method_dir": str(rar_method_dir()),
            "selection_adam_step": RAR_SELECTION_ADAM_STEP,
            "base_pde_points": RAR_BASE_N_RES,
            "added_pde_points": RAR_ADDED_N_RES,
            "final_pde_points": RAR_FINAL_N_RES,
            "primary_cross_method_loss": "RAR total_loss monitor on original shared 6400 PDE points",
            "diagnostic_optimization_loss": "RAR optimization_total_loss on current 6400/7680 training set",
        },
        "loss_statistics": {
            "center": "geometric mean across seeds",
            "band": "+/- 1 sample std in log10(loss) space",
        },
        "checkpoint_statistics": "arithmetic mean +/- sample std on original scale",
        "accuracy": {
            "relative_l2": "flattened full-field discrete vector 2-norm",
            "maximum_absolute_error": "max absolute pointwise error on 200x200 grid",
            "frequency_resolved": "relative L2 error of spatial-mode coefficients after projection onto sin(2*pi*x) and sin(10*pi*x)",
        },
        "paper_ready_rows": int(len(summary)),
    }
    save_json(ANALYSIS_DIR / "analysis_manifest.json", manifest)


def print_summary(summary):
    print("\n" + "=" * 140)
    print("Wave four-method checkpoint statistics")
    print("=" * 140)
    cols = [
        "checkpoint_label",
        "method_label",
        "total_loss_mean",
        "total_loss_sample_std",
        "relative_l2_error_mean",
        "relative_l2_error_sample_std",
        "maximum_absolute_error_mean",
        "maximum_absolute_error_sample_std",
        "low_mode_relative_l2_mean",
        "high_mode_relative_l2_mean",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print("=" * 140)


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    print("=" * 100)
    print("Wave: Standard PINN / RAR-PINN / Stan-PINN / NMAF-AAT")
    print("Unified fair analysis")
    print("=" * 100)
    print(f"Result root: {RESULT_ROOT.resolve()}")
    print(f"RAR directory: {rar_method_dir().resolve()}")
    print(
        "Primary loss comparison: original shared 6400 PDE points + identical BC/IC."
    )
    print(
        "RAR 7680-point optimization objective is diagnostic only, not the main four-method loss."
    )

    individual = rebuild_all_checkpoint_metrics()
    summary = summarize_checkpoint_metrics(individual)
    print_summary(summary)

    plot_four_method_common_loss()
    plot_rar_monitor_vs_optimization()

    plot_grouped_metric(
        summary,
        "relative_l2_error",
        r"Relative $L_2$ error",
        "wave_four_method_relative_l2.png",
        log_scale=True,
    )
    plot_grouped_metric(
        summary,
        "maximum_absolute_error",
        "Maximum absolute error",
        "wave_four_method_max_abs_error.png",
        log_scale=True,
    )
    plot_frequency_mode_error(summary)

    for checkpoint_name in CHECKPOINT_NAMES:
        plot_representative_profiles(checkpoint_name)
        plot_representative_error_fields(checkpoint_name)

    plot_representative_timewise_final()
    make_manifest(summary)

    print("\n" + "=" * 100)
    print("Wave four-method unified analysis completed.")
    print(f"Output directory: {ANALYSIS_DIR.resolve()}")
    print("Main loss figure: wave_four_method_common_total_loss.png")
    print("=" * 100)


if __name__ == "__main__":
    main()
