from __future__ import annotations

from pathlib import Path
import os

FORCE_CPU = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "nc_common.py"

if not COMMON_PATH.exists():
    raise FileNotFoundError(
        "Cannot find nc_common.py in the same Ra1e6 folder."
    )

spec = importlib.util.spec_from_file_location(
    "nc_common_local_ra6_analysis_40000",
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
        f"This analysis is for Ra=1e6, but local common file has "
        f"RA_EXPONENT={F.RA_EXPONENT}."
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
TARGET_COMBINED_STEP = 40000
ORIGINAL_NOMINAL_TOTAL_STEP = 30000

FORCE_ORIGINAL_AND_PAD = {
    ("tanh", 9999),
    ("tanh", 1234),
    ("rar", 9999),
    ("rar", 1234),
}

OUT = F.RESULT_ROOT / "four_method_analysis_to_40000"
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


def method_label(method):
    return F.METHOD_LABELS.get(method, method)


def quantity_label(name):
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
        out[k] = float(v.detach().cpu()) if torch.is_tensor(v) else finite_or_nan(v)
    return out


def original_checkpoint_path(method, seed):
    return (
        F.fixed_out_dir(method, seed)
        / f"Ra1e{F.RA_EXPONENT}_{method}_seed{int(seed)}_final.pth"
    )


def continuation_dir(method, seed):
    return F.fixed_out_dir(method, seed) / "continued_lbfgs_extra10000"


def continuation_checkpoint_candidates(method, seed):
    d = continuation_dir(method, seed)
    if not d.exists():
        return []

    exact = (
        d
        / f"Ra1e{F.RA_EXPONENT}_{method}_seed{int(seed)}"
          "_continued_lbfgs_extra10000_final.pth"
    )

    paths = []
    if exact.exists():
        paths.append(exact)

    for p in sorted(d.glob("*continued*final.pth")):
        if p not in paths:
            paths.append(p)

    return paths


def load_checkpoint(path):
    try:
        return torch.load(path, map_location=ANALYSIS_DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=ANALYSIS_DEVICE)


def checkpoint_summary(path):
    ck = load_checkpoint(path)
    if isinstance(ck, dict):
        obj = ck.get("summary", {})
        return obj if isinstance(obj, dict) else {}
    return {}


def continuation_reported_final_step(method, seed, ckpt_path=None):
    candidates = []

    if ckpt_path is not None and Path(ckpt_path).exists():
        s = checkpoint_summary(Path(ckpt_path))

        for key in ("final_nominal_combined_step", "final_combined_step"):
            v = finite_or_nan(s.get(key, np.nan))
            if np.isfinite(v):
                candidates.append(int(round(v)))

        original_step = finite_or_nan(
            s.get("original_nominal_total_step", np.nan)
        )
        if not np.isfinite(original_step):
            original_step = finite_or_nan(
                s.get("original_combined_step", np.nan)
            )

        extra_evals = finite_or_nan(
            s.get("extra_lbfgs_actual_closure_evaluations", np.nan)
        )

        if np.isfinite(original_step) and np.isfinite(extra_evals):
            candidates.append(int(round(original_step + extra_evals)))

    combined_path = (
        continuation_dir(method, seed)
        / "training_history_original_plus_extra10000.csv"
    )

    if combined_path.exists():
        try:
            d = pd.read_csv(combined_path)
            if "combined_step" in d.columns:
                vals = pd.to_numeric(d["combined_step"], errors="coerce").dropna()
                if len(vals):
                    candidates.append(int(vals.max()))
        except Exception:
            pass

    return max(candidates) if candidates else None


def continuation_reached_40000(method, seed):
    if (method, int(seed)) in FORCE_ORIGINAL_AND_PAD:
        return False, None, None

    candidates = continuation_checkpoint_candidates(method, seed)
    if not candidates:
        return False, None, None

    for path in candidates:
        step = continuation_reported_final_step(method, seed, ckpt_path=path)
        if step is not None and step >= TARGET_COMBINED_STEP:
            return True, path, step

    return False, None, None


def selected_prediction_checkpoint(method, seed):
    reached, cont_path, final_step = continuation_reached_40000(method, seed)

    if reached:
        return {
            "path": cont_path,
            "model_source": "continued_40000_model",
            "prediction_step": TARGET_COMBINED_STEP,
            "continuation_reached_40000": True,
            "reported_continuation_final_step": final_step,
        }

    old = original_checkpoint_path(method, seed)
    if not old.exists():
        raise FileNotFoundError(old)

    return {
        "path": old,
        "model_source": "original_final_model_used_for_plateau_case",
        "prediction_step": ORIGINAL_NOMINAL_TOTAL_STEP,
        "continuation_reached_40000": False,
        "reported_continuation_final_step": final_step,
    }


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        flags = [torch.is_tensor(v) for v in checkpoint.values()]
        if flags and all(flags):
            return checkpoint
    raise TypeError("Cannot find model_state_dict in checkpoint.")


def build_eval_model(method, seed):
    sel = selected_prediction_checkpoint(method, seed)
    path = sel["path"]

    ck = load_checkpoint(path)
    state = extract_state_dict(ck)

    if hasattr(F, "cast_state_dict"):
        state = F.cast_state_dict(state, F.DTYPE_EVAL)
    else:
        state = {
            k: (
                v.to(dtype=F.DTYPE_EVAL)
                if torch.is_tensor(v) and v.is_floating_point()
                else v
            )
            for k, v in state.items()
        }

    model = F.build_model(method, F.DTYPE_EVAL)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, ck, sel


def clean_history_frame(d):
    required = {"combined_step", "total_loss"}
    missing = required - set(d.columns)
    if missing:
        raise KeyError(f"history missing required columns: {sorted(missing)}")

    out = d.copy()
    out["combined_step"] = pd.to_numeric(out["combined_step"], errors="coerce")
    out["total_loss"] = pd.to_numeric(out["total_loss"], errors="coerce")

    if "formal_elapsed_seconds" in out.columns:
        out["formal_elapsed_seconds"] = pd.to_numeric(
            out["formal_elapsed_seconds"], errors="coerce"
        )

    out = out.dropna(subset=["combined_step", "total_loss"])
    out = out[
        np.isfinite(out["combined_step"])
        & np.isfinite(out["total_loss"])
        & (out["total_loss"] > 0.0)
    ].copy()

    out["combined_step"] = out["combined_step"].astype(int)
    out = (
        out.sort_values("combined_step")
        .drop_duplicates("combined_step", keep="last")
        .reset_index(drop=True)
    )
    return out


def original_history(method, seed):
    path = F.fixed_out_dir(method, seed) / "training_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return clean_history_frame(pd.read_csv(path)), path


def continuation_combined_history(method, seed):
    ddir = continuation_dir(method, seed)

    combined_path = ddir / "training_history_original_plus_extra10000.csv"

    if combined_path.exists():
        d = clean_history_frame(pd.read_csv(combined_path))
        if int(d["combined_step"].max()) >= TARGET_COMBINED_STEP:
            d = d[d["combined_step"] <= TARGET_COMBINED_STEP].copy()
            return d, combined_path

    extra_path = ddir / "continued_lbfgs_extra10000_history.csv"
    if not extra_path.exists():
        raise FileNotFoundError(
            "Continuation reached 40000 but no usable continuation history "
            f"was found in {ddir}"
        )

    old, _ = original_history(method, seed)
    extra = pd.read_csv(extra_path)

    required = {"combined_step", "total_loss"}
    missing = required - set(extra.columns)
    if missing:
        raise KeyError(f"{extra_path} missing columns: {sorted(missing)}")

    extra["combined_step"] = pd.to_numeric(extra["combined_step"], errors="coerce")
    extra["total_loss"] = pd.to_numeric(extra["total_loss"], errors="coerce")
    extra = extra.dropna(subset=["combined_step", "total_loss"]).copy()
    extra["combined_step"] = extra["combined_step"].astype(int)

    if "formal_elapsed_seconds_extra" in extra.columns:
        original_ck = load_checkpoint(original_checkpoint_path(method, seed))
        original_summary = (
            original_ck.get("summary", {})
            if isinstance(original_ck, dict)
            else {}
        )
        old_time = finite_or_nan(
            original_summary.get("formal_training_time_seconds", np.nan)
        )

        if np.isfinite(old_time):
            extra["formal_elapsed_seconds"] = (
                pd.to_numeric(
                    extra["formal_elapsed_seconds_extra"], errors="coerce"
                )
                + old_time
            )

    extra = clean_history_frame(extra)
    extra = extra[
        (extra["combined_step"] > int(old["combined_step"].max()))
        & (extra["combined_step"] <= TARGET_COMBINED_STEP)
    ]

    combined = pd.concat([old, extra], ignore_index=True, sort=False)
    combined = clean_history_frame(combined)
    return combined, extra_path


def estimate_last_step_time_increment(hist):
    if "formal_elapsed_seconds" not in hist.columns:
        return np.nan, "unavailable_no_formal_elapsed_seconds"

    d = hist[["combined_step", "formal_elapsed_seconds"]].copy()
    d["formal_elapsed_seconds"] = pd.to_numeric(
        d["formal_elapsed_seconds"], errors="coerce"
    )
    d = d.dropna()
    d = d[np.isfinite(d["formal_elapsed_seconds"])]
    d = d.sort_values("combined_step").drop_duplicates(
        "combined_step", keep="last"
    )

    r29999 = d[d["combined_step"] == 29999]
    r30000 = d[d["combined_step"] == 30000]

    if not r29999.empty and not r30000.empty:
        t0 = float(r29999.iloc[-1]["formal_elapsed_seconds"])
        t1 = float(r30000.iloc[-1]["formal_elapsed_seconds"])
        dt = t1 - t0
        if np.isfinite(dt) and dt >= 0.0:
            return dt, "exact_time_30000_minus_29999"

    if len(d) >= 2:
        a = d.iloc[-2]
        b = d.iloc[-1]
        ds = int(b["combined_step"]) - int(a["combined_step"])
        dt = float(b["formal_elapsed_seconds"]) - float(
            a["formal_elapsed_seconds"]
        )
        if ds > 0 and np.isfinite(dt) and dt >= 0.0:
            return dt / ds, "last_two_available_points_per_step"

    return np.nan, "unavailable_insufficient_time_history"


def pad_plateau_history_to_40000(method, seed):
    hist, source_path = original_history(method, seed)
    hist = hist[hist["combined_step"] <= TARGET_COMBINED_STEP].copy()

    last = hist.iloc[-1]
    last_step = int(last["combined_step"])
    last_loss = float(last["total_loss"])

    dt_per_step, dt_source = estimate_last_step_time_increment(hist)

    last_time = np.nan
    if "formal_elapsed_seconds" in hist.columns:
        last_time = finite_or_nan(
            last.get("formal_elapsed_seconds", np.nan)
        )

    if last_step < TARGET_COMBINED_STEP:
        new_steps = np.arange(
            last_step + 1, TARGET_COMBINED_STEP + 1, dtype=int
        )

        pad = pd.DataFrame({
            "stage": "plateau_padding",
            "stage_step": np.nan,
            "combined_step": new_steps,
            "total_loss": np.full(len(new_steps), last_loss, dtype=float),
            "training_n_interior": float(
                last.get("training_n_interior", np.nan)
            ),
        })

        if np.isfinite(last_time) and np.isfinite(dt_per_step):
            pad["formal_elapsed_seconds"] = (
                last_time + (new_steps - last_step) * dt_per_step
            )
        else:
            pad["formal_elapsed_seconds"] = np.nan

        hist = pd.concat([hist, pad], ignore_index=True, sort=False)

    hist = clean_history_frame(hist)

    projected_time = np.nan
    if "formal_elapsed_seconds" in hist.columns:
        row40000 = hist[hist["combined_step"] == TARGET_COMBINED_STEP]
        if not row40000.empty:
            projected_time = finite_or_nan(
                row40000.iloc[-1].get("formal_elapsed_seconds", np.nan)
            )

    audit = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "history_mode": "plateau_padded_to_40000",
        "source_history_path": str(source_path),
        "original_last_available_step": int(last_step),
        "original_last_available_loss": float(last_loss),
        "time_increment_seconds_per_step_used": finite_or_nan(dt_per_step),
        "time_increment_source": dt_source,
        "projected_formal_training_time_at_40000_seconds": finite_or_nan(
            projected_time
        ),
    }
    return hist, audit


def unified_history_to_40000(method, seed):
    reached, cont_path, reported_step = continuation_reached_40000(
        method, seed
    )

    if reached:
        hist, source = continuation_combined_history(method, seed)
        hist = hist[hist["combined_step"] <= TARGET_COMBINED_STEP].copy()

        row40000 = hist[hist["combined_step"] == TARGET_COMBINED_STEP]
        if row40000.empty:
            raise RuntimeError(
                f"{method} seed={seed} was classified as a 40000 run, "
                "but its combined history has no step 40000."
            )

        actual_time = np.nan
        if "formal_elapsed_seconds" in hist.columns:
            actual_time = finite_or_nan(
                row40000.iloc[-1].get("formal_elapsed_seconds", np.nan)
            )

        audit = {
            "method": method,
            "method_label": method_label(method),
            "seed": int(seed),
            "history_mode": "actual_trained_to_40000",
            "source_history_path": str(source),
            "original_last_available_step": np.nan,
            "original_last_available_loss": np.nan,
            "time_increment_seconds_per_step_used": np.nan,
            "time_increment_source": "not_applicable_actual_40000",
            "projected_formal_training_time_at_40000_seconds": actual_time,
            "continuation_checkpoint": str(cont_path),
            "reported_continuation_final_step": reported_step,
        }
        return hist, audit

    return pad_plateau_history_to_40000(method, seed)


def recompute_total_loss(model, seed):
    base = F.load_points_np(seed)
    monitor = F.make_tensors(base, base["interior"], F.DTYPE_EVAL)

    with torch.enable_grad():
        d = F.loss_components(model, monitor)
        losses = detached_losses(d)

    del monitor, d
    gc.collect()
    return losses


def evaluate_one(method, seed, time_audit):
    model, checkpoint, sel = build_eval_model(method, seed)

    with torch.enable_grad():
        physical = F.physical_quantities(model)

    losses = {}
    if RECOMPUTE_TOTAL_LOSS:
        losses = recompute_total_loss(model, seed)

    row = {
        "Ra_exponent": int(F.RA_EXPONENT),
        "Ra": float(F.RA),
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "checkpoint_path": str(sel["path"]),
        "model_source": sel["model_source"],
        "prediction_model_step": int(sel["prediction_step"]),
        "continuation_reached_40000": bool(
            sel["continuation_reached_40000"]
        ),
        "reported_continuation_final_step": sel[
            "reported_continuation_final_step"
        ],
        "formal_training_time_to_40000_seconds": finite_or_nan(
            time_audit.get(
                "projected_formal_training_time_at_40000_seconds", np.nan
            )
        ),
        "time_to_40000_mode": time_audit.get("history_mode", ""),
        "time_increment_seconds_per_step_used": finite_or_nan(
            time_audit.get("time_increment_seconds_per_step_used", np.nan)
        ),
        "time_increment_source": time_audit.get(
            "time_increment_source", ""
        ),
    }

    if losses:
        for key in ("total_loss", "pde_loss", "bc_loss", "f1_loss", "f2_loss"):
            row[key] = finite_or_nan(losses.get(key, np.nan))

    scalar_errors = []
    for key, benchmark in EXPLICIT_BENCHMARK.items():
        pred = finite_or_nan(physical.get(key, np.nan))
        err = relerr_percent(pred, benchmark)

        row[key] = pred
        row[key + "_benchmark"] = benchmark
        row[key + "_relative_error_percent"] = err

        if np.isfinite(err):
            scalar_errors.append(float(err))

    all_mre = float(np.mean(scalar_errors)) if scalar_errors else np.nan
    row[
        "all_reference_physical_quantities_mean_relative_error_percent"
    ] = all_mre
    row["mean_scalar_physical_relative_error_percent"] = all_mre
    row["number_of_reference_physical_quantities_used"] = int(
        len(scalar_errors)
    )

    del model, checkpoint
    gc.collect()
    return row


def build_all_unified_histories():
    history_map = {}
    audit_rows = []

    for method in METHODS:
        for seed in F.SEEDS:
            seed = int(seed)
            hist, audit = unified_history_to_40000(method, seed)

            if TARGET_COMBINED_STEP not in set(
                hist["combined_step"].astype(int)
            ):
                raise RuntimeError(
                    f"{method} seed={seed}: no step 40000 after unification."
                )

            history_map[(method, seed)] = hist
            audit_rows.append(audit)

            hist.to_csv(
                OUT / f"history_to40000_{method}_seed{seed}.csv",
                index=False,
            )

            print(
                f"HISTORY | {method_label(method):18s} | "
                f"seed={seed:5d} | {audit['history_mode']}"
            )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        OUT / "history_and_time_extension_audit.csv",
        index=False,
    )
    return history_map, audit_df


def write_benchmark_table():
    rows = []
    for key, value in EXPLICIT_BENCHMARK.items():
        rows.append({
            "Ra_exponent": int(F.RA_EXPONENT),
            "Ra": float(F.RA),
            "quantity": key,
            "quantity_label": quantity_label(key),
            "benchmark_value_used": value,
            "included_in_relative_error": (
                value is not None and np.isfinite(finite_or_nan(value))
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "explicit_benchmark_used.csv", index=False)
    return df


def build_individual_table(audit_df):
    rows = []
    skipped = []

    for method in METHODS:
        for seed in F.SEEDS:
            seed = int(seed)

            a = audit_df[
                (audit_df["method"] == method) & (audit_df["seed"] == seed)
            ]
            if a.empty:
                raise RuntimeError(
                    f"No time/history audit row for {method}, seed={seed}."
                )

            time_audit = a.iloc[0].to_dict()

            try:
                row = evaluate_one(method, seed, time_audit)
                rows.append(row)

                print(
                    f"MODEL   | {method_label(method):18s} | "
                    f"seed={seed:5d} | "
                    f"source={row['model_source']} | "
                    f"model step={row['prediction_model_step']} | "
                    f"loss={row['total_loss']:.6e}"
                )
            except Exception as exc:
                skipped.append({
                    "method": method,
                    "method_label": method_label(method),
                    "seed": seed,
                    "reason": repr(exc),
                })
                print(
                    f"SKIPPED | {method_label(method):18s} | "
                    f"seed={seed:5d} | {exc}"
                )

    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv(
        OUT / "skipped_model_checkpoint_results.csv", index=False
    )

    if not rows:
        raise RuntimeError("No usable model checkpoints were found.")

    df = pd.DataFrame(rows)

    order = {m: i for i, m in enumerate(METHODS)}
    df["_method_order"] = df["method"].map(order)
    df = (
        df.sort_values(["_method_order", "seed"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    df.to_csv(OUT / "individual_results_to40000.csv", index=False)
    return df, skipped_df


def build_stats_table(df, filename="mean_std_results_to40000.csv"):
    exclude = {
        "method",
        "method_label",
        "seed",
        "checkpoint_path",
        "model_source",
        "time_to_40000_mode",
        "time_increment_source",
    }

    numeric = []
    for c in df.columns:
        if c in exclude:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            numeric.append(c)

    rows = []

    for method in METHODS:
        d = df[df["method"] == method]
        if d.empty:
            continue

        r = {
            "method": method,
            "method_label": method_label(method),
            "number_of_seeds": int(d["seed"].nunique()),
            "available_seeds": ",".join(
                str(int(s)) for s in sorted(d["seed"].unique())
            ),
        }

        for c in numeric:
            vals = pd.to_numeric(d[c], errors="coerce").dropna()
            r[c + "_mean"] = float(vals.mean()) if len(vals) else np.nan
            r[c + "_sample_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            )
            r[c + "_valid_n"] = int(len(vals))

        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / filename, index=False)
    return out


def build_paper_ready_table(stats):
    rows = []

    for _, r in stats.iterrows():
        def fmt(metric):
            m = r.get(metric + "_mean", np.nan)
            s = r.get(metric + "_sample_std", np.nan)
            if pd.isna(m):
                return "NA"
            return (
                f"{m:.6e} ± {s:.6e}"
                if pd.notna(s)
                else f"{m:.6e}"
            )

        rows.append({
            "Method": r["method_label"],
            "Number of seeds": int(r["number_of_seeds"]),
            "Recomputed final total loss": fmt("total_loss"),
            "Mean physical relative error (%)": fmt(
                "all_reference_physical_quantities_mean_relative_error_percent"
            ),
            "Formal training time to 40000 (s)": fmt(
                "formal_training_time_to_40000_seconds"
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paper_ready_results_to40000.csv", index=False)
    return out


def build_loss_curve_statistics(history_map):
    outputs = []

    for method in METHODS:
        frames = []
        used_seeds = []

        for seed in F.SEEDS:
            seed = int(seed)
            d = history_map[(method, seed)][
                ["combined_step", "total_loss"]
            ].copy()
            d = d.rename(
                columns={"total_loss": f"loss_seed_{seed}"}
            )
            frames.append(d)
            used_seeds.append(seed)

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="combined_step", how="inner")

        if merged.empty:
            raise RuntimeError(
                f"No common loss steps across seeds for {method}."
            )

        loss_cols = [f"loss_seed_{s}" for s in used_seeds]
        values = merged[loss_cols].to_numpy(dtype=float)

        valid = (
            np.all(np.isfinite(values), axis=1)
            & np.all(values > 0.0, axis=1)
        )

        merged = merged.loc[valid].copy()
        values = values[valid]

        log_values = np.log10(values)
        log_mean = np.mean(log_values, axis=1)
        log_std = np.std(log_values, axis=1, ddof=1)

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

    out = pd.concat(outputs, ignore_index=True)
    out.to_csv(
        OUT / "four_method_loss_log_statistics_to40000.csv", index=False
    )
    return out


def plot_loss_burgers_style(loss_stats):
    configure_burgers_loss_style()

    fig, ax = plt.subplots(
        figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT)
    )

    for method in METHODS:
        frame = loss_stats[loss_stats["method"] == method].copy()
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

    ax.set_yscale("log")
    ax.set_xlim(1, TARGET_COMBINED_STEP)

    if LOSS_YLIM is not None:
        ax.set_ylim(*LOSS_YLIM)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=1000,
        )
    )

    ax.grid(
        True, which="major", axis="y",
        linestyle="--", linewidth=0.8, alpha=0.55
    )
    ax.grid(
        True, which="minor", axis="y",
        linestyle="--", linewidth=0.45, alpha=0.32
    )
    ax.grid(
        True, which="major", axis="x",
        linestyle="-", linewidth=0.6, alpha=0.25
    )

    ax.legend(
        frameon=True,
        handlelength=2.5,
        handletextpad=0.8,
    )

    fig.tight_layout()

    path = OUT / "Ra1e6_four_method_total_loss_to40000_burgers_style.png"
    fig.savefig(
        path, dpi=IMAGE_DPI, format="png", bbox_inches="tight"
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)
    return path


def _log_safe_yerr(means, stds):
    means = np.asarray(means, float)
    stds = np.nan_to_num(
        np.asarray(stds, float),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    lower = np.minimum(stds, 0.95 * means)
    return np.vstack([lower, stds])


def bar_with_seed_points(
    df,
    metric,
    ylabel,
    filename,
    log_y=False,
    scientific=False,
):
    if df is None or df.empty or metric not in df.columns:
        return

    methods = [m for m in METHODS if m in set(df["method"])]
    x = np.arange(len(methods), dtype=float)

    means = np.array([
        pd.to_numeric(
            df[df.method == m][metric], errors="coerce"
        ).dropna().mean()
        for m in methods
    ], dtype=float)

    stds = np.array([
        pd.to_numeric(
            df[df.method == m][metric], errors="coerce"
        ).dropna().std(ddof=1)
        for m in methods
    ], dtype=float)

    stds = np.nan_to_num(
        stds, nan=0.0, posinf=0.0, neginf=0.0
    )

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
    colors = {
        9999: "#1f77b4",
        1234: "#ff7f0e",
        2026: "#2ca02c",
    }

    for seed in F.SEEDS:
        seed = int(seed)
        xs, ys = [], []

        for i, method in enumerate(methods):
            g = df[(df.method == method) & (df.seed == seed)]
            if g.empty:
                continue

            value = finite_or_nan(g.iloc[0][metric])
            if not np.isfinite(value):
                continue

            xs.append(i + offsets.get(seed, 0.0))
            ys.append(value)

        if xs:
            ax.scatter(
                xs,
                ys,
                marker=markers.get(seed, "o"),
                s=40,
                facecolors=colors.get(seed, None),
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods])
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")

    ax.grid(True, axis="y", which="both", alpha=0.35)
    ax.legend(frameon=True, ncol=3)

    for rect, value in zip(bars, means):
        if not np.isfinite(value):
            continue

        label = f"{value:.2e}" if scientific else f"{value:.2f}"

        ax.annotate(
            label,
            xy=(
                rect.get_x() + rect.get_width() * 0.90,
                value * 0.90,
            ),
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
    fig.savefig(
        OUT / filename,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


def make_bar_figures(df):
    bar_with_seed_points(
        df,
        "total_loss",
        "Recomputed total loss",
        "Ra1e6_final_recomputed_total_loss_to40000.png",
        log_y=True,
        scientific=True,
    )

    bar_with_seed_points(
        df,
        "all_reference_physical_quantities_mean_relative_error_percent",
        "Mean relative error of all physical quantities (%)",
        "Ra1e6_all_physical_quantities_mean_relative_error_to40000.png",
    )

    bar_with_seed_points(
        df,
        "formal_training_time_to_40000_seconds",
        "Formal training time to 40000 (s)",
        "Ra1e6_formal_training_time_to40000.png",
    )

    for key, benchmark in EXPLICIT_BENCHMARK.items():
        if benchmark is None:
            continue

        metric = key + "_relative_error_percent"

        bar_with_seed_points(
            df,
            metric,
            f"Relative error of {quantity_label(key)} (%)",
            f"Ra1e6_{key}_relative_error_to40000.png",
        )


def main():
    configure_plot_style()

    print("=" * 120)
    print("Ra=1e6 four-method analysis to nominal iteration 40000")
    print(f"CPU-only analysis device: {ANALYSIS_DEVICE}")
    print("Forced plateau/original-model cases:", sorted(FORCE_ORIGINAL_AND_PAD))
    print("=" * 120)

    benchmark_df = write_benchmark_table()

    history_map, audit_df = build_all_unified_histories()

    df, skipped_df = build_individual_table(audit_df)

    stats = build_stats_table(df)
    paper = build_paper_ready_table(stats)

    loss_stats = build_loss_curve_statistics(history_map)
    loss_figure = plot_loss_burgers_style(loss_stats)

    configure_plot_style()
    make_bar_figures(df)

    manifest = {
        "version":
            "RA1E6_FOUR_METHOD_ANALYSIS_TO_40000_WITH_PLATEAU_PADDING_CPU_V1",
        "Ra_exponent": int(F.RA_EXPONENT),
        "Ra": float(F.RA),
        "device": str(ANALYSIS_DEVICE),
        "force_cpu": True,
        "target_combined_step": int(TARGET_COMBINED_STEP),
        "methods": METHODS,
        "seeds": [int(s) for s in F.SEEDS],
        "forced_plateau_cases": [
            {"method": m, "seed": int(s)}
            for m, s in sorted(FORCE_ORIGINAL_AND_PAD)
        ],
        "prediction_rule": (
            "use validated 40000 continuation checkpoint when available; "
            "otherwise use original final checkpoint"
        ),
        "loss_padding_rule": (
            "for non-40000 cases, copy last available total_loss through 40000"
        ),
        "time_padding_rule": (
            "prefer dt=time(30000)-time(29999); otherwise use final two "
            "available history points; extrapolate linearly to 40000"
        ),
        "loss_plot_style":
            "same Burgers-style settings as uploaded Ra1e6 analysis",
        "loss_figure": str(loss_figure),
    }

    (OUT / "analysis_to40000_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPrediction/model source by method and seed:")
    print(
        df[[
            "method_label",
            "seed",
            "model_source",
            "prediction_model_step",
            "continuation_reached_40000",
            "formal_training_time_to_40000_seconds",
            "time_to_40000_mode",
        ]].to_string(index=False)
    )

    print("\nMean ± std statistics:")
    print(stats.to_string(index=False))

    print("\nPaper-ready summary:")
    print(paper.to_string(index=False))

    if len(skipped_df):
        print(
            "\nSome model rows were skipped. See:",
            OUT / "skipped_model_checkpoint_results.csv",
        )

    print("\nOutputs:", OUT.resolve())


if __name__ == "__main__":
    main()
