from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ac_sensitivity_common import (
    RESULT_ROOT, TRAIN_EXPERIMENTS, BASELINE_SPECS,
    SEEDS, METHODS, DEVICE, DTYPE,
    build_model, method_label, count_parameters, count_adaptive_parameters,
    nominal_training_count, physical_coordinate_count, sample_counts,
    baseline_checkpoint_path, reference_file, verify_required_baseline_files,
)

OUT = RESULT_ROOT / "postprocessed_lines_L2_labels_inside_v7"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_DPI = 800
SHOW_FIGURES = False
PRED_BATCH_SIZE = 32768


METHOD_COLORS = {
    "tanh": "#9BBCE0",
    "nmaf_aat": "#C44E52",
}


SEED_COLORS = {
    2026: "#1f77b4",
    99: "#ff7f0e",
    1234: "#2ca02c",
}
SEED_MARKERS = {
    2026: "o",
    99: "s",
    1234: "^",
}

METHOD_LINESTYLES = {"tanh": "--", "nmaf_aat": "-"}
FIGSIZE = (8.7, 5.5)
GROUP_WIDTH = 0.72
BAR_ALPHA = 0.78
BAR_EDGE = "black"
BAR_EDGE_WIDTH = 0.7
CAPSIZE = 5

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

STUDIES = {
    "depth": {
        "levels": [5.0, 7.0, 9.0],
        "labels": ["5", "7", "9"],
        "xlabel": "Number of hidden layers",
        "prefix": "sensitivity_depth",
    },
    "width": {
        "levels": [64.0, 128.0, 192.0],
        "labels": ["64", "128", "192"],
        "xlabel": "Neurons per hidden layer",
        "prefix": "sensitivity_width",
    },
    "data": {
        "levels": [0.5, 1.0, 2.0, 4.0],
        "labels": ["0.5×", "1×", "2×", "4×"],
        "xlabel": "Training-data multiplier",
        "prefix": "sensitivity_data",
    },
    "noise": {
        "levels": [0.0, 1.0, 5.0, 10.0],
        "labels": ["0", "1", "5", "10"],
        "xlabel": "Initial condition noise level (%)",
        "prefix": "robustness_noise",
    },
}


def torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_reference():
    d = scipy.io.loadmat(reference_file())
    x = np.asarray(d["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(d["tt"], dtype=np.float64).reshape(-1)
    u = np.real(np.asarray(d["uu"])).astype(np.float64)
    if u.shape == (x.size, t.size):
        u = u.T
    elif u.shape != (t.size, x.size):
        raise ValueError(f"Unexpected AC.mat uu shape: {u.shape}")
    X, T = np.meshgrid(x, t, indexing="xy")
    return X.reshape(-1,1), T.reshape(-1,1), u.reshape(-1)


def state_dict_from_checkpoint(ck):
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return ck["model_state_dict"]
    if isinstance(ck, dict) and ck and all(torch.is_tensor(v) for v in ck.values()):
        return ck
    raise ValueError("Checkpoint has no recognizable model_state_dict.")


def predict(model, x, t):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), PRED_BATCH_SIZE):
            xb = torch.tensor(x[i:i+PRED_BATCH_SIZE], dtype=DTYPE, device=DEVICE)
            tb = torch.tensor(t[i:i+PRED_BATCH_SIZE], dtype=DTYPE, device=DEVICE)
            out.append(model(xb, tb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def errors(pred, exact):
    rel = np.linalg.norm(pred-exact) / np.linalg.norm(exact)
    mx = np.max(np.abs(pred-exact))
    return float(rel), float(mx)


def baseline_summary_path(method, seed):
    return baseline_checkpoint_path(method, seed).parent / "training_summary.json"


def new_checkpoint_path(spec, method, seed):
    return RESULT_ROOT / spec.name / method / f"seed_{seed}" / "final_checkpoint.pth"


def new_summary_path(spec, method, seed):
    return RESULT_ROOT / spec.name / method / f"seed_{seed}" / "training_summary.json"


def metadata(ck, summary_path):
    loss = np.nan
    runtime = np.nan
    actual_eval = np.nan

    if isinstance(ck, dict):
        lm = ck.get("latest_metrics", {}) or {}
        loss = float(lm.get("total_loss", np.nan))
        runtime = float(ck.get("elapsed_seconds", ck.get("total_time_seconds", np.nan)))
        actual_eval = float(
            ck.get("actual_eval",
            ck.get("actual_lbfgs_eval",
            ck.get("lbfgs_eval",
            ck.get("final_actual_lbfgs_eval", np.nan))))
        )

    if summary_path.is_file():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        if not np.isfinite(loss):
            fm = s.get("final_metrics", {}) or {}
            loss = float(fm.get("total_loss", np.nan))
        if not np.isfinite(runtime):
            runtime = float(s.get("total_time_seconds", np.nan))
        if not np.isfinite(actual_eval):
            actual_eval = float(
                s.get("actual_eval", s.get("final_actual_lbfgs_eval", np.nan))
            )

    return loss, runtime, actual_eval


def eval_one(spec, method, seed, baseline, ref):
    xf, tf, exact = ref
    if baseline:
        ckpt_path = baseline_checkpoint_path(method, seed)
        summary_path = baseline_summary_path(method, seed)
    else:
        ckpt_path = new_checkpoint_path(spec, method, seed)
        summary_path = new_summary_path(spec, method, seed)

    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    ck = torch_load(ckpt_path)
    model = build_model(method, spec.depth, spec.width)
    model.load_state_dict(state_dict_from_checkpoint(ck), strict=True)

    pred = predict(model, xf, tf)
    rel, mx = errors(pred, exact)
    loss, runtime, actual_eval = metadata(ck, summary_path)

    nres, nbc, nic = sample_counts(spec)
    nparam = count_parameters(model)

    row = {
        "study": spec.study,
        "experiment": "baseline" if baseline else spec.name,
        "level": float(spec.level),
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "baseline": bool(baseline),
        "depth": int(spec.depth),
        "width": int(spec.width),
        "data_factor": float(spec.data_factor),
        "noise_pct": float(spec.noise_pct),
        "n_res": int(nres),
        "n_bc": int(nbc),
        "n_ic": int(nic),
        "final_total_loss": float(loss),
        "relative_L2_error": rel,
        "maximum_absolute_error": mx,
        "runtime_s": float(runtime),
        "actual_lbfgs_evals": float(actual_eval),
        "total_parameters": int(nparam),
        "adaptive_parameters": int(count_adaptive_parameters(model)),
        "nominal_training_count": int(nominal_training_count(spec)),
        "physical_coordinate_count": int(physical_coordinate_count(spec)),
        "parameter_to_nominal_point_ratio": float(nparam / nominal_training_count(spec)),
        "parameter_to_coordinate_point_ratio": float(nparam / physical_coordinate_count(spec)),
    }

    print(
        f"[POST] {row['study']:5s} | {row['level']:g} | "
        f"{row['method_label']:13s} | seed {seed:4d} | "
        f"loss={row['final_total_loss']:.3e} | "
        f"L2={row['relative_L2_error']:.3e}"
    )

    del model, ck
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def collect_all():
    ref = load_reference()
    rows = []


    bspec = BASELINE_SPECS["depth"]
    for seed in SEEDS:
        for method in METHODS:
            rows.append(eval_one(bspec, method, seed, True, ref))


    for spec in TRAIN_EXPERIMENTS:
        for seed in SEEDS:
            for method in METHODS:
                rows.append(eval_one(spec, method, seed, False, ref))

    return pd.DataFrame(rows)


def study_frame(all_df, study):
    base = all_df[all_df["baseline"]].copy()
    base["study"] = study
    base["level"] = float(BASELINE_SPECS[study].level)
    non = all_df[(~all_df["baseline"]) & (all_df["study"] == study)].copy()
    return pd.concat([base, non], ignore_index=True).sort_values(
        ["level","method","seed"]
    ).reset_index(drop=True)


def summarize(df):
    rows = []
    for (level, method, label), g in df.groupby(
        ["level","method","method_label"], sort=True
    ):
        r = {"level": float(level), "method": method, "method_label": label}
        for metric in [
            "final_total_loss","relative_L2_error","maximum_absolute_error",
            "runtime_s","actual_lbfgs_evals"
        ]:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(float)
            r[metric+"_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            r[metric+"_std"] = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0

        lv = pd.to_numeric(g["final_total_loss"], errors="coerce").dropna().to_numpy(float)
        lv = lv[lv>0]
        logs = np.log10(lv)
        r["final_total_loss_geomean"] = float(10**logs.mean())
        r["final_total_loss_log10_std"] = float(logs.std(ddof=1)) if len(logs)>1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["level","method"]).reset_index(drop=True)


def log_yerr(means, logstd):
    lower = means - means/(10**logstd)
    upper = means*(10**logstd) - means
    return np.vstack([lower, upper])


def method_centers(n_groups, method_index):
    x = np.arange(n_groups, dtype=float)
    bw = GROUP_WIDTH / len(METHODS)
    start = -GROUP_WIDTH/2 + bw/2
    return x + start + method_index*bw


def plot_metric(study, df, st, metric, ylabel, filename=None, log=False,
                ax=None, show_legend=True):
    info = STUDIES[study]
    levels = np.asarray(info["levels"], dtype=float)
    xg = np.arange(len(levels), dtype=float)
    bw = GROUP_WIDTH / len(METHODS)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = ax.figure
    bars_all = []

    for mi, method in enumerate(METHODS):
        s = st[st.method==method].set_index("level").reindex(levels)
        centers = method_centers(len(levels), mi)

        if metric == "final_total_loss":
            means = s["final_total_loss_geomean"].to_numpy(float)
            lstd = s["final_total_loss_log10_std"].to_numpy(float)
            yerr = log_yerr(means, lstd)
        else:
            means = s[metric+"_mean"].to_numpy(float)
            std = s[metric+"_std"].to_numpy(float)
            yerr = std

        bars = ax.bar(
            centers, means,
            width=bw*0.88,
            yerr=yerr,
            capsize=CAPSIZE,
            alpha=BAR_ALPHA,
            color=METHOD_COLORS[method],
            edgecolor=BAR_EDGE,
            linewidth=BAR_EDGE_WIDTH,
            zorder=2,
        )


        line_values = means.copy()
        line_values[~np.isfinite(line_values)] = np.nan
        if log:
            line_values[line_values <= 0] = np.nan
        ax.plot(centers, line_values, color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method], linewidth=1.8, zorder=3)


        seed_local_offsets = {2026:-0.10*bw, 99:0.0, 1234:0.10*bw}
        for seed in (2026,99,1234):
            g = df[(df.method==method)&(df.seed==seed)].set_index("level").reindex(levels)
            ys = g[metric].to_numpy(float)
            valid = np.isfinite(ys)
            if log:
                valid &= ys>0
            xs = centers + seed_local_offsets[seed]
            ax.scatter(
                xs[valid], ys[valid],
                marker=SEED_MARKERS[seed],
                s=40,
                facecolors=SEED_COLORS[seed],
                edgecolors="black",
                linewidths=0.6,
                label=f"seed {seed}" if mi==0 else None,
                zorder=4,
            )

        for rect, val in zip(bars, means):
            bars_all.append((rect,float(val)))

    ax.set_xticks(xg)
    ax.set_xticklabels(info["labels"])
    ax.set_xlabel(info["xlabel"])
    ax.set_ylabel(ylabel)

    if log:
        ax.set_yscale("log")

    ax.grid(True, axis="y", which="both", alpha=0.35, zorder=0)

    if show_legend:
        ax.legend(handles=legend_handles(), frameon=True, ncol=2)


    for rect, value in bars_all:
        if not np.isfinite(value) or value <= 0:
            continue
        ax.annotate(
            f"{value:.2e}",
            xy=(rect.get_x()+rect.get_width()*0.90, value*0.90),
            xytext=(0,0),
            textcoords="offset points",
            ha="right", va="top",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=5,
        )

    if not standalone:
        return ax

    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print("Saved:", path)


def legend_handles():

    methods = [Line2D([0], [0], color=METHOD_COLORS[m],
                      linestyle=METHOD_LINESTYLES[m], linewidth=2.2,
                      marker="s", markersize=8,
                      markerfacecolor=METHOD_COLORS[m],
                      markeredgecolor="black", markeredgewidth=0.7,
                      label=method_label(m)) for m in METHODS]
    seeds = [Line2D([0], [0], linestyle="None", marker=SEED_MARKERS[s],
                    markersize=7, markerfacecolor=SEED_COLORS[s],
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=f"seed {s}") for s in (2026, 99, 1234)]
    return methods + seeds


def plot_L2_main(all_df):

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    for letter, study, ax in zip("abcd", ("depth", "width", "data", "noise"), axes.flat):
        df = study_frame(all_df, study)
        plot_metric(study, df, summarize(df), "relative_L2_error",
                    r"Relative $L_2$ error", ax=ax, show_legend=False)
        ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes,
                ha="left", va="top", fontsize=13, zorder=6)
    handles = legend_handles()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(handles), frameon=False, fontsize=12,
               handlelength=2.8, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.0, w_pad=2.0)
    path = OUT / "sensitivity_robustness_L2_main_4panels_labels_inside_v7.png"
    fig.savefig(path, dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print("Saved:", path)


def plot_study(study, df, st):
    p = STUDIES[study]["prefix"]
    plot_metric(
        study, df, st,
        "final_total_loss", "Final total loss",
        p+"_final_total_loss_BAR.png", log=True
    )
    plot_metric(
        study, df, st,
        "relative_L2_error", r"Relative $L_2$ error",
        p+"_relative_L2_error_BAR.png", log=False
    )
    plot_metric(
        study, df, st,
        "maximum_absolute_error", "Maximum absolute error",
        p+"_maximum_absolute_error_BAR.png", log=False
    )


def main():
    print("="*100)
    print("Allen-Cahn sensitivity / robustness — BAR post-processing")
    print("No training will be performed.")
    print("="*100)

    verify_required_baseline_files(require_reference=True)
    all_df = collect_all()
    all_df.to_csv(OUT/"all_unique_runs.csv", index=False)

    for study in ("depth","width","data","noise"):
        df = study_frame(all_df, study)
        st = summarize(df)
        df.to_csv(OUT/f"{study}_all_seeds.csv", index=False)
        st.to_csv(OUT/f"{study}_mean_std.csv", index=False)
        plot_study(study, df, st)

    plot_L2_main(all_df)

    print("\nFinished. All grouped bar charts are in:")
    print(OUT)


if __name__ == "__main__":
    main()
