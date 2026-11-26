import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ac_tanh_adaptive5_common import (
    EXPERIMENT_DIR, SEEDS, METHODS, method_label, fixed_history_path,
    fixed_final_checkpoint, load_fixed_model, evaluate_model,
)

METHOD_ORDER = METHODS
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis_fixed_20000_tanhbase_five_methods"
IMAGE_DPI = 800
SHOW_FIGURES = True
BAR_COLOR = "#C0B3D0"
FINAL_EVAL = 20000
LINESTYLES = {"tanh":"-", "gaaf":"--", "llaaf":"-.", "nlaaf":":", "tanh_aat":(0,(5,2))}


def style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size":14, "axes.labelsize":14, "legend.fontsize":11,
                         "xtick.labelsize":12, "ytick.labelsize":12, "lines.linewidth":1.2})


def read_hist(method, seed):
    p = fixed_history_path(method, seed)
    df = pd.read_csv(p)

    if "lbfgs_eval" not in df.columns:
        for c in ["eval", "iteration", "current_it"]:
            if c in df.columns: df = df.rename(columns={c:"lbfgs_eval"}); break
    if "total_loss" not in df.columns:
        for c in ["loss", "TotalLoss"]:
            if c in df.columns: df = df.rename(columns={c:"total_loss"}); break
    if "lbfgs_eval" not in df.columns or "total_loss" not in df.columns:
        raise RuntimeError(f"Cannot identify lbfgs_eval/total_loss columns in {p}: {list(df.columns)}")
    df = df[["lbfgs_eval", "total_loss"]].copy()
    df["lbfgs_eval"] = pd.to_numeric(df["lbfgs_eval"], errors="coerce")
    df["total_loss"] = pd.to_numeric(df["total_loss"], errors="coerce")
    df = df.dropna().sort_values("lbfgs_eval")
    df = df[df["lbfgs_eval"] <= FINAL_EVAL]
    if df.empty or df["lbfgs_eval"].max() < FINAL_EVAL:
        raise RuntimeError(f"{method_label(method)} seed={seed} does not reach eval {FINAL_EVAL}: max={df['lbfgs_eval'].max() if len(df) else None}")
    return df


def curve_stats(method):
    merged = None
    for seed in SEEDS:
        d = read_hist(method, seed).rename(columns={"total_loss":f"loss_{seed}"})
        merged = d if merged is None else merged.merge(d, on="lbfgs_eval", how="inner")
    vals = merged[[f"loss_{s}" for s in SEEDS]].to_numpy(float)
    if np.any(vals <= 0): raise RuntimeError(f"Nonpositive loss in {method}")
    logv = np.log10(vals)
    lm = logv.mean(axis=1); ls = logv.std(axis=1, ddof=1)
    out = pd.DataFrame({"lbfgs_eval":merged["lbfgs_eval"], "geometric_mean_total_loss":10**lm,
                        "log10_sample_std":ls, "lower_band":10**(lm-ls), "upper_band":10**(lm+ls)})
    for s in SEEDS: out[f"total_loss_seed_{s}"] = merged[f"loss_{s}"].to_numpy(float)
    out.to_csv(ANALYSIS_DIR/f"{method}_total_loss_log_statistics.csv", index=False)
    return out


def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    for method in METHOD_ORDER:
        d = curve_stats(method)
        line, = ax.plot(d["lbfgs_eval"], d["geometric_mean_total_loss"],
                        linestyle=LINESTYLES[method], label=method_label(method))
        ax.fill_between(d["lbfgs_eval"], d["lower_band"], d["upper_band"],
                        color=line.get_color(), alpha=0.13, linewidth=0)
    ax.set_yscale("log"); ax.set_xlim(0, FINAL_EVAL)
    ax.set_xlabel("L-BFGS Function Evaluations"); ax.set_ylabel("Total loss")
    ax.grid(True, which="both", alpha=0.35); ax.legend(frameon=True)
    fig.tight_layout(); fig.savefig(ANALYSIS_DIR/"allen_cahn_tanhbase_five_methods_total_loss.png", dpi=IMAGE_DPI, format="png", bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)


def final_rows():
    rows = []
    for method in METHOD_ORDER:
        for seed in SEEDS:
            model, ck = load_fixed_model(method, seed)
            actual = ck.get("actual_lbfgs_eval", ck.get("lbfgs_eval", FINAL_EVAL)) if isinstance(ck, dict) else FINAL_EVAL
            if int(actual) < FINAL_EVAL:
                raise RuntimeError(f"Invalid fixed-budget checkpoint: {method} seed={seed}, actual_eval={actual}")
            latest = ck.get("latest_metrics", {}) if isinstance(ck, dict) else {}
            loss = latest.get("total_loss")
            if loss is None:
                h = read_hist(method, seed); loss = float(h.loc[h["lbfgs_eval"] <= FINAL_EVAL].iloc[-1]["total_loss"])
            metrics = evaluate_model(model)
            rows.append({"method":method, "method_label":method_label(method), "seed":seed,
                         "total_loss":float(loss), "relative_l2":metrics["relative_l2_error"],
                         "maximum_absolute_error":metrics["maximum_absolute_error"]})
    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS_DIR/"individual_final_metrics.csv", index=False)
    return df


def stats_table(df):
    rows=[]
    for m in METHOD_ORDER:
        g=df[df.method==m]
        row={"method":m,"method_label":method_label(m),"number_of_seeds":len(g)}
        for metric in ["total_loss","relative_l2","maximum_absolute_error"]:
            v=g[metric].to_numpy(float); row[f"{metric}_mean"]=v.mean(); row[f"{metric}_sample_std"]=v.std(ddof=1)
        rows.append(row)
    s=pd.DataFrame(rows); s.to_csv(ANALYSIS_DIR/"mean_std_final_metrics.csv", index=False)
    paper=[]
    for _,r in s.iterrows():
        paper.append({"Method":r.method_label,
                      "Total loss":f"{r.total_loss_mean:.6e} ± {r.total_loss_sample_std:.6e}",
                      "Relative L2 error":f"{r.relative_l2_mean:.6e} ± {r.relative_l2_sample_std:.6e}",
                      "Maximum absolute error":f"{r.maximum_absolute_error_mean:.6e} ± {r.maximum_absolute_error_sample_std:.6e}"})
    pd.DataFrame(paper).to_csv(ANALYSIS_DIR/"paper_ready_mean_std_table.csv", index=False)
    return s


def yerr_log(means,stds):
    return np.vstack([np.minimum(stds,0.95*means),stds])


def bar(df, stats, metric, ylabel, filename):
    methods=METHOD_ORDER; x=np.arange(len(methods),dtype=float)
    means=np.array([stats.loc[stats.method==m,f"{metric}_mean"].iloc[0] for m in methods])
    stds=np.array([stats.loc[stats.method==m,f"{metric}_sample_std"].iloc[0] for m in methods])
    fig,ax=plt.subplots(figsize=(8.7,5.5))
    bars=ax.bar(x,means,yerr=yerr_log(means,stds),capsize=5,alpha=.78,color=BAR_COLOR,edgecolor="black",linewidth=.7)
    offs=np.linspace(-.12,.12,len(SEEDS))
    for i,m in enumerate(methods):
        g=df[df.method==m].set_index("seed")
        for off,seed in zip(offs,SEEDS): ax.scatter(i+off,float(g.loc[seed,metric]),s=28,alpha=.75,zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([method_label(m) for m in methods]); ax.set_ylabel(ylabel); ax.set_yscale("log")
    ax.grid(True,axis="y",which="both",alpha=.35); ax.set_axisbelow(True)
    for rect,v in zip(bars,means):
        ax.annotate(f"{v:.2e}",xy=(rect.get_x()+0.90*rect.get_width(),0.90*v),ha="right",va="top",fontsize=9,
                    bbox=dict(facecolor="white",alpha=.65,edgecolor="none",pad=.7))
    fig.tight_layout(); fig.savefig(ANALYSIS_DIR/filename,dpi=IMAGE_DPI,format="png",bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)


def main():
    ANALYSIS_DIR.mkdir(parents=True,exist_ok=True); style(); plot_loss_curves()
    df=final_rows(); s=stats_table(df)
    bar(df,s,"total_loss","Total loss","fixed20000_total_loss_bar.png")
    bar(df,s,"relative_l2",r"Relative $L_2$ error","fixed20000_relative_l2_bar.png")
    bar(df,s,"maximum_absolute_error","Maximum absolute error","fixed20000_max_abs_error_bar.png")
    manifest={"problem":"Allen-Cahn","methods":[method_label(m) for m in METHOD_ORDER],
              "base_activation":"tanh for all five methods","fixed_budget":"20000 L-BFGS function evaluations",
              "loss_curve_statistics":"geometric mean; +/-1 sample std in log10(loss)",
              "final_metric_statistics":"arithmetic mean +/- sample std with three seed points",
              "slope_recovery":False,"plot_format":"lossless PNG, 800 dpi"}
    (ANALYSIS_DIR/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(s.to_string(index=False)); print(f"\nOutputs: {ANALYSIS_DIR.resolve()}")

if __name__=="__main__": main()
