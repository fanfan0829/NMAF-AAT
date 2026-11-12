import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ac_tanh_adaptive5_common import (
    EXPERIMENT_DIR, NEW_TIME_RESULT_ROOT, SEEDS, METHODS, TIME_COMPETITORS,
    method_label, load_tanh_aat_reference,
)

METHOD_ORDER = METHODS
ANALYSIS_DIR = EXPERIMENT_DIR / "analysis_seedwise_tanh_aat_time_five_methods"
IMAGE_DPI = 800
SHOW_FIGURES = True
BAR_COLOR = "#C0B3D0"


def style():
    plt.rcParams["mathtext.fontset"]="custom"; plt.rcParams["mathtext.rm"]="Times New Roman"; plt.rcParams["mathtext.it"]="Times New Roman:italic"
    plt.rcParams["font.family"]="Times New Roman"
    plt.rcParams.update({"font.size":14,"axes.labelsize":14,"legend.fontsize":12,"xtick.labelsize":12,"ytick.labelsize":12,"lines.linewidth":1.0})


def summary_path(method,seed):
    return NEW_TIME_RESULT_ROOT/method/f"seed_{int(seed)}"/"seedwise_matched_summary.json"


def load_competitor(method,seed):
    p=summary_path(method,seed)
    if not p.exists(): raise FileNotFoundError(p)
    obj=json.loads(p.read_text(encoding="utf-8"))
    if obj.get("reference_method")!="tanh_aat" or int(obj.get("reference_seed"))!=int(seed): raise RuntimeError(f"Bad reference in {p}")
    if bool(obj.get("average_target_used",False)): raise RuntimeError(f"Averaged target used in {p}")
    return {"method":method,"method_label":method_label(method),"seed":int(seed),"source":str(p),
            "target_wall_time_seconds":float(obj["target_wall_time_seconds"]),
            "actual_training_time_seconds":float(obj["actual_stage_training_time_seconds"]),
            "time_overshoot_seconds":float(obj["time_overshoot_seconds"]),
            "loss":float(obj["monitor_total_loss"]),"relative_l2":float(obj["relative_l2_error"]),
            "maximum_absolute_error":float(obj["maximum_absolute_error"]),
            "lbfgs_outer_steps_completed":float(obj["lbfgs_outer_steps_completed"]),
            "lbfgs_closure_evals_completed":float(obj["lbfgs_closure_evals_completed"])}


def load_reference(seed):
    ref=load_tanh_aat_reference(seed)
    return {"method":"tanh_aat","method_label":"tanh-AAT","seed":int(seed),"source":ref["metric_source"],
            "target_wall_time_seconds":float(ref["target_wall_time_seconds"]),
            "actual_training_time_seconds":float(ref["target_wall_time_seconds"]),"time_overshoot_seconds":0.0,
            "loss":float(ref["tanh_aat_total_loss"]),"relative_l2":float(ref["tanh_aat_relative_l2_error"]),
            "maximum_absolute_error":float(ref["tanh_aat_maximum_absolute_error"]),
            "lbfgs_outer_steps_completed":np.nan,"lbfgs_closure_evals_completed":20000.0}


def individual():
    rows=[]
    for m in TIME_COMPETITORS:
        for s in SEEDS: rows.append(load_competitor(m,s))
    for s in SEEDS: rows.append(load_reference(s))
    df=pd.DataFrame(rows); order={m:i for i,m in enumerate(METHOD_ORDER)}
    df["_o"]=df.method.map(order); df=df.sort_values(["_o","seed"]).drop(columns="_o").reset_index(drop=True)
    df.to_csv(ANALYSIS_DIR/"seedwise_individual_runs.csv",index=False); return df


def stats(df):
    rows=[]
    for m in METHOD_ORDER:
        g=df[df.method==m]; row={"method":m,"method_label":method_label(m),"number_of_seeds":int(g.seed.nunique())}
        for metric in ["target_wall_time_seconds","actual_training_time_seconds","loss","relative_l2","maximum_absolute_error","lbfgs_closure_evals_completed"]:
            v=g[metric].to_numpy(float); v=v[np.isfinite(v)]
            row[f"{metric}_mean"]=v.mean() if len(v) else np.nan; row[f"{metric}_sample_std"]=v.std(ddof=1) if len(v)>1 else np.nan
        rows.append(row)
    st=pd.DataFrame(rows); st.to_csv(ANALYSIS_DIR/"seedwise_mean_std_statistics.csv",index=False)
    paper=[]
    for _,r in st.iterrows():
        def fmt(x): return f"{r[f'{x}_mean']:.6e} ± {r[f'{x}_sample_std']:.6e}"
        paper.append({"Method":r.method_label,"Total loss":fmt("loss"),"Relative L2 error":fmt("relative_l2"),
                      "Maximum absolute error":fmt("maximum_absolute_error"),"Actual training time (s)":fmt("actual_training_time_seconds"),
                      "L-BFGS closure evaluations":fmt("lbfgs_closure_evals_completed")})
    pd.DataFrame(paper).to_csv(ANALYSIS_DIR/"paper_ready_seedwise_tanh_aat_time_table.csv",index=False)
    return st


def paired(df):
    rows=[]
    for seed in SEEDS:
        ref=df[(df.method=="tanh_aat")&(df.seed==seed)].iloc[0]
        row={"seed":seed,"tanh_aat_target_wall_time_seconds":ref.target_wall_time_seconds}
        for m in METHOD_ORDER:
            r=df[(df.method==m)&(df.seed==seed)].iloc[0]
            row[f"{m}_actual_time_seconds"]=r.actual_training_time_seconds; row[f"{m}_loss"]=r.loss
            row[f"{m}_relative_l2"]=r.relative_l2; row[f"{m}_max_abs_error"]=r.maximum_absolute_error
            if m!="tanh_aat":
                row[f"{m}_loss_ratio_to_tanh_aat"]=r.loss/ref.loss; row[f"{m}_l2_ratio_to_tanh_aat"]=r.relative_l2/ref.relative_l2
        rows.append(row)
    pd.DataFrame(rows).to_csv(ANALYSIS_DIR/"paired_seedwise_results.csv",index=False)


def yerr(means,stds): return np.vstack([np.minimum(stds,.95*means),stds])

def bar(df,st,metric,ylabel,filename):
    methods=METHOD_ORDER; x=np.arange(len(methods),dtype=float)
    means=np.array([st.loc[st.method==m,f"{metric}_mean"].iloc[0] for m in methods]); stds=np.array([st.loc[st.method==m,f"{metric}_sample_std"].iloc[0] for m in methods])
    fig,ax=plt.subplots(figsize=(8.7,5.5)); bars=ax.bar(x,means,yerr=yerr(means,stds),capsize=5,alpha=.78,color=BAR_COLOR,edgecolor="black",linewidth=.7)
    offs=np.linspace(-.12,.12,len(SEEDS))
    for i,m in enumerate(methods):
        g=df[df.method==m].set_index("seed")
        for off,seed in zip(offs,SEEDS): ax.scatter(i+off,float(g.loc[seed,metric]),s=28,alpha=.75,zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([method_label(m) for m in methods]); ax.set_ylabel(ylabel); ax.set_yscale("log"); ax.grid(True,axis="y",which="both",alpha=.35); ax.set_axisbelow(True)
    for rect,v in zip(bars,means): ax.annotate(f"{v:.2e}",xy=(rect.get_x()+.90*rect.get_width(),.90*v),ha="right",va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.65,edgecolor="none",pad=.7))
    fig.tight_layout(); fig.savefig(ANALYSIS_DIR/filename,dpi=IMAGE_DPI,format="png",bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)


def main():
    ANALYSIS_DIR.mkdir(parents=True,exist_ok=True); style(); df=individual()
    for seed in SEEDS:
        g=df[df.seed==seed]; targets=g.target_wall_time_seconds.to_numpy(float)
        if not np.allclose(targets,targets[0],rtol=1e-9,atol=1e-9): raise RuntimeError(f"Target mismatch seed={seed}: {targets}")
    st=stats(df); paired(df)
    bar(df,st,"loss","Total loss","seedwise_tanh_aat_time_total_loss_bar.png")
    bar(df,st,"relative_l2",r"Relative $L_2$ error","seedwise_tanh_aat_time_relative_l2_bar.png")
    bar(df,st,"maximum_absolute_error","Maximum absolute error","seedwise_tanh_aat_time_max_abs_error_bar.png")
    manifest={"problem":"Allen-Cahn","methods":[method_label(m) for m in METHOD_ORDER],
              "protocol":"same-seed tanh-AAT matched wall-clock","reference_method":"tanh-AAT",
              "average_target_used":False,"tanh_aat_retrained_in_time_stage":False,"base_activation":"tanh for all five methods",
              "slope_recovery":False,"statistics":"arithmetic mean +/- sample std with three seed points","plot_format":"lossless PNG, 800 dpi"}
    (ANALYSIS_DIR/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(st.to_string(index=False)); print(f"\nOutputs: {ANALYSIS_DIR.resolve()}")

if __name__=="__main__": main()
