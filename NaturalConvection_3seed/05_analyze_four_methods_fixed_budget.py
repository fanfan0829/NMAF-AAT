from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import nc_common as F

METHODS = ["tanh", "rar", "stan", "nmaf_aat"]
OUT = F.RESULT_ROOT / "four_method_analysis"
OUT.mkdir(parents=True, exist_ok=True)
BAR_COLOR = "#C0B3D0"
IMAGE_DPI = 800
SHOW_FIGURES = True


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "lines.linewidth": 1.0,
    })


def load_summary(method, seed):
    p = F.formal_summary_path(method, seed)
    if not p.exists(): raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def build_final_table():
    rows=[]
    for method in METHODS:
        for seed in F.SEEDS:
            obj=load_summary(method,seed)
            row={
                "method":method,"method_label":F.METHOD_LABELS[method],"seed":seed,
                "total_loss":obj["final_common_total_loss"],
                "formal_training_time_seconds":obj["formal_training_time_seconds"],
                "lbfgs_closure_evaluations":obj["lbfgs_closure_evaluations"],
            }
            for k,v in obj["physical_quantities"].items():
                row[k]=v
                b=F.BENCHMARK.get(k)
                row[k+"_relative_error_percent"] = F.relerr_percent(v,b) if b is not None else np.nan
            rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(OUT/"individual_fixed_budget_results.csv",index=False)
    stats=[]
    numeric=[c for c in df.columns if c not in {"method","method_label","seed"}]
    for method in METHODS:
        d=df[df.method==method]; r={"method":method,"method_label":F.METHOD_LABELS[method],"number_of_seeds":len(d)}
        for c in numeric:
            vals=pd.to_numeric(d[c],errors="coerce").dropna()
            r[c+"_mean"]=float(vals.mean()) if len(vals) else np.nan
            r[c+"_sample_std"]=float(vals.std(ddof=1)) if len(vals)>1 else np.nan
        stats.append(r)
    st=pd.DataFrame(stats); st.to_csv(OUT/"mean_std_fixed_budget_results.csv",index=False)
    return df,st


def log_stats():
    method_stats=[]
    for method in METHODS:
        frames=[]
        for seed in F.SEEDS:
            p=F.fixed_out_dir(method,seed)/"training_history.csv"
            d=pd.read_csv(p)[["combined_step","total_loss"]].rename(columns={"total_loss":f"loss_{seed}"})
            frames.append(d)
        m=frames[0]
        for d in frames[1:]: m=m.merge(d,on="combined_step",how="inner")
        vals=m[[f"loss_{s}" for s in F.SEEDS]].to_numpy(float)
        y=np.log10(vals); mu=y.mean(axis=1); sd=y.std(axis=1,ddof=1)
        st=pd.DataFrame({
            "combined_step":m.combined_step.astype(int),
            "geometric_mean_total_loss":10**mu,
            "lower_band":10**(mu-sd),"upper_band":10**(mu+sd),
        })
        st.insert(0,"method",method); method_stats.append(st)
    out=pd.concat(method_stats,ignore_index=True); out.to_csv(OUT/"four_method_loss_log_statistics.csv",index=False)
    return out


def plot_loss(st):
    fig,ax=plt.subplots(figsize=(8.7,5.5))
    styles={"tanh":"-","rar":"--","stan":"-.","nmaf_aat":":"}
    for method in METHODS:
        d=st[st.method==method]
        line,=ax.plot(d.combined_step,d.geometric_mean_total_loss,linestyle=styles[method],linewidth=1.6,label=F.METHOD_LABELS[method])
        ax.fill_between(d.combined_step,d.lower_band,d.upper_band,color=line.get_color(),alpha=0.14,linewidth=0)
    ax.axvline(F.ADAM_STEPS,linestyle="--",linewidth=0.8,alpha=0.6)
    ax.set_yscale("log"); ax.set_xlabel("Training index"); ax.set_ylabel("Total loss")
    ax.grid(True,axis="y",which="both",alpha=0.35); ax.legend(frameon=True)
    fig.tight_layout(); fig.savefig(OUT/f"Ra1e{F.RA_EXPONENT}_four_method_loss_mean_std.png",dpi=IMAGE_DPI,format="png",bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)


def _log_safe_yerr(means,stds):
    means=np.asarray(means,float); stds=np.asarray(stds,float)
    return np.vstack([np.minimum(stds,0.95*means),stds])


def bar_with_seed_points(df,metric,ylabel,filename,log_y=False,scientific=False):
    x=np.arange(len(METHODS),dtype=float)
    means=np.array([df[df.method==m][metric].astype(float).mean() for m in METHODS])
    stds=np.array([df[df.method==m][metric].astype(float).std(ddof=1) for m in METHODS])
    fig,ax=plt.subplots(figsize=(8.7,5.5))
    bars=ax.bar(x,means,yerr=_log_safe_yerr(means,stds) if log_y else stds,capsize=5,alpha=0.78,color=BAR_COLOR,edgecolor="black",linewidth=0.7)
    offsets={9999:-0.12,1234:0.0,2026:0.12}; markers={9999:"o",1234:"s",2026:"^"}; colors={9999:"#1f77b4",1234:"#ff7f0e",2026:"#2ca02c"}
    for seed in F.SEEDS:
        ys=[float(df[(df.method==m)&(df.seed==seed)].iloc[0][metric]) for m in METHODS]
        ax.scatter(x+offsets[seed],ys,marker=markers[seed],s=40,facecolors=colors[seed],edgecolors="black",linewidths=0.6,label=f"seed {seed}",zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([F.METHOD_LABELS[m] for m in METHODS]); ax.set_ylabel(ylabel)
    if log_y: ax.set_yscale("log")
    ax.grid(True,axis="y",which="both",alpha=0.35); ax.legend(frameon=True,ncol=3)
    for rect,value in zip(bars,means):
        label=f"{value:.2e}" if scientific else f"{value:.2f}"
        ax.annotate(label,xy=(rect.get_x()+rect.get_width()*0.90,value*0.90),xytext=(0,0),textcoords="offset points",ha="right",va="top",fontsize=10,bbox=dict(boxstyle="round,pad=0.15",facecolor="white",edgecolor="none",alpha=0.85),zorder=5)
    fig.tight_layout(); fig.savefig(OUT/filename,dpi=IMAGE_DPI,format="png",bbox_inches="tight")
    if SHOW_FIGURES: plt.show()
    plt.close(fig)


def main():
    configure_plot_style(); df,stats=build_final_table(); st=log_stats(); plot_loss(st)
    bar_with_seed_points(df,"total_loss","Total loss",f"Ra1e{F.RA_EXPONENT}_fixed_budget_total_loss.png",log_y=True,scientific=True)
    for k,b in F.BENCHMARK.items():
        if b is None: continue
        metric=k+"_relative_error_percent"
        bar_with_seed_points(df,metric,f"Relative error of {F.QUANTITY_LABELS[k]} (%)",f"Ra1e{F.RA_EXPONENT}_fixed_budget_{k}_relative_error.png")
    print(stats.to_string(index=False))

if __name__=="__main__": main()
