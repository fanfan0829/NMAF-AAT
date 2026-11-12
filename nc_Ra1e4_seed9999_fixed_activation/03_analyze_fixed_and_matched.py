from pathlib import Path
import json, gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import nc_fixed_activation_common as F

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = F.RESULT_ROOT / "analysis_seed9999"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 9999
ANALYSIS_DEVICE = torch.device("cpu")
F.DEVICE = ANALYSIS_DEVICE

BAR_COLOR = "#C0B3D0"
DPI = 800


def style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "lines.linewidth": 1.7,
    })


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fixed_ckpt(method):
    return F.fixed_out_dir(method, SEED) / f"Ra1e{F.RA_EXPONENT}_{method}_seed{SEED}_final.pth"


def matched_ckpt(method):
    if method == "nmaf":
        return fixed_ckpt("nmaf")
    return F.matched_out_dir(method, SEED) / f"Ra1e{F.RA_EXPONENT}_{method}_seed{SEED}_matched_time.pth"


def model_from(path, method):
    ck = load_ckpt(path)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model = F.build_model(method, F.DTYPE_EVAL)
    model.load_state_dict(F.cast_state_dict(state, F.DTYPE_EVAL), strict=True)
    model.eval()
    return model, ck


def relerr_percent(pred, ref):
    if ref is None or not np.isfinite(float(ref)):
        return np.nan
    return abs(float(pred)-float(ref))/abs(float(ref))*100.0


def evaluate_checkpoint(method, path, protocol):
    model, ck = model_from(path, method)
    base = F.load_points_np(SEED)
    monitor = F.make_tensors(base, base["interior"], F.DTYPE_EVAL)
    with torch.enable_grad():
        losses = F.detached_losses(F.loss_components(model, monitor))
        q = F.physical_quantities(model)
    summary = ck.get("summary", {}) if isinstance(ck, dict) else {}

    row = {
        "protocol": protocol,
        "method": method,
        "method_label": F.METHOD_LABELS[method],
        "activation_formula": F.METHOD_FORMULAS[method],
        "seed": SEED,
        "checkpoint": str(path),
        "recomputed_total_loss": float(losses["total_loss"]),
        "recomputed_pde_loss": float(losses["pde_loss"]),
        "recomputed_bc_loss": float(losses["bc_loss"]),
    }
    if protocol == "fixed_budget":
        row.update({
            "formal_adam_time_seconds": summary.get("formal_adam_time_seconds", np.nan),
            "formal_lbfgs_time_seconds": summary.get("formal_lbfgs_time_seconds", np.nan),
            "formal_training_time_seconds": summary.get("formal_training_time_seconds", np.nan),
            "adam_steps_completed": summary.get("adam_steps", F.ADAM_STEPS),
            "lbfgs_closure_evaluations": summary.get("lbfgs_closure_evaluations", np.nan),
        })
    else:
        if method == "nmaf":

            row.update({
                "target_formal_training_time_seconds": summary.get("formal_training_time_seconds", np.nan),
                "selected_formal_training_time_seconds": summary.get("formal_training_time_seconds", np.nan),
                "unused_target_time_seconds": 0.0,
                "adam_steps_completed": summary.get("adam_steps", F.ADAM_STEPS),
                "lbfgs_outer_updates_completed": np.nan,
                "lbfgs_closure_evaluations_attempted": summary.get("lbfgs_closure_evaluations", np.nan),
                "source": "fixed-budget NMAF reference (not retrained)",
            })
        else:
            row.update({
                "target_formal_training_time_seconds": summary.get("target_formal_training_time_seconds", np.nan),
                "selected_formal_training_time_seconds": summary.get("selected_formal_training_time_seconds", np.nan),
                "unused_target_time_seconds": summary.get("unused_target_time_seconds", np.nan),
                "adam_steps_completed": summary.get("adam_steps_completed", np.nan),
                "lbfgs_outer_updates_completed": summary.get("lbfgs_outer_updates_completed", np.nan),
                "lbfgs_closure_evaluations_attempted": summary.get("lbfgs_closure_evaluations_attempted", np.nan),
                "source": "matched-time run",
            })

    errs=[]
    for key, ref in F.BENCHMARK.items():
        pred=float(q[key])
        err=relerr_percent(pred, ref)
        row[key]=pred
        row[key+"_benchmark"]=ref
        row[key+"_relative_error_percent"]=err
        if np.isfinite(err): errs.append(err)
    row["number_of_reference_physical_quantities_used"] = len(errs)
    row["all_reference_physical_quantities_mean_relative_error_percent"] = float(np.mean(errs))
    row["all_reference_physical_quantities_max_relative_error_percent"] = (
        float(np.max(errs)) if errs else np.nan
    )

    del model, monitor
    gc.collect()
    return row


def collect(protocol):
    rows=[]; skipped=[]
    for m in F.METHODS:
        p = fixed_ckpt(m) if protocol=="fixed_budget" else matched_ckpt(m)
        if not p.exists():
            skipped.append({"protocol":protocol,"method":m,"checkpoint":str(p),"reason":"missing"})
            continue
        try:
            row=evaluate_checkpoint(m,p,protocol)
            rows.append(row)
            print(
                f"USING {protocol:12s} | {F.METHOD_LABELS[m]:12s} | "
                f"loss={row['recomputed_total_loss']:.6e} | "
                f"mean physical err={row['all_reference_physical_quantities_mean_relative_error_percent']:.4f}% | "
                f"max physical err={row['all_reference_physical_quantities_max_relative_error_percent']:.4f}%"
            )
        except Exception as e:
            skipped.append({"protocol":protocol,"method":m,"checkpoint":str(p),"reason":repr(e)})
            print("SKIP",protocol,m,e)
    df=pd.DataFrame(rows)
    pd.DataFrame(skipped).to_csv(OUT/f"{protocol}_skipped.csv",index=False)
    if not df.empty:
        order={m:i for i,m in enumerate(F.METHODS)}
        df["_order"]=df.method.map(order)
        df=df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        df.to_csv(OUT/f"{protocol}_checkpoint_recomputed_results.csv",index=False)
    return df


def activation_formula_table():
    pd.DataFrame([{"method":m,"method_label":F.METHOD_LABELS[m],"formula":F.METHOD_FORMULAS[m]} for m in F.METHODS]).to_csv(OUT/"activation_formulas.csv",index=False)
    pd.DataFrame([{"quantity":k,"benchmark":v,"included_in_mean_relative_error": v is not None} for k,v in F.BENCHMARK.items()]).to_csv(OUT/"Ra1e4_explicit_benchmarks.csv",index=False)


def plot_fixed_loss_curves():
    style()
    groups = [
        ("common_plus_new", ["tanh","relu","sigmoid","swish","sine","gaussian","gelu","mish","xtanh","nmaf"]),
        ("hybrid", ["tanswish","sigswish","sigtanh","gausstanh","gausssigmoid","gaussswish","nmaf"]),
    ]
    for name, methods in groups:
        fig,ax=plt.subplots(figsize=(7.0,5.0))
        used=0
        for m in methods:
            p=F.fixed_out_dir(m,SEED)/"training_history.csv"
            if not p.exists(): continue
            d=pd.read_csv(p)
            if "combined_step" not in d or "total_loss" not in d: continue
            ax.plot(d["combined_step"],d["total_loss"],label=F.METHOD_LABELS[m])
            used+=1
        if not used:
            plt.close(fig); continue
        ax.set_yscale("log")
        ax.set_xlabel("Training Step / L-BFGS Function Evaluation")
        ax.set_ylabel("Total Loss")
        ax.grid(True,which="both",alpha=.4)
        ax.legend(frameon=True,ncol=2)
        fig.tight_layout()
        fig.savefig(OUT/f"Ra1e4_seed9999_fixed_loss_{name}.png",dpi=DPI,bbox_inches="tight")
        plt.close(fig)


def bar(df, metric, ylabel, fname, log=False, sci=False):
    if df.empty or metric not in df: return
    d=df[np.isfinite(pd.to_numeric(df[metric],errors="coerce"))].copy()
    if d.empty:return
    style(); fig,ax=plt.subplots(figsize=(10.8,5.5))
    x=np.arange(len(d)); vals=pd.to_numeric(d[metric]).to_numpy(float)
    bars=ax.bar(x,vals,color=BAR_COLOR,edgecolor="black",linewidth=.7,alpha=.8)
    ax.set_xticks(x); ax.set_xticklabels(d.method_label,rotation=45,ha="right")
    ax.set_ylabel(ylabel)
    if log: ax.set_yscale("log")
    ax.grid(True,axis="y",which="both",alpha=.35)
    for rect,v in zip(bars,vals):
        if not np.isfinite(v): continue
        lab=f"{v:.2e}" if sci else f"{v:.2f}"
        ax.annotate(lab,xy=(rect.get_x()+rect.get_width()*.90,v*.90),ha="right",va="top",fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.12",facecolor="white",edgecolor="none",alpha=.82))
    fig.tight_layout(); fig.savefig(OUT/fname,dpi=DPI,bbox_inches="tight"); plt.close(fig)


def make_paper_tables(fixed,matched):
    if not fixed.empty:
        cols=["method_label","recomputed_total_loss","formal_training_time_seconds","all_reference_physical_quantities_mean_relative_error_percent","all_reference_physical_quantities_max_relative_error_percent"]
        fixed[[c for c in cols if c in fixed]].to_csv(OUT/"fixed_budget_paper_summary.csv",index=False)
    if not matched.empty:
        cols=["method_label","target_formal_training_time_seconds","selected_formal_training_time_seconds","adam_steps_completed","lbfgs_outer_updates_completed","lbfgs_closure_evaluations_attempted","recomputed_total_loss","all_reference_physical_quantities_mean_relative_error_percent","all_reference_physical_quantities_max_relative_error_percent"]
        matched[[c for c in cols if c in matched]].to_csv(OUT/"matched_time_efficiency_paper_summary.csv",index=False)


def main():
    activation_formula_table()
    fixed=collect("fixed_budget")
    matched=collect("matched_time")
    make_paper_tables(fixed,matched)
    plot_fixed_loss_curves()
    if not fixed.empty:
        bar(fixed,"recomputed_total_loss","Recomputed total loss","Ra1e4_seed9999_fixed_final_loss.png",log=True,sci=True)
    if not matched.empty:
        bar(matched,"all_reference_physical_quantities_mean_relative_error_percent","Mean relative error of reference physical quantities (%)","Ra1e4_seed9999_matched_time_mean_physical_error.png")
    print("Analysis output:",OUT.resolve())

if __name__ == "__main__":
    main()
