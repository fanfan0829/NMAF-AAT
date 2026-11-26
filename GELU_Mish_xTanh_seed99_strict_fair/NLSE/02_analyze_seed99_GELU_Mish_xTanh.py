import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import torch
import matplotlib.pyplot as plt
from nlse_fixed_activation_seed99_common import *
METHODS=["gelu","mish","xtanh"]; SEED=99; ANALYSIS_DIR=RESULT_ROOT/"three_activation_analysis_seed99"; REFERENCE_FILE=Path("NLS.mat")

def relative_l2(exact,pred):
    exact=np.asarray(exact,dtype=np.complex128).reshape(-1); pred=np.asarray(pred,dtype=np.complex128).reshape(-1)
    return float(np.linalg.norm(exact-pred,ord=2)/np.linalg.norm(exact,ord=2))

def load_reference():
    d=scipy.io.loadmat(REFERENCE_FILE); x=np.asarray(d["x"]).reshape(-1); t=np.asarray(d["tt"] if "tt" in d else d["t"]).reshape(-1); exact=np.asarray(d["uu"])
    shape=(x.size,t.size)
    if exact.shape!=shape:
        if exact.T.shape==shape: exact=exact.T
        else: raise ValueError(f"Unexpected uu shape {exact.shape}; expected {shape}")
    X,T=np.meshgrid(x,t,indexing="ij"); return X,T,exact
@torch.no_grad()
def predict(model,X,T,shape,batch_size=65536):
    xs=X.reshape(-1,1); ts=T.reshape(-1,1); parts=[]
    for i in range(0,len(xs),batch_size):
        x=torch.as_tensor(xs[i:i+batch_size],dtype=DTYPE,device=DEVICE); t=torch.as_tensor(ts[i:i+batch_size],dtype=DTYPE,device=DEVICE); parts.append(model(x,t).cpu().numpy())
    z=np.vstack(parts); u=z[:,0].reshape(shape); v=z[:,1].reshape(shape); return u+1j*v

def main():
    ANALYSIS_DIR.mkdir(parents=True,exist_ok=True)
    if not REFERENCE_FILE.exists(): raise FileNotFoundError(REFERENCE_FILE)
    X,T,exact=load_reference(); rows=[]; histories={}; manifest={}
    for method in METHODS:
        sd=RESULT_ROOT/method/f"seed_{SEED}"; hp=sd/"lbfgs_history.csv"; histories[method]=pd.read_csv(hp)
        cp=sd/f"nlse_{method}_seed{SEED}_eval{max(CHECKPOINT_EVALS):05d}.pth"; ck=torch.load(cp,map_location="cpu",weights_only=False)
        model=build_model(method); state={k:(v.double() if torch.is_tensor(v) and v.dtype==torch.float32 else v) for k,v in ck["model_state_dict"].items()}; model.load_state_dict(state,strict=True)
        pred=predict(model,X,T,exact.shape); ae=np.abs(exact-pred)
        rows.append({"method":method,"method_label":method_label(method),"seed":SEED,"actual_lbfgs_eval":int(ck.get("actual_lbfgs_eval",ck.get("lbfgs_eval",len(histories[method])))),"final_total_loss":float(ck["latest_metrics"]["total_loss"]),"relative_l2_error":relative_l2(exact,pred),"maximum_absolute_error":float(ae.max())})
        manifest[method]={"history_file":str(hp),"number_of_saved_loss_rows":int(len(histories[method])),"contains_every_lbfgs_closure_evaluation":True}
        del model
        if DEVICE.type=="cuda": torch.cuda.empty_cache()
    df=pd.DataFrame(rows); df.to_csv(ANALYSIS_DIR/"final_metrics_seed99.csv",index=False); (ANALYSIS_DIR/"loss_history_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(7,5))
    for m in METHODS: ax.plot(histories[m]["lbfgs_eval"],histories[m]["total_loss"],label=method_label(m),linewidth=1.5)
    ax.set_yscale("log"); ax.set_xlabel("L-BFGS function evaluation"); ax.set_ylabel("Total loss"); ax.grid(True,which="both",alpha=.35); ax.legend(); fig.tight_layout(); fig.savefig(ANALYSIS_DIR/"total_loss_seed99.png",dpi=800,bbox_inches="tight"); plt.show(); plt.close(fig)
    print(df.to_string(index=False)); print(f"\nAnalysis saved to: {ANALYSIS_DIR.resolve()}")
if __name__=="__main__": main()
