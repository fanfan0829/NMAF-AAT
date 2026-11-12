import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from wave_fixed_activation_seed99_common import *
METHODS=['gelu','mish','xtanh']; SEED=99; N_TEST=200; ANALYSIS_DIR=RESULT_ROOT/'three_activation_analysis_seed99'; PLOT_MAX_POINTS=20000

def relative_l2(exact,pred):
    exact=np.asarray(exact,dtype=np.float64).reshape(-1); pred=np.asarray(pred,dtype=np.float64).reshape(-1); return float(np.linalg.norm(exact-pred,ord=2)/np.linalg.norm(exact,ord=2))
@torch.no_grad()
def predict(model,X,T,shape,batch_size=65536):
    xs=X.reshape(-1,1); ts=T.reshape(-1,1); out=[]
    for i in range(0,len(xs),batch_size):
        x=torch.as_tensor(xs[i:i+batch_size],dtype=DTYPE_LBFGS,device=DEVICE); t=torch.as_tensor(ts[i:i+batch_size],dtype=DTYPE_LBFGS,device=DEVICE); out.append(model(x,t).cpu().numpy())
    return np.vstack(out).reshape(shape)

def _plot_view(df):
    if len(df)<=PLOT_MAX_POINTS: return df
    idx=np.unique(np.linspace(0,len(df)-1,PLOT_MAX_POINTS,dtype=int)); return df.iloc[idx]

def main():
    ANALYSIS_DIR.mkdir(parents=True,exist_ok=True)
    x=np.linspace(0.,1.,N_TEST); t=np.linspace(0.,1.,N_TEST); X,T=np.meshgrid(x,t,indexing='ij'); exact=np.sin(2*np.pi*X)*np.sin(2*np.pi*T)+np.sin(10*np.pi*X)*np.sin(10*np.pi*T)
    rows=[]; histories={}; manifest={}
    final_name=CHECKPOINT_SPECS[-1]['name']
    for method in METHODS:
        sd=RESULT_ROOT/method/f'seed_{SEED}'; hp=sd/'training_history.csv'; histories[method]=pd.read_csv(hp)
        cp=checkpoint_path(sd,method,SEED,final_name); ck=torch.load(cp,map_location='cpu',weights_only=False)
        model=build_model(method,DTYPE_LBFGS); state=state_dict_to_dtype(ck['model_state_dict'],DTYPE_LBFGS); model.load_state_dict(state,strict=True)
        pred=predict(model,X,T,exact.shape); ae=np.abs(exact-pred)
        rows.append({'method':method,'method_label':method_label(method),'seed':SEED,'adam_iterations':int(ck.get('adam_iteration',ADAM_EPOCHS)),'lbfgs_evals':int(ck.get('lbfgs_eval',0)),'final_global_iteration':int(ck.get('total_iteration',ADAM_EPOCHS+int(ck.get('lbfgs_eval',0)))),'final_total_loss':float(ck['latest_metrics']['total_loss']),'relative_l2_error':relative_l2(exact,pred),'maximum_absolute_error':float(ae.max())})
        manifest[method]={'history_file':str(hp),'number_of_saved_loss_rows':int(len(histories[method])),'contains_every_adam_step_and_every_lbfgs_closure_evaluation':True,'plot_is_downsampled_only_for_display':len(histories[method])>PLOT_MAX_POINTS}
        del model
        if DEVICE.type=='cuda': torch.cuda.empty_cache()
    df=pd.DataFrame(rows); df.to_csv(ANALYSIS_DIR/'final_metrics_seed99.csv',index=False); (ANALYSIS_DIR/'loss_history_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,5))
    for m in METHODS:
        d=_plot_view(histories[m]); ax.plot(d['global_iteration'],d['total_loss'],label=method_label(m),linewidth=1.35)
    ax.set_yscale('log'); ax.set_xlabel('Global iteration / L-BFGS evaluation sequence'); ax.set_ylabel('Total loss'); ax.grid(True,which='both',alpha=.35); ax.legend(); fig.tight_layout(); fig.savefig(ANALYSIS_DIR/'total_loss_seed99.png',dpi=800,bbox_inches='tight'); plt.show(); plt.close(fig)
    print(df.to_string(index=False)); print(f'\nAnalysis saved to: {ANALYSIS_DIR.resolve()}')
if __name__=='__main__': main()
