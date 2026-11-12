import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from wave_common_fixed_points_adam_lbfgs import (
    SEEDS, REPRESENTATIVE_SEED, RESULT_ROOT, DEVICE, DTYPE_LBFGS,
    HIDDEN_LAYERS, HIDDEN_NEURONS, ADAM_EPOCHS, LBFGS_MAX_EVAL,
    TOTAL_TRAINING_STEPS, IMAGE_DPI, TanhPINN, NMAFAATPINN, exact_solution,
)
from train_wave_rar_integrated_adam500k_lbfgs200k import (
    METHOD as RAR_METHOD, METHOD_LABEL as RAR_LABEL, METHOD_DIR as RAR_DIR,
    RAR_SELECTION_ADAM_STEP, RAR_BASE_N_RES, RAR_ADD_N_RES, RAR_FINAL_N_RES,
    point_path,
)

N_TEST=200
OUTPUT_DIR=RESULT_ROOT/'tanh_vs_rar_integrated_vs_nmaf_aat'
METHODS={
    'tanh': dict(label='Standard PINN',dir=RESULT_ROOT/'tanh',prefix='wave_tanh',style='-'),
    RAR_METHOD: dict(label=RAR_LABEL,dir=RAR_DIR,prefix=f'wave_{RAR_METHOD}',style='-.'),
    'nmaf_aat': dict(label='NMAF-AAT',dir=RESULT_ROOT/'nmaf_aat',prefix='wave_nmaf_aat',style='--'),
}
STAGES=[
    dict(name='adam0500000',phase='adam',target_total=500000,target_adam=500000,target_lb=0),
    dict(name='total0700000',phase='adam_lbfgs',target_total=700000,target_adam=500000,target_lb=200000),
]


def style():
    plt.rcParams['mathtext.fontset']='custom'; plt.rcParams['mathtext.rm']='Times New Roman'; plt.rcParams['mathtext.it']='Times New Roman:italic'; plt.rcParams['font.family']='Times New Roman'
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'lines.linewidth':1.7})


def build_model(key):
    if key in ('tanh',RAR_METHOD):
        return TanhPINN(HIDDEN_LAYERS,HIDDEN_NEURONS).to(dtype=DTYPE_LBFGS,device=DEVICE)
    return NMAFAATPINN(HIDDEN_LAYERS,HIDDEN_NEURONS,dtype=DTYPE_LBFGS).to(dtype=DTYPE_LBFGS,device=DEVICE)


def ckpt_path(key,seed,name):
    info=METHODS[key]
    return info['dir']/f'seed_{seed}'/f'{info["prefix"]}_seed{seed}_{name}.pth'


def load_ckpt(key,seed,name):
    p=ckpt_path(key,seed,name)
    if not p.exists(): raise FileNotFoundError(p)
    ck=torch.load(p,map_location='cpu',weights_only=False)
    sd={k:(v.to(dtype=DTYPE_LBFGS) if torch.is_tensor(v) else v) for k,v in ck['model_state_dict'].items()}
    return ck,sd


def eval_grid():
    x=torch.linspace(0,1,N_TEST,dtype=DTYPE_LBFGS,device=DEVICE).reshape(-1,1)
    t=torch.linspace(0,1,N_TEST,dtype=DTYPE_LBFGS,device=DEVICE).reshape(-1,1)
    X,T=torch.meshgrid(x.squeeze(),t.squeeze(),indexing='ij')
    xf=X.reshape(-1,1); tf=T.reshape(-1,1)
    with torch.no_grad(): ue=exact_solution(xf,tf).detach().cpu().numpy().reshape(N_TEST,N_TEST)
    return x.cpu().numpy().reshape(-1),t.cpu().numpy().reshape(-1),X.cpu().numpy(),T.cpu().numpy(),xf,tf,ue


def predict(model):
    x,t,X,T,xf,tf,ue=eval_grid(); model.eval(); parts=[]
    with torch.no_grad():
        for s in range(0,len(xf),20000): parts.append(model(xf[s:s+20000],tf[s:s+20000]).detach().cpu().numpy())
    up=np.vstack(parts).reshape(N_TEST,N_TEST)
    return x,t,X,T,ue,up


def rel_l2(ue,up):
    return float(np.linalg.norm((ue-up).reshape(-1),2)/np.linalg.norm(ue.reshape(-1),2))


def rebuild_predictions():
    rows=[]
    for key,info in METHODS.items():
        mrows=[]
        for seed in SEEDS:
            sdir=info['dir']/f'seed_{seed}'
            for st in STAGES:
                ck,sd=load_ckpt(key,seed,st['name']); model=build_model(key); model.load_state_dict(sd,strict=True)
                x,t,X,T,ue,up=predict(model); ae=np.abs(ue-up); latest=ck.get('latest_metrics',{})
                pred_name=f'{info["prefix"]}_seed{seed}_{st["name"]}_prediction.npz'
                np.savez_compressed(sdir/pred_name,x=x,t=t,X=X,T=T,u_exact=ue,u_pred=up,abs_error=ae)
                row={
                    'method':key,'method_label':info['label'],'seed':int(seed),'phase':st['phase'],'checkpoint_name':st['name'],
                    'total_iteration':int(st['target_total']),
                    'actual_total_iteration':int(ck.get('total_iteration',st['target_total'])),
                    'adam_iteration':int(ck.get('adam_iteration',st['target_adam'])),
                    'lbfgs_eval':int(st['target_lb']),'actual_lbfgs_eval':int(ck.get('lbfgs_eval',st['target_lb'])),
                    'total_loss':float(latest['total_loss']),
                    'optimization_total_loss':float(latest.get('optimization_total_loss',np.nan)) if key==RAR_METHOD else np.nan,
                    'relative_l2_error':rel_l2(ue,up),'maximum_absolute_error':float(ae.max()),
                    'u_exact_min':float(ue.min()),'u_exact_max':float(ue.max()),'u_pred_min':float(up.min()),'u_pred_max':float(up.max()),
                    'elapsed_seconds':float(ck.get('elapsed_seconds',np.nan)),'prediction_file':pred_name,
                }
                rows.append(row); mrows.append(row)
        adir=info['dir']/'analysis'; adir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(mrows).sort_values(['total_iteration','seed']).to_csv(adir/'all_individual_runs.csv',index=False)
    return pd.DataFrame(rows)


def verify_grid():
    ref=None; rows=[]
    for key,info in METHODS.items():
        for seed in SEEDS:
            for st in STAGES:
                p=info['dir']/f'seed_{seed}'/f'{info["prefix"]}_seed{seed}_{st["name"]}_prediction.npz'
                d=np.load(p); cur=(np.asarray(d['x']),np.asarray(d['t']),np.asarray(d['u_exact']))
                if ref is None: ref=tuple(a.copy() for a in cur)
                if not np.array_equal(cur[0],ref[0]): raise RuntimeError(f'x-grid mismatch: {p}')
                if not np.array_equal(cur[1],ref[1]): raise RuntimeError(f't-grid mismatch: {p}')
                if not np.array_equal(cur[2],ref[2]): raise RuntimeError(f'u_exact mismatch: {p}')
                rows.append({'method':key,'seed':seed,'checkpoint':st['name'],'identical_x':True,'identical_t':True,'identical_u_exact':True})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/'verified_identical_l2_evaluation_grid.csv',index=False)


def loss_stats(key):
    merged=None
    for seed in SEEDS:
        p=METHODS[key]['dir']/f'seed_{seed}'/'training_history.csv'
        if not p.exists(): raise FileNotFoundError(p)
        h=pd.read_csv(p)[['global_iteration','total_loss']]
        h=h[h['global_iteration']<=TOTAL_TRAINING_STEPS].drop_duplicates('global_iteration',keep='last').rename(columns={'total_loss':f'total_loss_seed_{seed}'})
        merged=h if merged is None else merged.merge(h,on='global_iteration',how='inner')
    cols=[f'total_loss_seed_{s}' for s in SEEDS]; vals=merged[cols].to_numpy(float)
    if np.any(vals<=0) or np.any(~np.isfinite(vals)): raise ValueError(f'Invalid loss for {key}')
    logs=np.log10(vals); mu=logs.mean(1); sd=logs.std(1,ddof=1)
    out=pd.DataFrame({'global_iteration':merged['global_iteration'].to_numpy(int),'geometric_mean_total_loss':10**mu,'log10_sample_std':sd,'lower_band':10**(mu-sd),'upper_band':10**(mu+sd)})
    for s in SEEDS: out[f'total_loss_seed_{s}']=merged[f'total_loss_seed_{s}'].to_numpy(float)
    adir=METHODS[key]['dir']/'analysis'; adir.mkdir(parents=True,exist_ok=True); out.to_csv(adir/f'{key}_total_loss_log_statistics.csv',index=False)
    return out


def plot_loss(stats):
    fig,ax=plt.subplots(figsize=(8.8,5.4)); exports=[]
    for key,info in METHODS.items():
        st=stats[key]; ex=st.copy(); ex.insert(0,'method_key',key); ex.insert(1,'method_label',info['label']); exports.append(ex)
        line,=ax.plot(st['global_iteration'],st['geometric_mean_total_loss'],linestyle=info['style'],label=info['label'])
        ax.fill_between(st['global_iteration'],st['lower_band'],st['upper_band'],color=line.get_color(),alpha=.14,linewidth=0)
    pd.concat(exports,ignore_index=True).to_csv(OUTPUT_DIR/'wave_three_method_common_loss_data.csv',index=False)
    ax.axvline(RAR_SELECTION_ADAM_STEP,linestyle=':',linewidth=1,alpha=.65,label='RAR acquisition')
    ax.axvline(ADAM_EPOCHS,linestyle='-.',linewidth=1,alpha=.65,label='Adam to L-BFGS')
    ax.set_yscale('log'); ax.set_xlabel('Training Iterations'); ax.set_ylabel('Total Loss on Original Fixed Points'); ax.set_xlim(1,TOTAL_TRAINING_STEPS)
    ax.grid(True,which='both',linestyle='-',alpha=.5); ax.legend(frameon=True); fig.tight_layout()
    p=OUTPUT_DIR/'wave_tanh_rar_integrated_nmaf_aat_total_loss.jpg'; fig.savefig(p,dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig); return p


def ms(v):
    v=pd.to_numeric(v,errors='coerce').dropna()
    return (float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else np.nan,len(v))


def fmt(m,s): return f'{m:.6e}' if pd.isna(s) else f'{m:.6e} ± {s:.6e}'


def accuracy_tables(df):
    metrics=['total_loss','relative_l2_error','maximum_absolute_error','elapsed_seconds']; rows=[]
    for (key,it),g in df.groupby(['method','total_iteration']):
        r={'method_key':key,'method_label':METHODS[key]['label'],'total_iteration':int(it),'number_of_seeds':int(g['seed'].nunique())}
        for m in metrics:
            mean,std,n=ms(g[m]); r[f'{m}_mean']=mean; r[f'{m}_sample_std']=std; r[f'{m}_valid_n']=n
        rows.append(r)
    out=pd.DataFrame(rows); order={'tanh':0,RAR_METHOD:1,'nmaf_aat':2}; out['_o']=out['method_key'].map(order); out=out.sort_values(['total_iteration','_o']).drop(columns='_o').reset_index(drop=True)
    out.to_csv(OUTPUT_DIR/'wave_three_method_error_summary.csv',index=False)
    paper=out[['method_label','total_iteration','number_of_seeds']].copy()
    for m in metrics: paper[m]=[fmt(a,b) for a,b in zip(out[f'{m}_mean'],out[f'{m}_sample_std'])]
    paper.to_csv(OUTPUT_DIR/'wave_three_method_paper_ready_table.csv',index=False)
    return out,paper


def plot_rar_points():
    p=point_path(REPRESENTATIVE_SEED)
    if not p.exists(): raise FileNotFoundError(p)
    d=np.load(p); x=d['x_res'].reshape(-1); t=d['t_res'].reshape(-1); src=d['source'].reshape(-1); a=src==0; b=src==1
    assert a.sum()==RAR_BASE_N_RES and b.sum()==RAR_ADD_N_RES and len(x)==RAR_FINAL_N_RES
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(t[a],x[a],s=4,alpha=.22,label=f'Original shared ({RAR_BASE_N_RES})'); ax.scatter(t[b],x[b],s=7,alpha=.75,label=f'RAR added ({RAR_ADD_N_RES})')
    ax.set_xlabel('t'); ax.set_ylabel('x'); ax.set_title('Wave RAR Final PDE Points (6400 + 1280)'); ax.grid(True,alpha=.3); ax.legend(frameon=True); fig.tight_layout()
    adir=RAR_DIR/'analysis'; adir.mkdir(parents=True,exist_ok=True); fig.savefig(adir/'wave_rar_seed99_base6400_plus1280_points.jpg',dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig)


def plot_rep_fields():
    seed=REPRESENTATIVE_SEED
    for key,info in METHODS.items():
        for st in STAGES:
            p=info['dir']/f'seed_{seed}'/f'{info["prefix"]}_seed{seed}_{st["name"]}_prediction.npz'; d=np.load(p)
            X,T,ue,up,ae=d['X'],d['T'],d['u_exact'],d['u_pred'],d['abs_error']; mn=float(ue.min()); mx=float(ue.max())
            fig,axes=plt.subplots(1,3,figsize=(18,5)); ims=[axes[0].contourf(X,T,ue,cmap='RdBu_r',levels=50,vmin=mn,vmax=mx),axes[1].contourf(X,T,up,cmap='RdBu_r',levels=50,vmin=mn,vmax=mx),axes[2].contourf(X,T,ae,cmap='Reds',levels=50)]
            for ax,im,title in zip(axes,ims,['Exact Solution',f'{info["label"]} Prediction','Absolute Error']): ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('t'); fig.colorbar(im,ax=ax)
            fig.tight_layout(); adir=info['dir']/'analysis'; adir.mkdir(parents=True,exist_ok=True); fig.savefig(adir/f'wave_{key}_seed{seed}_{st["name"]}_field.jpg',dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig)


def rar_cost():
    rows=[]
    for seed in SEEDS:
        p=RAR_DIR/f'seed_{seed}'/'training_summary.json'; d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'seed':seed,'rar_selection_seconds':d.get('rar_selection_seconds',np.nan),'end_to_end_training_seconds':d['total_time_seconds'],'final_n_res':d['final_n_res']})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/'wave_rar_cost_individual.csv',index=False)


if __name__=='__main__':
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); style()
    individual=rebuild_predictions(); individual.to_csv(OUTPUT_DIR/'wave_three_method_individual_results.csv',index=False)
    verify_grid(); stats={k:loss_stats(k) for k in METHODS}; loss_path=plot_loss(stats); summary,paper=accuracy_tables(individual); plot_rar_points(); plot_rep_fields(); rar_cost()
    print('='*124); print('Wave Standard PINN vs integrated-RAR PINN vs NMAF-AAT'); print('='*124)
    print('RAR: original6400 through Adam100000, +1280 once, retain same model+Adam optimizer through Adam500000, then L-BFGS200000.')
    print('Primary loss for all methods: original shared 6400-point evaluation.'); print('Accuracy: identical analytical 200x200 grid hard-checked.'); print('Loss figure:',loss_path); print('-'*124); print(paper.to_string(index=False)); print('='*124)
