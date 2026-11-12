import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from wave_common_fixed_points_adam_lbfgs import CHECKPOINT_SPECS,DEVICE,DTYPE_LBFGS,HIDDEN_LAYERS,HIDDEN_NEURONS,IMAGE_DPI,INITIAL_ALPHA,INITIAL_BETA,NMAFAATPINN,REPRESENTATIVE_SEED,RESULT_ROOT,SEEDS,TanhPINN,adaptive_parameter_summary,exact_solution,method_label
METHOD='nmaf_aat'; METHOD_LABEL=method_label(METHOD); METHOD_DIR=RESULT_ROOT/METHOD; ANALYSIS_DIR=METHOD_DIR/'analysis'; N_TEST=200; N_X_TIMEWISE=1000; N_T_TIMEWISE=50

def configure_plot_style():
    plt.rcParams['mathtext.fontset']='custom'; plt.rcParams['mathtext.rm']='Times New Roman'; plt.rcParams['mathtext.it']='Times New Roman:italic'; plt.rcParams['font.family']='Times New Roman'; plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'lines.linewidth':1.6})

def save_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def build_model():
    if METHOD=='tanh': return TanhPINN(HIDDEN_LAYERS,HIDDEN_NEURONS).to(dtype=DTYPE_LBFGS,device=DEVICE)
    if METHOD=='nmaf_aat': return NMAFAATPINN(HIDDEN_LAYERS,HIDDEN_NEURONS,dtype=DTYPE_LBFGS).to(dtype=DTYPE_LBFGS,device=DEVICE)
    raise ValueError(METHOD)
def ckpt_path(seed,name): return METHOD_DIR/f'seed_{seed}'/f'wave_{METHOD}_seed{seed}_{name}.pth'
def load_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=False); sd={k:(v.to(dtype=DTYPE_LBFGS) if torch.is_tensor(v) else v) for k,v in ck['model_state_dict'].items()}
    latest=ck.get('latest_metrics',{}); aat=ck.get('adaptive_parameter_summary',{})
    msg=f"Loaded checkpoint: {path} | phase={ck.get('phase','NA')} | total_iteration={ck.get('total_iteration','NA')} | adam_iteration={ck.get('adam_iteration','NA')} | lbfgs_eval={ck.get('lbfgs_eval','NA')} | checkpoint_total_loss={latest.get('total_loss','NA')}"
    if METHOD=='nmaf_aat': msg+=f" | alpha_mean={aat.get('alpha_mean','NA')} | beta_mean={aat.get('beta_mean','NA')}"
    print(msg); return ck,sd

def predict_full_field(model,n_test=N_TEST):
    x=torch.linspace(0,1,n_test,dtype=DTYPE_LBFGS,device=DEVICE).reshape(-1,1); t=torch.linspace(0,1,n_test,dtype=DTYPE_LBFGS,device=DEVICE).reshape(-1,1)
    X,T=torch.meshgrid(x.squeeze(),t.squeeze(),indexing='ij'); xf=X.reshape(-1,1); tf=T.reshape(-1,1)
    model.eval(); parts=[]
    with torch.no_grad():
        for s in range(0,xf.shape[0],20000): parts.append(model(xf[s:s+20000],tf[s:s+20000]).detach().cpu().numpy())
        ue=exact_solution(xf,tf).detach().cpu().numpy().reshape(n_test,n_test)
    up=np.vstack(parts).reshape(n_test,n_test)
    return x.cpu().numpy().reshape(-1),t.cpu().numpy().reshape(-1),X.cpu().numpy(),T.cpu().numpy(),ue,up

def relative_l2(u_exact,u_pred):
    exact_flat=np.asarray(u_exact).reshape(-1); pred_flat=np.asarray(u_pred).reshape(-1)
    return float(np.linalg.norm(exact_flat-pred_flat,2)/np.linalg.norm(exact_flat,2))

def timewise_l2_errors(model,n_x=N_X_TIMEWISE,n_t=N_T_TIMEWISE):
    x=torch.linspace(0,1,n_x,dtype=DTYPE_LBFGS,device=DEVICE).reshape(-1,1); t_values=np.linspace(0,1,n_t); rows=[]; model.eval()
    with torch.no_grad():
        for tv in t_values:
            tt=torch.full_like(x,float(tv)); up=model(x,tt); ue=exact_solution(x,tt)
            val=torch.norm((ue-up).reshape(-1),p=2)/torch.norm(ue.reshape(-1),p=2); rows.append({'t':float(tv),'relative_l2_error':float(val.detach().cpu())})
    return pd.DataFrame(rows)

def ms(vals):
    vals=pd.to_numeric(vals,errors='coerce').dropna()
    if len(vals)==0: return np.nan,np.nan,0
    if len(vals)==1: return float(vals.iloc[0]),np.nan,1
    return float(vals.mean()),float(vals.std(ddof=1)),int(len(vals))
def fmt(m,s): return 'NA' if pd.isna(m) else (f'{m:.6e}' if pd.isna(s) else f'{m:.6e} ± {s:.6e}')

def rebuild_seed_summary(seed):
    seed_dir=METHOD_DIR/f'seed_{seed}'; rows=[]
    for spec in CHECKPOINT_SPECS:
        name=spec['name']; ck,sd=load_checkpoint(ckpt_path(seed,name)); model=build_model(); model.load_state_dict(sd,strict=True)
        x,t,X,T,ue,up=predict_full_field(model); ae=np.abs(ue-up)
        pred_name=f'wave_{METHOD}_seed{seed}_{name}_prediction.npz'; np.savez_compressed(seed_dir/pred_name,x=x,t=t,X=X,T=T,u_exact=ue,u_pred=up,abs_error=ae)
        tw=timewise_l2_errors(model); tw_name=f'wave_{METHOD}_seed{seed}_{name}_timewise_l2.csv'; tw.to_csv(seed_dir/tw_name,index=False)
        latest=ck.get('latest_metrics',{})
        row={'method':METHOD,'method_label':METHOD_LABEL,'seed':int(seed),'checkpoint_name':name,'phase':ck.get('phase',spec['phase']),'total_iteration':int(ck.get('total_iteration',spec['total_iteration'])),'adam_iteration':int(ck.get('adam_iteration',spec['adam_iteration'])),'lbfgs_eval':int(ck.get('lbfgs_eval',spec['lbfgs_eval'])),'total_loss':float(latest.get('total_loss',np.nan)),'residual_loss':float(latest.get('residual_loss',np.nan)),'data_loss':float(latest.get('data_loss',np.nan)),'bc_loss':float(latest.get('bc_loss',np.nan)),'ic1_loss':float(latest.get('ic1_loss',np.nan)),'ic2_loss':float(latest.get('ic2_loss',np.nan)),'relative_l2_error':relative_l2(ue,up),'maximum_absolute_error':float(ae.max()),'u_exact_min':float(ue.min()),'u_exact_max':float(ue.max()),'u_pred_min':float(up.min()),'u_pred_max':float(up.max()),'elapsed_seconds':float(ck.get('elapsed_seconds',np.nan)),'prediction_file':pred_name,'timewise_l2_file':tw_name}
        if METHOD=='nmaf_aat': row.update(adaptive_parameter_summary(model))
        save_json(seed_dir/f'metrics_{name}.json',row); rows.append(row)
    pd.DataFrame(rows).to_csv(seed_dir/'seed_summary.csv',index=False)

def rebuild_all():
    for seed in SEEDS: rebuild_seed_summary(seed)
def combine():
    df=pd.concat([pd.read_csv(METHOD_DIR/f'seed_{s}'/'seed_summary.csv') for s in SEEDS],ignore_index=True).sort_values(['total_iteration','seed']).reset_index(drop=True)
    df.to_csv(ANALYSIS_DIR/'all_individual_runs.csv',index=False); return df

def stats_table(df):
    metrics=['total_loss','residual_loss','data_loss','relative_l2_error','maximum_absolute_error','elapsed_seconds']+(['delta_alpha_l2','delta_beta_l2'] if METHOD=='nmaf_aat' else [])
    rows=[]
    for it,g in df.groupby('total_iteration'):
        row={'method':METHOD,'method_label':METHOD_LABEL,'phase':str(g['phase'].iloc[0]),'checkpoint_name':str(g['checkpoint_name'].iloc[0]),'total_iteration':int(it),'number_of_seeds':int(g['seed'].nunique())}
        for metric in metrics:
            m,s,n=ms(g[metric]); row[f'{metric}_mean']=m; row[f'{metric}_sample_std']=s; row[f'{metric}_valid_n']=n
        rows.append(row)
    out=pd.DataFrame(rows); out.to_csv(ANALYSIS_DIR/'mean_std_statistics.csv',index=False)
    pr=[]
    for _,r in out.iterrows():
        row={'Method':r['method_label'],'Phase':r['phase'],'Total iteration':int(r['total_iteration']),'Seeds':int(r['number_of_seeds'])}
        for metric in metrics: row[metric]=fmt(r[f'{metric}_mean'],r[f'{metric}_sample_std'])
        pr.append(row)
    pd.DataFrame(pr).to_csv(ANALYSIS_DIR/'paper_ready_mean_std_table.csv',index=False); return out

def print_tables(df,st):
    cols=['seed','phase','total_iteration','adam_iteration','lbfgs_eval','total_loss','relative_l2_error','maximum_absolute_error','u_exact_min','u_exact_max','u_pred_min','u_pred_max']+(['delta_alpha_l2','delta_beta_l2'] if METHOD=='nmaf_aat' else [])
    print('\n'+'='*116); print(f'{METHOD_LABEL} individual checkpoint accuracy'); print(df[cols].to_string(index=False,float_format=lambda v:f'{v:.6e}')); print('='*116)
    print(f'{METHOD_LABEL} checkpoint accuracy statistics')
    for _,r in st.iterrows():
        line=f"Total iter {int(r['total_iteration']):7d} | Phase = {r['phase']} | L2 = {fmt(r['relative_l2_error_mean'],r['relative_l2_error_sample_std'])} | Max abs error = {fmt(r['maximum_absolute_error_mean'],r['maximum_absolute_error_sample_std'])}"
        if METHOD=='nmaf_aat': line+=f" | ||alpha-1||2 = {fmt(r['delta_alpha_l2_mean'],r['delta_alpha_l2_sample_std'])} | ||beta||2 = {fmt(r['delta_beta_l2_mean'],r['delta_beta_l2_sample_std'])}"
        print(line)
    print('='*116)

def loss_curve():
    merged=None
    for seed in SEEDS:
        h=pd.read_csv(METHOD_DIR/f'seed_{seed}'/'training_history.csv')[['global_iteration','total_loss']].drop_duplicates('global_iteration',keep='last').rename(columns={'total_loss':f'total_loss_seed_{seed}'})
        merged=h if merged is None else merged.merge(h,on='global_iteration',how='inner')
    cols=[f'total_loss_seed_{s}' for s in SEEDS]; vals=merged[cols].to_numpy(float); logs=np.log10(vals); mu=logs.mean(1); sd=logs.std(1,ddof=1)
    out=pd.DataFrame({'global_iteration':merged['global_iteration'].to_numpy(int),'geometric_mean_total_loss':10**mu,'log10_sample_std':sd,'lower_band':10**(mu-sd),'upper_band':10**(mu+sd)})
    for s in SEEDS: out[f'total_loss_seed_{s}']=merged[f'total_loss_seed_{s}'].to_numpy(float)
    out.to_csv(ANALYSIS_DIR/f'{METHOD}_total_loss_log_statistics.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5)); line,=ax.plot(out['global_iteration'],out['geometric_mean_total_loss'],label=METHOD_LABEL,linewidth=1.8); ax.fill_between(out['global_iteration'],out['lower_band'],out['upper_band'],color=line.get_color(),alpha=.16,linewidth=0)
    for spec in CHECKPOINT_SPECS: ax.axvline(spec['total_iteration'],color='0.35',linestyle='-.',linewidth=.9,alpha=.65)
    ax.set_yscale('log'); ax.set_xlabel('Training Iterations'); ax.set_ylabel('Total Loss'); ax.grid(True,which='both',linestyle='-',alpha=.5); ax.legend(frameon=True); fig.tight_layout(); fig.savefig(ANALYSIS_DIR/f'wave_{METHOD}_total_loss.jpg',dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig)

def rep_figs():
    for spec in CHECKPOINT_SPECS:
        name=spec['name']; data=np.load(METHOD_DIR/f'seed_{REPRESENTATIVE_SEED}'/f'wave_{METHOD}_seed{REPRESENTATIVE_SEED}_{name}_prediction.npz')
        X,T,ue,up,ae=data['X'],data['T'],data['u_exact'],data['u_pred'],data['abs_error']; fig,axes=plt.subplots(1,3,figsize=(18,5)); mn=float(ue.min()); mx=float(ue.max())
        ims=[axes[0].contourf(X,T,ue,cmap='RdBu_r',levels=50,vmin=mn,vmax=mx),axes[1].contourf(X,T,up,cmap='RdBu_r',levels=50,vmin=mn,vmax=mx),axes[2].contourf(X,T,ae,cmap='Reds',levels=50)]
        titles=['Exact Solution',f'{METHOD_LABEL} Prediction','Absolute Error']
        for ax,im,title in zip(axes,ims,titles): ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('t'); fig.colorbar(im,ax=ax)
        fig.tight_layout(); fig.savefig(ANALYSIS_DIR/f'wave_{METHOD}_seed{REPRESENTATIVE_SEED}_{name}_field.jpg',dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for spec in CHECKPOINT_SPECS:
        name=spec['name']; tw=pd.read_csv(METHOD_DIR/f'seed_{REPRESENTATIVE_SEED}'/f'wave_{METHOD}_seed{REPRESENTATIVE_SEED}_{name}_timewise_l2.csv'); ax.plot(tw['t'],tw['relative_l2_error'],marker='o',markersize=3,markevery=5,linewidth=1.6,label=f'{METHOD_LABEL}, {name}')
    ax.set_xlabel('Time t'); ax.set_ylabel(r'Relative $L_2$ error'); ax.set_yscale('log'); ax.set_xlim(0,1); ax.grid(True,which='both',linestyle='--',alpha=.6); ax.legend(frameon=True); fig.tight_layout(); fig.savefig(ANALYSIS_DIR/f'wave_{METHOD}_seed{REPRESENTATIVE_SEED}_timewise_l2.jpg',dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig)
    if METHOD=='nmaf_aat':
        for spec in CHECKPOINT_SPECS:
            name=spec['name']; _,sd=load_checkpoint(ckpt_path(REPRESENTATIVE_SEED,name)); a=sd['activation.alpha'].detach().cpu().numpy(); b=sd['activation.beta'].detach().cpu().numpy(); fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].hist(a,bins=20); axes[0].axvline(INITIAL_ALPHA,linestyle='--',linewidth=1.2); axes[0].set_title(r'Learned $\alpha$'); axes[1].hist(b,bins=20); axes[1].axvline(INITIAL_BETA,linestyle='--',linewidth=1.2); axes[1].set_title(r'Learned $\beta$'); fig.tight_layout(); fig.savefig(ANALYSIS_DIR/f'wave_{METHOD}_seed{REPRESENTATIVE_SEED}_{name}_aat_hist.jpg',dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig)

if __name__=='__main__':
    ANALYSIS_DIR.mkdir(parents=True,exist_ok=True); configure_plot_style(); rebuild_all(); df=combine(); st=stats_table(df); print_tables(df,st); loss_curve(); rep_figs(); print('\n'+'='*116); print(f'{METHOD_LABEL} analysis completed.'); print(f'Output directory: {ANALYSIS_DIR.resolve()}'); print('='*116)
