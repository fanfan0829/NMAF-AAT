import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wave_common_fixed_points_adam_lbfgs import CHECKPOINT_SPECS,IMAGE_DPI,RESULT_ROOT
OUTPUT_DIR=RESULT_ROOT/'tanh_vs_nmaf_aat_comparison'
METHODS={'tanh':{'label':'Standard PINN','analysis_csv':'tanh_total_loss_log_statistics.csv','color':'#1f77b4','linestyle':'-','zorder':5},'nmaf_aat':{'label':'NMAF-AAT','analysis_csv':'nmaf_aat_total_loss_log_statistics.csv','color':'#2ca02c','linestyle':'--','zorder':6}}
def config(): plt.rcParams['mathtext.fontset']='custom'; plt.rcParams['mathtext.rm']='Times New Roman'; plt.rcParams['mathtext.it']='Times New Roman:italic'; plt.rcParams['font.family']='Times New Roman'; plt.rcParams.update({'font.size':12})
def ms(vals):
    vals=pd.to_numeric(vals,errors='coerce').dropna()
    if len(vals)==0: return np.nan,np.nan,0
    if len(vals)==1: return float(vals.iloc[0]),np.nan,1
    return float(vals.mean()),float(vals.std(ddof=1)),int(len(vals))
def fmt(m,s): return 'NA' if pd.isna(m) else (f'{m:.6e}' if pd.isna(s) else f'{m:.6e} ± {s:.6e}')
def plot_loss():
    config(); OUTPUT_DIR.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(8,5)); exported=[]
    for key,info in METHODS.items():
        p=RESULT_ROOT/key/'analysis'/info['analysis_csv']
        if not p.exists(): raise FileNotFoundError(f'Missing loss statistics: {p}. Run analysis first.')
        st=pd.read_csv(p); tmp=st.copy(); tmp.insert(0,'method',key); tmp.insert(1,'method_label',info['label']); exported.append(tmp)
        ax.semilogy(st['global_iteration'],st['geometric_mean_total_loss'],linewidth=1.8,color=info['color'],linestyle=info['linestyle'],label=info['label'],zorder=info['zorder'])
        ax.fill_between(st['global_iteration'],st['lower_band'],st['upper_band'],color=info['color'],alpha=.14,linewidth=0,zorder=info['zorder']-1)
    pd.concat(exported,ignore_index=True).to_csv(OUTPUT_DIR/'wave_tanh_vs_nmaf_aat_total_loss_data.csv',index=False)
    for spec in CHECKPOINT_SPECS: ax.axvline(spec['total_iteration'],color='0.35',linestyle='-.',linewidth=.9,alpha=.65)
    ax.set_xlabel('Training Iterations'); ax.set_ylabel('Total Loss'); ax.grid(True,which='both',linestyle='-',alpha=.5); ax.legend(frameon=True); fig.tight_layout(); f=OUTPUT_DIR/'wave_tanh_vs_nmaf_aat_total_loss.jpg'; fig.savefig(f,dpi=IMAGE_DPI,format='jpg',bbox_inches='tight'); plt.show(); plt.close(fig); return f
def compare_stats():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); ind=[]
    for key,info in METHODS.items():
        p=RESULT_ROOT/key/'analysis'/'all_individual_runs.csv'
        if not p.exists(): raise FileNotFoundError(f'Missing individual results: {p}. Run analysis first.')
        d=pd.read_csv(p); d['method_key']=key; d['method_label']=info['label']; ind.append(d)
    individual=pd.concat(ind,ignore_index=True).sort_values(['total_iteration','method_key','seed']).reset_index(drop=True); individual.to_csv(OUTPUT_DIR/'wave_tanh_vs_nmaf_aat_individual_results.csv',index=False)
    rows=[]
    for (key,it),g in individual.groupby(['method_key','total_iteration']):
        row={'method_key':key,'method_label':METHODS[key]['label'],'phase':str(g['phase'].iloc[0]),'total_iteration':int(it),'number_of_seeds':int(g['seed'].nunique())}
        for metric in ['relative_l2_error','maximum_absolute_error','total_loss']:
            m,s,n=ms(g[metric]); row[f'{metric}_mean']=m; row[f'{metric}_sample_std']=s; row[f'{metric}_valid_n']=n
        rows.append(row)
    summary=pd.DataFrame(rows); order={k:i for i,k in enumerate(METHODS)}; summary['method_order']=summary['method_key'].map(order); summary=summary.sort_values(['total_iteration','method_order']).drop(columns=['method_order']).reset_index(drop=True); summary.to_csv(OUTPUT_DIR/'wave_tanh_vs_nmaf_aat_error_summary.csv',index=False)
    pr=summary[['method_label','phase','total_iteration','number_of_seeds']].copy()
    for metric in ['relative_l2_error','maximum_absolute_error','total_loss']:
        pr[metric]=[fmt(m,s) for m,s in zip(summary[f'{metric}_mean'],summary[f'{metric}_sample_std'])]
    pr.to_csv(OUTPUT_DIR/'wave_tanh_vs_nmaf_aat_paper_ready_table.csv',index=False); return individual,summary,pr
if __name__=='__main__':
    fig=plot_loss(); individual,summary,pr=compare_stats(); print('='*116); print('Wave equation Standard PINN vs NMAF-AAT comparison completed.'); print(f'Output directory: {OUTPUT_DIR.resolve()}'); print(f'Loss figure: {fig}'); print('='*116); print('\nPaper-ready table:'); print(pr.to_string(index=False))
