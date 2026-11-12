import os, csv, json, random, time
from pathlib import Path
import numpy as np
import scipy.io
import torch
import torch.nn as nn

SEEDS=[99]
REPRESENTATIVE_SEED=99
RESULT_ROOT=Path('wave_activation_ablation_seed99_results')
FORMAL_RESULT_ROOT=Path('wave_multiseed_results')
POINTS_DIR=FORMAL_RESULT_ROOT/'fixed_training_points'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE_ADAM=torch.float32
DTYPE_LBFGS=torch.float64
HIDDEN_LAYERS=4
HIDDEN_NEURONS=128
N_RES=6400
N_BC_IC=512
N_BC=N_BC_IC
N_IC1=N_BC_IC//2
N_IC2=N_BC_IC//2
ADAM_EPOCHS=500000
ADAM_LR=1e-4
LBFGS_MAX_ITER=200000
LBFGS_MAX_EVAL=200000
LBFGS_LR=1.0
LBFGS_HISTORY_SIZE=50
LBFGS_TOLERANCE_GRAD=1e-9
LBFGS_TOLERANCE_CHANGE=1e-11
LBFGS_LINE_SEARCH_FN='strong_wolfe'
TOTAL_TRAINING_STEPS=ADAM_EPOCHS+LBFGS_MAX_EVAL
CHECKPOINT_SPECS=(
    {'name':'adam0500000','phase':'adam','total_iteration':ADAM_EPOCHS,'adam_iteration':ADAM_EPOCHS,'lbfgs_eval':0},
    {'name':'total0700000','phase':'adam_lbfgs','total_iteration':TOTAL_TRAINING_STEPS,'adam_iteration':ADAM_EPOCHS,'lbfgs_eval':LBFGS_MAX_EVAL},
)
PRINT_EVERY_ADAM=1000
PRINT_EVERY_LBFGS=100
IMAGE_DPI=800
INITIAL_ALPHA=1.0
INITIAL_BETA=0.0
STAN_INITIAL_BETA=1.0


def set_seed(seed:int):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def sync():
    if DEVICE.type=='cuda': torch.cuda.synchronize()

def exact_solution(x,t):
    return torch.sin(2*np.pi*x)*torch.sin(2*np.pi*t)+torch.sin(10*np.pi*x)*torch.sin(10*np.pi*t)

class TanhPINN(nn.Module):
    def __init__(self,hidden_layers=HIDDEN_LAYERS,hidden_neurons=HIDDEN_NEURONS):
        super().__init__(); self.input_layer=nn.Linear(2,hidden_neurons)
        self.hidden_layers=nn.ModuleList([nn.Linear(hidden_neurons,hidden_neurons) for _ in range(hidden_layers)])
        self.output_layer=nn.Linear(hidden_neurons,1); self.activation=nn.Tanh()
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1); out=self.activation(self.input_layer(z))
        for layer in self.hidden_layers: out=self.activation(layer(out))
        return self.output_layer(out)

class AdaptiveActivation(nn.Module):
    def __init__(self,in_features,dtype,initial_alpha=INITIAL_ALPHA,initial_beta=INITIAL_BETA):
        super().__init__(); self.alpha=nn.Parameter(torch.ones(in_features,dtype=dtype)*initial_alpha); self.beta=nn.Parameter(torch.ones(in_features,dtype=dtype)*initial_beta)
    def forward(self,x):
        X=self.alpha*x+self.beta
        return -X*(torch.exp(-0.5*X**2)-torch.tanh(X))

class NMAFAATPINN(nn.Module):
    def __init__(self,hidden_layers=HIDDEN_LAYERS,hidden_neurons=HIDDEN_NEURONS,dtype=DTYPE_ADAM):
        super().__init__(); self.input_layer=nn.Linear(2,hidden_neurons)
        self.hidden_layers=nn.ModuleList([nn.Linear(hidden_neurons,hidden_neurons) for _ in range(hidden_layers)])
        self.output_layer=nn.Linear(hidden_neurons,1); self.activation=AdaptiveActivation(hidden_neurons,dtype=dtype)
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1); out=self.activation(self.input_layer(z))
        for layer in self.hidden_layers: out=self.activation(layer(out))
        return self.output_layer(out)

class StanActivation(nn.Module):


    def __init__(self,in_features,dtype,initial_beta=STAN_INITIAL_BETA):
        super().__init__()
        self.beta=nn.Parameter(torch.ones(in_features,dtype=dtype)*initial_beta)
    def forward(self,z):
        tanh_z=torch.tanh(z)
        return tanh_z+self.beta*z*tanh_z

class StanPINN(nn.Module):
    def __init__(self,hidden_layers=HIDDEN_LAYERS,hidden_neurons=HIDDEN_NEURONS,dtype=DTYPE_ADAM):
        super().__init__()


        self.input_layer=nn.Linear(2,hidden_neurons)
        self.hidden_layers=nn.ModuleList([nn.Linear(hidden_neurons,hidden_neurons) for _ in range(hidden_layers)])
        self.output_layer=nn.Linear(hidden_neurons,1)
        self.stan_activations=nn.ModuleList([
            StanActivation(hidden_neurons,dtype=dtype)
            for _ in range(hidden_layers+1)
        ])
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1)
        out=self.stan_activations[0](self.input_layer(z))
        for i,layer in enumerate(self.hidden_layers):
            out=self.stan_activations[i+1](layer(out))
        return self.output_layer(out)


class FixedActivationPINN(nn.Module):
    def __init__(self,activation_name,hidden_layers=HIDDEN_LAYERS,hidden_neurons=HIDDEN_NEURONS):
        super().__init__(); self.activation_name=activation_name
        self.input_layer=nn.Linear(2,hidden_neurons)
        self.hidden_layers=nn.ModuleList([nn.Linear(hidden_neurons,hidden_neurons) for _ in range(hidden_layers)])
        self.output_layer=nn.Linear(hidden_neurons,1)
        if activation_name=='gelu': self.activation=nn.GELU(approximate='none')
        elif activation_name=='mish': self.activation=nn.Mish()
        elif activation_name=='xtanh': self.activation=None
        else: raise ValueError(activation_name)
    def _activate(self,z):
        return z*torch.tanh(z) if self.activation_name=='xtanh' else self.activation(z)
    def forward(self,x,t):
        z=torch.cat([x,t],dim=1); out=self._activate(self.input_layer(z))
        for layer in self.hidden_layers: out=self._activate(layer(out))
        return self.output_layer(out)


def build_model(method,dtype):
    if method in {'gelu','mish','xtanh'}: m=FixedActivationPINN(method)
    else: raise ValueError(method)
    return m.to(dtype=dtype,device=DEVICE)

def method_label(method): return {'gelu':'GELU','mish':'Mish','xtanh':'x*tanh(x)'}[method]
def count_parameters(m): return sum(p.numel() for p in m.parameters())
def count_adaptive_parameters(m):
    if hasattr(m,'activation') and hasattr(m.activation,'alpha'):
        return m.activation.alpha.numel()+m.activation.beta.numel()
    if hasattr(m,'stan_activations'):
        return sum(a.beta.numel() for a in m.stan_activations)
    return 0

def adaptive_parameter_summary(m):
    if not(hasattr(m,'activation') and hasattr(m.activation,'alpha')): return {}
    a=m.activation.alpha.detach().cpu().numpy(); b=m.activation.beta.detach().cpu().numpy()
    return {'alpha_mean':float(a.mean()),'alpha_std':float(a.std(ddof=1)) if a.size>1 else 0.0,'alpha_min':float(a.min()),'alpha_max':float(a.max()),'beta_mean':float(b.mean()),'beta_std':float(b.std(ddof=1)) if b.size>1 else 0.0,'beta_min':float(b.min()),'beta_max':float(b.max()),'delta_alpha_l2':float(np.linalg.norm(a-INITIAL_ALPHA,2)),'delta_beta_l2':float(np.linalg.norm(b-INITIAL_BETA,2))}

def stan_parameter_summary(m):
    if not hasattr(m,'stan_activations'): return {}
    beta=torch.cat([a.beta.detach().reshape(-1).cpu() for a in m.stan_activations]).numpy()
    return {
        'beta_mean':float(beta.mean()),
        'beta_std':float(beta.std(ddof=1)) if beta.size>1 else 0.0,
        'beta_min':float(beta.min()),
        'beta_max':float(beta.max()),
        'delta_beta_l2':float(np.linalg.norm(beta-STAN_INITIAL_BETA,2)),
        'number_of_beta_parameters':int(beta.size),
        'number_of_activation_locations':int(len(m.stan_activations)),
    }

def save_json(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def training_points_path(seed): return POINTS_DIR/f'wave_training_points_seed{seed}.npz'

def generate_points_for_seed(seed):
    rng=np.random.default_rng(seed)
    x_res=rng.random((N_RES,1)); t_res=rng.random((N_RES,1))
    x_bc=np.vstack([np.zeros((N_BC//2,1)),np.ones((N_BC-N_BC//2,1))]); t_bc=rng.random((N_BC,1))
    x_ic1=rng.random((N_IC1,1)); t_ic1=np.zeros((N_IC1,1))
    x_ic2=rng.random((N_IC2,1)); t_ic2=np.zeros((N_IC2,1))
    return {k:v.astype(np.float64) for k,v in dict(x_res=x_res,t_res=t_res,x_bc=x_bc,t_bc=t_bc,x_ic1=x_ic1,t_ic1=t_ic1,x_ic2=x_ic2,t_ic2=t_ic2).items()}

def generate_and_save_all_training_points(overwrite=True):
    POINTS_DIR.mkdir(parents=True,exist_ok=True); files=[]
    for seed in SEEDS:
        p=training_points_path(seed)
        if overwrite or not p.exists(): np.savez_compressed(p,**generate_points_for_seed(seed)); print('Saved fixed training points:',p)
        else: print('Existing fixed point file kept:',p)
        files.append(str(p))
    save_json(POINTS_DIR/'training_points_manifest.json',{'seeds':SEEDS,'n_res':N_RES,'n_bc_ic':N_BC_IC,'n_bc':N_BC,'n_ic1':N_IC1,'n_ic2':N_IC2,'sampling':'fixed random sampling with numpy default_rng','files':files})

def load_training_points(seed,dtype):
    p=training_points_path(seed)
    if not p.exists(): raise FileNotFoundError(f'Missing fixed training points: {p}. Run generate_wave_fixed_training_points_seeds_2026_99_1234.py first.')
    d=np.load(p); keys=['x_res','t_res','x_bc','t_bc','x_ic1','t_ic1','x_ic2','t_ic2']
    return {k:torch.tensor(d[k],dtype=dtype,device=DEVICE) for k in keys}

def compute_loss_components(model,pts):
    x_res=pts['x_res'].detach().clone().requires_grad_(True); t_res=pts['t_res'].detach().clone().requires_grad_(True)
    u=model(x_res,t_res)
    u_t=torch.autograd.grad(u,t_res,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    u_x=torch.autograd.grad(u,x_res,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    u_tt=torch.autograd.grad(u_t,t_res,torch.ones_like(u_t),create_graph=True,retain_graph=True)[0]
    u_xx=torch.autograd.grad(u_x,x_res,torch.ones_like(u_x),create_graph=True,retain_graph=True)[0]
    L_res=torch.mean((u_tt-u_xx)**2)
    L_bc=torch.mean(model(pts['x_bc'],pts['t_bc'])**2)
    L_ic1=torch.mean(model(pts['x_ic1'],pts['t_ic1'])**2)
    x_ic2=pts['x_ic2'].detach().clone().requires_grad_(True); t_ic2=pts['t_ic2'].detach().clone().requires_grad_(True)
    u_ic2=model(x_ic2,t_ic2)
    u_t_ic2=torch.autograd.grad(u_ic2,t_ic2,torch.ones_like(u_ic2),create_graph=True,retain_graph=True)[0]
    target=2*np.pi*torch.sin(2*np.pi*x_ic2)+10*np.pi*torch.sin(10*np.pi*x_ic2)
    L_ic2=torch.mean((u_t_ic2-target)**2)
    L_data=L_bc+L_ic1+L_ic2; L_total=L_res+L_data
    return L_total,L_res,L_data,L_bc,L_ic1,L_ic2

def cpu_state_dict(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def state_dict_to_dtype(sd,dtype): return {k:(v.to(dtype=dtype) if torch.is_tensor(v) else v) for k,v in sd.items()}
def checkpoint_path(seed_dir,method,seed,name): return seed_dir/f'wave_{method}_seed{seed}_{name}.pth'

def method_config(method,seed,model):
    cfg={'method':method,'method_label':method_label(method),'seed':int(seed),'device':str(DEVICE),'dtype_adam':str(DTYPE_ADAM),'dtype_lbfgs':str(DTYPE_LBFGS),'hidden_layers':HIDDEN_LAYERS,'hidden_neurons':HIDDEN_NEURONS,'n_res':N_RES,'n_bc_ic':N_BC_IC,'n_bc':N_BC,'n_ic1':N_IC1,'n_ic2':N_IC2,'training_points_file':str(training_points_path(seed)),'adam_epochs':ADAM_EPOCHS,'adam_lr':ADAM_LR,'lbfgs_max_iter':LBFGS_MAX_ITER,'lbfgs_max_eval':LBFGS_MAX_EVAL,'lbfgs_lr':LBFGS_LR,'lbfgs_history_size':LBFGS_HISTORY_SIZE,'lbfgs_tolerance_grad':LBFGS_TOLERANCE_GRAD,'lbfgs_tolerance_change':LBFGS_TOLERANCE_CHANGE,'lbfgs_line_search_fn':LBFGS_LINE_SEARCH_FN,'total_training_steps':TOTAL_TRAINING_STEPS,'checkpoint_specs':CHECKPOINT_SPECS,'pde':'u_tt - u_xx = 0','exact_solution':'sin(2*pi*x)sin(2*pi*t)+sin(10*pi*x)sin(10*pi*t)','ic1':'u(x,0)=0','ic2':'u_t(x,0)=2*pi*sin(2*pi*x)+10*pi*sin(10*pi*x)','bc':'u(0,t)=0, u(1,t)=0','total_parameters':count_parameters(model),'adaptive_parameters':count_adaptive_parameters(model)}
    cfg['activation']={'gelu':"GELU exact: x*Phi(x), PyTorch nn.GELU(approximate='none')",'mish':'Mish: x*tanh(softplus(x)), PyTorch nn.Mish()','xtanh':'x*tanh(x)'}[method]
    cfg['comparison_role']='fixed activation only; no trainable activation parameters'
    return cfg

def save_checkpoint(path,method,model,optimizer,seed,phase,name,total_iteration,adam_iteration,lbfgs_eval,history,elapsed,metrics,cfg):
    ck={'method':method,'method_label':method_label(method),'seed':int(seed),'phase':phase,'checkpoint_name':name,'total_iteration':int(total_iteration),'adam_iteration':int(adam_iteration),'lbfgs_eval':int(lbfgs_eval),'model_state_dict':cpu_state_dict(model),'optimizer_state_dict':optimizer.state_dict() if optimizer is not None else None,'loss_history':np.asarray(history,dtype=object),'elapsed_seconds':float(elapsed),'latest_metrics':metrics,'config':cfg}
    if method=='nmaf_aat': ck['adaptive_parameter_summary']=adaptive_parameter_summary(model)
    if method=='stan': ck['stan_parameter_summary']=stan_parameter_summary(model)
    torch.save(ck,path)

def train_single_seed(method,seed):
    set_seed(seed); seed_dir=RESULT_ROOT/method/f'seed_{seed}'; seed_dir.mkdir(parents=True,exist_ok=True)
    model=build_model(method,DTYPE_ADAM); pts=load_training_points(seed,DTYPE_ADAM); cfg=method_config(method,seed,model); save_json(seed_dir/'method_config.json',cfg)
    opt=torch.optim.Adam(model.parameters(),lr=ADAM_LR); hist=[]; latest=None; sync(); t0=time.time()
    print('='*100); print(f'{method_label(method)} | seed={seed} | Adam stage starts: epochs={ADAM_EPOCHS}, lr={ADAM_LR}'); print('='*100)
    for ep in range(1,ADAM_EPOCHS+1):
        opt.zero_grad(set_to_none=True); loss,lres,ldata,lbc,lic1,lic2=compute_loss_components(model,pts); loss.backward(); opt.step(); sync(); elapsed=time.time()-t0
        latest={'total_loss':float(loss.detach().cpu()),'residual_loss':float(lres.detach().cpu()),'data_loss':float(ldata.detach().cpu()),'bc_loss':float(lbc.detach().cpu()),'ic1_loss':float(lic1.detach().cpu()),'ic2_loss':float(lic2.detach().cpu()),'elapsed_seconds':float(elapsed)}
        hist.append([ep,'adam',ep,0,latest['total_loss'],latest['residual_loss'],latest['data_loss'],latest['bc_loss'],latest['ic1_loss'],latest['ic2_loss'],latest['elapsed_seconds']])
        if ep==1 or ep%PRINT_EVERY_ADAM==0: print(f'{method_label(method):13s} | seed {seed:4d} | Adam iter {ep:7d}/{ADAM_EPOCHS} | Total={latest["total_loss"]:.6e} | Residual={latest["residual_loss"]:.6e} | Data={latest["data_loss"]:.6e}')
    spec=CHECKPOINT_SPECS[0]
    save_checkpoint(checkpoint_path(seed_dir,method,seed,spec['name']),method,model,opt,seed,'adam',spec['name'],ADAM_EPOCHS,ADAM_EPOCHS,0,hist,time.time()-t0,latest,cfg)
    print('Saved Adam checkpoint:', checkpoint_path(seed_dir,method,seed,spec['name']))
    print('='*100); print(f'{method_label(method)} | seed={seed} | L-BFGS stage starts: max_eval={LBFGS_MAX_EVAL}, lr={LBFGS_LR}'); print('='*100)
    model2=build_model(method,DTYPE_LBFGS); model2.load_state_dict(state_dict_to_dtype(model.state_dict(),DTYPE_LBFGS),strict=True)
    del model, opt
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    pts2=load_training_points(seed,DTYPE_LBFGS)
    opt2=torch.optim.LBFGS(model2.parameters(),lr=LBFGS_LR,max_iter=LBFGS_MAX_ITER,max_eval=LBFGS_MAX_EVAL,history_size=LBFGS_HISTORY_SIZE,tolerance_grad=LBFGS_TOLERANCE_GRAD,tolerance_change=LBFGS_TOLERANCE_CHANGE,line_search_fn=LBFGS_LINE_SEARCH_FN)
    lb_eval=0
    def closure():
        nonlocal lb_eval, latest
        opt2.zero_grad(set_to_none=True); loss,lres,ldata,lbc,lic1,lic2=compute_loss_components(model2,pts2); loss.backward(); lb_eval+=1; giter=ADAM_EPOCHS+lb_eval; sync(); elapsed=time.time()-t0
        latest={'total_loss':float(loss.detach().cpu()),'residual_loss':float(lres.detach().cpu()),'data_loss':float(ldata.detach().cpu()),'bc_loss':float(lbc.detach().cpu()),'ic1_loss':float(lic1.detach().cpu()),'ic2_loss':float(lic2.detach().cpu()),'elapsed_seconds':float(elapsed)}
        hist.append([giter,'lbfgs',ADAM_EPOCHS,lb_eval,latest['total_loss'],latest['residual_loss'],latest['data_loss'],latest['bc_loss'],latest['ic1_loss'],latest['ic2_loss'],latest['elapsed_seconds']])
        if lb_eval==1 or lb_eval%PRINT_EVERY_LBFGS==0: print(f'{method_label(method):13s} | seed {seed:4d} | L-BFGS eval {lb_eval:7d}/{LBFGS_MAX_EVAL} | Global iter {giter:7d} | Total={latest["total_loss"]:.6e} | Residual={latest["residual_loss"]:.6e} | Data={latest["data_loss"]:.6e}')
        return loss
    opt2.step(closure); total_time=time.time()-t0; spec=CHECKPOINT_SPECS[1]
    save_checkpoint(checkpoint_path(seed_dir,method,seed,spec['name']),method,model2,opt2,seed,'adam_lbfgs',spec['name'],ADAM_EPOCHS+lb_eval,ADAM_EPOCHS,lb_eval,hist,total_time,latest,cfg)
    print('Saved final checkpoint:', checkpoint_path(seed_dir,method,seed,spec['name']))
    cols=['global_iteration','phase','adam_iteration','lbfgs_eval','total_loss','residual_loss','data_loss','bc_loss','ic1_loss','ic2_loss','elapsed_seconds']
    with (seed_dir/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols); w.writerows(hist)
    np.save(seed_dir/'training_history.npy',np.asarray(hist,dtype=object),allow_pickle=True)
    scipy.io.savemat(seed_dir/'training_history.mat',{'history':np.asarray(hist,dtype=object),'column_names':cols})
    summary={'method':method,'method_label':method_label(method),'seed':int(seed),'adam_epochs':ADAM_EPOCHS,'completed_lbfgs_evals':int(lb_eval),'target_lbfgs_evals':LBFGS_MAX_EVAL,'final_global_iteration':int(ADAM_EPOCHS+lb_eval),'target_total_training_steps':TOTAL_TRAINING_STEPS,'total_time_seconds':float(total_time),'final_metrics':latest,'training_points_file':str(training_points_path(seed)),'saved_checkpoints':[s['name'] for s in CHECKPOINT_SPECS]}
    if method=='nmaf_aat': summary['final_adaptive_parameter_summary']=adaptive_parameter_summary(model2)
    if method=='stan': summary['final_stan_parameter_summary']=stan_parameter_summary(model2)
    save_json(seed_dir/'training_summary.json',summary)
    print('-'*100); print(f'{method_label(method)} seed {seed} two-stage training completed.'); print(f'Final global iteration: {ADAM_EPOCHS+lb_eval}'); print(f'Final total loss: {latest["total_loss"]:.6e}'); print('-'*100)
    return summary

def train_all_seeds(method):
    (RESULT_ROOT/method).mkdir(parents=True,exist_ok=True); summaries=[train_single_seed(method,seed) for seed in SEEDS]
    save_json(RESULT_ROOT/method/'all_training_summaries.json',{'method':method,'method_label':method_label(method),'seeds':SEEDS,'summaries':summaries})
