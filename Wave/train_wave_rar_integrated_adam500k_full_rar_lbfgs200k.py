import csv, json, time, hashlib
from pathlib import Path
import numpy as np
import scipy.io
import torch

from wave_common_fixed_points_adam_lbfgs import (
    SEEDS, RESULT_ROOT, DEVICE, DTYPE_ADAM, DTYPE_LBFGS,
    HIDDEN_LAYERS, HIDDEN_NEURONS, N_RES, N_BC, N_IC1, N_IC2,
    ADAM_EPOCHS, ADAM_LR, LBFGS_MAX_ITER, LBFGS_MAX_EVAL, LBFGS_LR,
    LBFGS_HISTORY_SIZE, LBFGS_TOLERANCE_GRAD, LBFGS_TOLERANCE_CHANGE,
    LBFGS_LINE_SEARCH_FN, TOTAL_TRAINING_STEPS, CHECKPOINT_SPECS,
    PRINT_EVERY_ADAM, PRINT_EVERY_LBFGS, TanhPINN, set_seed, sync,
    load_training_points, training_points_path, state_dict_to_dtype,
    cpu_state_dict, count_parameters, save_json,
)

METHOD = 'rar_adam_integrated_base6400_plus1280'
METHOD_LABEL = 'RAR-guided PINN'
METHOD_DIR = RESULT_ROOT / METHOD

RAR_INTERVAL = 50000
RAR_ROUNDS = 10
RAR_SELECTION_ADAM_STEPS = tuple(
    RAR_INTERVAL * i
    for i in range(1, RAR_ROUNDS + 1)
)


RAR_SELECTION_ADAM_STEP = RAR_SELECTION_ADAM_STEPS[0]

RAR_CANDIDATES = 100000
RAR_CANDIDATE_BATCH = 10000
RAR_BASE_N_RES = N_RES

RAR_ADD_PER_ROUND = 128
RAR_ADD_N_RES = RAR_ROUNDS * RAR_ADD_PER_ROUND
RAR_FINAL_N_RES = RAR_BASE_N_RES + RAR_ADD_N_RES

assert N_RES == 6400
assert ADAM_EPOCHS == 500000
assert RAR_INTERVAL == 50000
assert RAR_ROUNDS == 10
assert RAR_SELECTION_ADAM_STEPS == (
    50000, 100000, 150000, 200000, 250000,
    300000, 350000, 400000, 450000, 500000,
)
assert RAR_ADD_PER_ROUND == 128
assert RAR_ADD_N_RES == 1280
assert RAR_FINAL_N_RES == 7680
assert LBFGS_MAX_EVAL == 200000


def build_model(dtype):
    return TanhPINN(HIDDEN_LAYERS, HIDDEN_NEURONS).to(dtype=dtype, device=DEVICE)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode('utf-8'))
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_exact_standard_init(seed):

    set_seed(seed)
    ref = build_model(DTYPE_ADAM)
    expected = {k: v.detach().cpu().clone() for k, v in ref.state_dict().items()}
    expected_hash = state_hash(ref)
    del ref
    set_seed(seed)
    model = build_model(DTYPE_ADAM)
    if state_hash(model) != expected_hash:
        raise RuntimeError(f'Initial-state hash mismatch for seed={seed}')
    for k, v in model.state_dict().items():
        if not torch.equal(v.detach().cpu(), expected[k]):
            raise RuntimeError(f'Initial tensor mismatch: seed={seed}, {k}')
    return model, expected_hash


def load_original_np(seed):
    p = training_points_path(seed)
    if not p.exists():
        raise FileNotFoundError(f'Missing fixed training points: {p}')
    with np.load(p) as d:
        return {k: np.asarray(d[k], dtype=np.float64).copy() for k in
                ['x_res','t_res','x_bc','t_bc','x_ic1','t_ic1','x_ic2','t_ic2']}


def make_points(base_pts, x_res_np, t_res_np, dtype):
    return {
        'x_res': torch.tensor(x_res_np, dtype=dtype, device=DEVICE),
        't_res': torch.tensor(t_res_np, dtype=dtype, device=DEVICE),
        'x_bc': base_pts['x_bc'].to(dtype=dtype), 't_bc': base_pts['t_bc'].to(dtype=dtype),
        'x_ic1': base_pts['x_ic1'].to(dtype=dtype), 't_ic1': base_pts['t_ic1'].to(dtype=dtype),
        'x_ic2': base_pts['x_ic2'].to(dtype=dtype), 't_ic2': base_pts['t_ic2'].to(dtype=dtype),
    }


def residual(model, x, t, training_graph=True):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=training_graph, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=training_graph, retain_graph=training_graph)[0]
    return u_tt - u_xx


def compute_both_losses(model, pts, base_n=RAR_BASE_N_RES):

    x = pts['x_res'].detach().clone().requires_grad_(True)
    t = pts['t_res'].detach().clone().requires_grad_(True)
    r = residual(model, x, t, training_graph=True)
    rsq = r ** 2
    opt_res = torch.mean(rsq)
    mon_res = torch.mean(rsq[:base_n])
    lbc = torch.mean(model(pts['x_bc'], pts['t_bc']) ** 2)
    lic1 = torch.mean(model(pts['x_ic1'], pts['t_ic1']) ** 2)
    x2 = pts['x_ic2'].detach().clone().requires_grad_(True)
    t2 = pts['t_ic2'].detach().clone().requires_grad_(True)
    u2 = model(x2, t2)
    ut2 = torch.autograd.grad(u2, t2, torch.ones_like(u2), create_graph=True, retain_graph=True)[0]
    target = 2*np.pi*torch.sin(2*np.pi*x2) + 10*np.pi*torch.sin(10*np.pi*x2)
    lic2 = torch.mean((ut2-target)**2)
    data = lbc + lic1 + lic2
    return dict(
        optimization_total_loss=opt_res+data,
        monitor_total_loss=mon_res+data,
        optimization_residual_loss=opt_res,
        monitor_residual_loss=mon_res,
        data_loss=data, bc_loss=lbc, ic1_loss=lic1, ic2_loss=lic2,
    )


def candidate_pool(seed, rar_round, n_candidates=RAR_CANDIDATES):


    rng = np.random.default_rng(
        int(seed)
        + 910271
        + 1000003 * int(rar_round)
    )
    return (
        rng.random((n_candidates, 1)).astype(np.float64),
        rng.random((n_candidates, 1)).astype(np.float64),
    )


def select_points(
    model,
    seed,
    rar_round,
    n_candidates=RAR_CANDIDATES,
    n_add=RAR_ADD_PER_ROUND,
):
    xall, tall = candidate_pool(
        seed,
        rar_round,
        n_candidates=n_candidates,
    )

    vals = []
    was_training = model.training
    model.eval()

    sync()
    t0 = time.time()

    for s in range(0, n_candidates, RAR_CANDIDATE_BATCH):
        e = min(s + RAR_CANDIDATE_BATCH, n_candidates)

        x = torch.tensor(
            xall[s:e],
            dtype=DTYPE_ADAM,
            device=DEVICE,
            requires_grad=True,
        )

        t = torch.tensor(
            tall[s:e],
            dtype=DTYPE_ADAM,
            device=DEVICE,
            requires_grad=True,
        )

        with torch.enable_grad():
            r = residual(
                model,
                x,
                t,
                training_graph=False,
            )

        vals.append(
            np.abs(
                r.detach().cpu().numpy()
            ).reshape(-1)
        )

    if was_training:
        model.train()

    ar = np.concatenate(vals)

    ids = np.argpartition(
        ar,
        -n_add,
    )[-n_add:]

    ids = ids[
        np.argsort(ar[ids])[::-1]
    ]

    sync()
    elapsed = time.time() - t0

    return (
        xall[ids].copy(),
        tall[ids].copy(),
        ar[ids].copy(),
        {
            'rar_round': int(rar_round),
            'candidate_count': int(n_candidates),
            'selected_count': int(n_add),
            'candidate_abs_residual_mean': float(ar.mean()),
            'candidate_abs_residual_max': float(ar.max()),
            'selected_abs_residual_mean': float(ar[ids].mean()),
            'selected_abs_residual_min': float(ar[ids].min()),
            'selected_abs_residual_max': float(ar[ids].max()),
            'selection_wall_seconds': float(elapsed),
        },
    )


def point_path(seed):
    return METHOD_DIR / f'seed_{seed}' / f'wave_rar_base6400_plus1280_points_seed{seed}.npz'


def ckpt_path(seed_dir, seed, name):
    return seed_dir / f'wave_{METHOD}_seed{seed}_{name}.pth'


def save_ckpt(path, model, optimizer, seed, phase, name, total_it, adam_it, lb_eval,
              history, elapsed, latest, cfg, rar_meta):
    torch.save({
        'method': METHOD, 'method_label': METHOD_LABEL, 'seed': int(seed),
        'phase': phase, 'checkpoint_name': name, 'total_iteration': int(total_it),
        'adam_iteration': int(adam_it), 'lbfgs_eval': int(lb_eval),
        'model_state_dict': cpu_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss_history': np.asarray(history, dtype=object), 'elapsed_seconds': float(elapsed),
        'latest_metrics': latest, 'config': cfg, 'rar_metadata': rar_meta,
    }, path)


def train_seed(seed):
    seed = int(seed)
    seed_dir = METHOD_DIR / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    model, init_hash = build_exact_standard_init(seed)
    original_np = load_original_np(seed)
    original_pts = load_training_points(seed, DTYPE_ADAM)
    ox, ot = original_np['x_res'], original_np['t_res']
    if len(ox) != N_RES: raise RuntimeError('Original Wave N_RES mismatch')
    cx, ct = ox.copy(), ot.copy()
    pts = make_points(original_pts, cx, ct, DTYPE_ADAM)
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    cfg = {
        'method': METHOD, 'method_label': METHOD_LABEL, 'seed': seed,
        'initial_state_sha256': init_hash, 'initialization_matches_standard': True,
        'hidden_layers': HIDDEN_LAYERS, 'hidden_neurons': HIDDEN_NEURONS,
        'original_shared_n_res': N_RES,
        'rar_added_n_res': RAR_ADD_N_RES,
        'final_n_res': RAR_FINAL_N_RES,
        'rar_fraction': RAR_ADD_N_RES / N_RES,
        'rar_interval': RAR_INTERVAL,
        'rar_rounds': RAR_ROUNDS,
        'rar_add_per_round': RAR_ADD_PER_ROUND,
        'rar_selection_adam_steps': list(RAR_SELECTION_ADAM_STEPS),
        'rar_first_selection_adam_step': RAR_SELECTION_ADAM_STEP,
        'rar_last_selection_adam_step': RAR_SELECTION_ADAM_STEPS[-1],
        'rar_candidates_per_round': RAR_CANDIDATES,
        'rar_score': 'abs(u_tt-u_xx)',
        'adam_epochs': ADAM_EPOCHS,
        'adam_lr': ADAM_LR,
        'model_reset_at_rar': False,
        'adam_optimizer_reset_at_rar': False,
        'adam_optimizer_state_retained': True,
        'lbfgs_max_iter': LBFGS_MAX_ITER, 'lbfgs_max_eval': LBFGS_MAX_EVAL,
        'lbfgs_lr': LBFGS_LR, 'lbfgs_history_size': LBFGS_HISTORY_SIZE,
        'lbfgs_tolerance_grad': LBFGS_TOLERANCE_GRAD,
        'lbfgs_tolerance_change': LBFGS_TOLERANCE_CHANGE,
        'lbfgs_line_search_fn': LBFGS_LINE_SEARCH_FN,
        'primary_total_loss': 'original shared 6400-point monitor loss',
        'optimization_total_loss': 'actual current training-set loss',
        'total_parameters': count_parameters(model),
    }
    save_json(seed_dir/'method_config.json', cfg)

    hist = []
    latest = None

    source_round = np.zeros(
        N_RES,
        dtype=np.int32,
    )

    rar_round_history = []
    cumulative_rar_selection_seconds = 0.0

    rar_meta = {
        'selection_completed': False,
        'base_n_res': N_RES,
        'added_n_res': 0,
        'final_n_res': N_RES,
        'rar_interval': RAR_INTERVAL,
        'rar_rounds_target': RAR_ROUNDS,
        'rar_add_per_round': RAR_ADD_PER_ROUND,
        'selection_adam_steps': list(RAR_SELECTION_ADAM_STEPS),
        'rounds': rar_round_history,
        'cumulative_selection_wall_seconds': 0.0,
    }

    sync()
    t0 = time.time()

    print('=' * 108)
    print(
        f'{METHOD_LABEL} | seed={seed} | '
        f'Adam500000 with RAR throughout the full Adam stage'
    )
    print(
        f'RAR every {RAR_INTERVAL} Adam steps | '
        f'{RAR_ROUNDS} rounds x {RAR_ADD_PER_ROUND} points '
        f'= +{RAR_ADD_N_RES} points'
    )
    print('=' * 108)

    for ep in range(1, ADAM_EPOCHS+1):
        opt.zero_grad(set_to_none=True)
        L = compute_both_losses(model, pts, base_n=N_RES)
        L['optimization_total_loss'].backward(); opt.step(); sync()
        elapsed=time.time()-t0
        latest={
            'total_loss':float(L['monitor_total_loss'].detach().cpu()),
            'optimization_total_loss':float(L['optimization_total_loss'].detach().cpu()),
            'residual_loss':float(L['monitor_residual_loss'].detach().cpu()),
            'optimization_residual_loss':float(L['optimization_residual_loss'].detach().cpu()),
            'data_loss':float(L['data_loss'].detach().cpu()),
            'bc_loss':float(L['bc_loss'].detach().cpu()),
            'ic1_loss':float(L['ic1_loss'].detach().cpu()),
            'ic2_loss':float(L['ic2_loss'].detach().cpu()),
            'n_res_training':len(cx), 'elapsed_seconds':float(elapsed),
        }
        hist.append([ep,'adam',ep,0,latest['total_loss'],latest['optimization_total_loss'],
                     latest['residual_loss'],latest['optimization_residual_loss'],latest['data_loss'],
                     latest['bc_loss'],latest['ic1_loss'],latest['ic2_loss'],len(cx),elapsed])
        if ep==1 or ep%PRINT_EVERY_ADAM==0:
            print(f'{METHOD_LABEL:13s} | seed {seed:4d} | Adam {ep:7d}/{ADAM_EPOCHS} | N_RES={len(cx):4d} | Monitor={latest["total_loss"]:.6e} | Opt={latest["optimization_total_loss"]:.6e}')

        if ep in RAR_SELECTION_ADAM_STEPS:
            rar_round = ep // RAR_INTERVAL

            optimizer_id = id(opt)
            model_id = id(model)
            state_count = len(opt.state)

            xa, ta, rr, st = select_points(
                model,
                seed,
                rar_round,
            )


            cx = np.vstack([
                cx,
                xa,
            ])

            ct = np.vstack([
                ct,
                ta,
            ])

            source_round = np.concatenate([
                source_round,
                np.full(
                    len(xa),
                    rar_round,
                    dtype=np.int32,
                ),
            ])

            expected_n_res = (
                N_RES
                + rar_round * RAR_ADD_PER_ROUND
            )

            if len(cx) != expected_n_res:
                raise RuntimeError(
                    f'RAR round {rar_round}: N_RES={len(cx)}, '
                    f'expected {expected_n_res}'
                )

            if not (
                np.array_equal(
                    cx[:N_RES],
                    ox,
                )
                and np.array_equal(
                    ct[:N_RES],
                    ot,
                )
            ):
                raise RuntimeError(
                    'Original 6400-point prefix changed'
                )

            cumulative_rar_selection_seconds += st[
                'selection_wall_seconds'
            ]

            round_record = {
                'rar_round': int(rar_round),
                'adam_step': int(ep),
                'n_res_before': int(
                    len(cx) - len(xa)
                ),
                'n_res_after': int(
                    len(cx)
                ),
                **st,
            }

            rar_round_history.append(
                round_record
            )

            rar_meta.update({
                'selection_completed': (
                    rar_round == RAR_ROUNDS
                ),
                'completed_rounds': int(
                    rar_round
                ),
                'added_n_res': int(
                    len(cx) - N_RES
                ),
                'final_n_res': int(
                    len(cx)
                ),
                'selected_points_file': str(
                    point_path(seed)
                ),
                'cumulative_selection_wall_seconds': float(
                    cumulative_rar_selection_seconds
                ),
                'rounds': rar_round_history,
            })


            np.savez_compressed(
                point_path(seed),
                x_res=cx,
                t_res=ct,

                source=(
                    source_round > 0
                ).astype(np.int32),

                source_round=source_round,
                original_shared_count=np.array(
                    [N_RES],
                    dtype=np.int32,
                ),
                adaptive_added_count=np.array(
                    [len(cx) - N_RES],
                    dtype=np.int32,
                ),
                final_n_res=np.array(
                    [len(cx)],
                    dtype=np.int32,
                ),
                completed_rar_rounds=np.array(
                    [rar_round],
                    dtype=np.int32,
                ),
                selection_adam_steps=np.asarray(
                    RAR_SELECTION_ADAM_STEPS[:rar_round],
                    dtype=np.int32,
                ),
            )

            save_json(
                seed_dir / 'rar_selection_summary.json',
                rar_meta,
            )


            pts = make_points(
                original_pts,
                cx,
                ct,
                DTYPE_ADAM,
            )

            if (
                id(opt) != optimizer_id
                or id(model) != model_id
                or len(opt.state) != state_count
            ):
                raise RuntimeError(
                    'Model/Adam optimizer state was unexpectedly reset '
                    f'at RAR round {rar_round}'
                )

            print(
                f'RAR round {rar_round:2d}/{RAR_ROUNDS} at Adam {ep:6d}: '
                f'+{RAR_ADD_PER_ROUND} -> N_RES={len(cx)} | '
                f'model retained | Adam state retained | '
                f'selected |f| mean={st["selected_abs_residual_mean"]:.6e}'
            )

    if len(cx) != RAR_FINAL_N_RES:
        raise RuntimeError(
            f'After Adam500000, final RAR N_RES={len(cx)}, '
            f'expected {RAR_FINAL_N_RES}'
        )

    if len(rar_round_history) != RAR_ROUNDS:
        raise RuntimeError(
            f'Completed RAR rounds={len(rar_round_history)}, '
            f'expected {RAR_ROUNDS}'
        )

    spec=CHECKPOINT_SPECS[0]
    save_ckpt(ckpt_path(seed_dir,seed,spec['name']),model,opt,seed,'adam',spec['name'],
              ADAM_EPOCHS,ADAM_EPOCHS,0,hist,time.time()-t0,latest,cfg,rar_meta)


    model2=build_model(DTYPE_LBFGS)
    model2.load_state_dict(state_dict_to_dtype(model.state_dict(),DTYPE_LBFGS),strict=True)
    del model,opt
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    base64=load_training_points(seed,DTYPE_LBFGS)
    pts64=make_points(base64,cx,ct,DTYPE_LBFGS)
    opt2=torch.optim.LBFGS(model2.parameters(),lr=LBFGS_LR,max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,history_size=LBFGS_HISTORY_SIZE,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn=LBFGS_LINE_SEARCH_FN)
    lb_eval=0
    print('='*108); print(f'{METHOD_LABEL} | seed={seed} | L-BFGS starts on final fixed {RAR_FINAL_N_RES} PDE points'); print('='*108)
    def closure():
        nonlocal lb_eval,latest
        opt2.zero_grad(set_to_none=True)
        L=compute_both_losses(model2,pts64,base_n=N_RES)
        L['optimization_total_loss'].backward(); lb_eval+=1; g=ADAM_EPOCHS+lb_eval; sync(); elapsed=time.time()-t0
        latest={'total_loss':float(L['monitor_total_loss'].detach().cpu()),
                'optimization_total_loss':float(L['optimization_total_loss'].detach().cpu()),
                'residual_loss':float(L['monitor_residual_loss'].detach().cpu()),
                'optimization_residual_loss':float(L['optimization_residual_loss'].detach().cpu()),
                'data_loss':float(L['data_loss'].detach().cpu()),'bc_loss':float(L['bc_loss'].detach().cpu()),
                'ic1_loss':float(L['ic1_loss'].detach().cpu()),'ic2_loss':float(L['ic2_loss'].detach().cpu()),
                'n_res_training':RAR_FINAL_N_RES,'elapsed_seconds':float(elapsed)}
        hist.append([g,'lbfgs',ADAM_EPOCHS,lb_eval,latest['total_loss'],latest['optimization_total_loss'],
                     latest['residual_loss'],latest['optimization_residual_loss'],latest['data_loss'],
                     latest['bc_loss'],latest['ic1_loss'],latest['ic2_loss'],RAR_FINAL_N_RES,elapsed])
        if lb_eval==1 or lb_eval%PRINT_EVERY_LBFGS==0:
            print(f'{METHOD_LABEL:13s} | seed {seed:4d} | L-BFGS {lb_eval:7d}/{LBFGS_MAX_EVAL} | Global {g:7d} | Monitor={latest["total_loss"]:.6e} | Opt={latest["optimization_total_loss"]:.6e}')
        return L['optimization_total_loss']
    opt2.step(closure); sync(); total_time=time.time()-t0
    spec=CHECKPOINT_SPECS[1]
    save_ckpt(ckpt_path(seed_dir,seed,spec['name']),model2,opt2,seed,'adam_lbfgs',spec['name'],
              ADAM_EPOCHS+lb_eval,ADAM_EPOCHS,lb_eval,hist,total_time,latest,cfg,rar_meta)

    cols=['global_iteration','phase','adam_iteration','lbfgs_eval','total_loss','optimization_total_loss',
          'residual_loss','optimization_residual_loss','data_loss','bc_loss','ic1_loss','ic2_loss',
          'n_res_training','elapsed_seconds']
    with (seed_dir/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(cols); w.writerows(hist)
    np.save(seed_dir/'training_history.npy',np.asarray(hist,dtype=object),allow_pickle=True)
    scipy.io.savemat(seed_dir/'training_history.mat',{'history':np.asarray(hist,dtype=object),'column_names':cols})
    summary={'method':METHOD,'method_label':METHOD_LABEL,'seed':seed,'initial_state_sha256':init_hash,
             'initialization_matches_standard':True,'original_shared_n_res':N_RES,'rar_added_n_res':RAR_ADD_N_RES,
             'final_n_res':RAR_FINAL_N_RES,
             'rar_interval':RAR_INTERVAL,
             'rar_rounds':RAR_ROUNDS,
             'rar_add_per_round':RAR_ADD_PER_ROUND,
             'rar_selection_adam_steps':list(RAR_SELECTION_ADAM_STEPS),
             'adam_epochs':ADAM_EPOCHS,
             'model_reset_at_rar':False,'adam_optimizer_reset_at_rar':False,
             'completed_lbfgs_evals':int(lb_eval),'target_lbfgs_evals':LBFGS_MAX_EVAL,
             'final_global_iteration':int(ADAM_EPOCHS+lb_eval),'target_total_training_steps':TOTAL_TRAINING_STEPS,
             'total_time_seconds':float(total_time),'rar_selection_seconds':float(rar_meta.get('cumulative_selection_wall_seconds',np.nan)),
             'final_metrics':latest,'selected_points_file':str(point_path(seed))}
    save_json(seed_dir/'training_summary.json',summary)
    return summary


def train_all_seeds():
    METHOD_DIR.mkdir(parents=True,exist_ok=True)
    summaries=[train_seed(seed) for seed in SEEDS]
    save_json(METHOD_DIR/'all_training_summaries.json',{'method':METHOD,'method_label':METHOD_LABEL,'seeds':SEEDS,'summaries':summaries})


if __name__ == '__main__':
    train_all_seeds()
