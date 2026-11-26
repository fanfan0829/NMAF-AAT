import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from allen_cahn_common_fixed_points_continuous_lbfgs import (
    SEEDS, REPRESENTATIVE_SEED, RESULT_ROOT, REFERENCE_FILE,
    DEVICE, DTYPE, HIDDEN_LAYERS, HIDDEN_NEURONS,
    N_RES, N_BC, N_IC, X_MIN, X_MAX, T_MIN, T_MAX,
    LAMBDA_RES, LAMBDA_BC, LAMBDA_IC,
    LBFGS_LR, LBFGS_MAX_EVAL, LBFGS_MAX_ITER, LBFGS_HISTORY_SIZE,
    LBFGS_TOLERANCE_GRAD, LBFGS_TOLERANCE_CHANGE, LBFGS_LINE_SEARCH_FN,
    CHECKPOINT_EVALS, PRINT_EVERY, IMAGE_DPI,
    TanhPINN, set_seed, synchronize, load_training_points,
    training_points_path, compute_loss_components, cpu_state_dict,
    count_parameters, save_json,
)

METHOD = "rar_sampling_only_base3000_plus600"
METHOD_LABEL = "RAR-guided PINN"
METHOD_DIR = RESULT_ROOT / METHOD


ADAM_ACQUISITION_STEPS = 2500
ADAM_LR = 1.0e-3
RAR_INITIAL_N_RES = N_RES
RAR_INTERVAL = 250
RAR_ROUNDS = 8
RAR_ADD_PER_ROUND = 75
RAR_CANDIDATES = 100000
RAR_CANDIDATE_BATCH = 10000
RAR_LAST_REFINEMENT_STEP = RAR_INTERVAL * RAR_ROUNDS
RAR_ADDED_TOTAL = RAR_ROUNDS * RAR_ADD_PER_ROUND
FINAL_N_RES = RAR_INITIAL_N_RES + RAR_ADDED_TOTAL
FINAL_ADAM_STEPS_AFTER_LAST_RAR = ADAM_ACQUISITION_STEPS - RAR_LAST_REFINEMENT_STEP
SAVE_ACQUISITION_MODEL = False


LBFGS_MONITOR_EVERY = 10

assert N_RES == 3000
assert RAR_ADDED_TOTAL == 600
assert FINAL_N_RES == 3600
assert RAR_LAST_REFINEMENT_STEP == 2000
assert FINAL_ADAM_STEPS_AFTER_LAST_RAR == 500
assert LBFGS_MAX_ITER == 20000
assert LBFGS_MAX_EVAL == 20000
assert CHECKPOINT_EVALS == (10000, 20000)


def build_fresh_tanh_model():
    return TanhPINN(
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
    ).double().to(DEVICE)


def tensor_state_sha256(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def assert_fresh_initialization_matches_standard(seed):

    set_seed(seed)
    reference = build_fresh_tanh_model()
    expected_hash = tensor_state_sha256(reference)
    expected = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    del reference

    set_seed(seed)
    model = build_fresh_tanh_model()
    actual_hash = tensor_state_sha256(model)

    if actual_hash != expected_hash:
        raise RuntimeError(f"Initialization hash mismatch for seed={seed}")
    for key, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), expected[key]):
            raise RuntimeError(f"Initialization tensor mismatch: seed={seed}, key={key}")
    return model, actual_hash


def selected_points_path(seed):
    return METHOD_DIR / f"seed_{seed}" / f"allen_cahn_rar_base3000_plus600_points_seed{seed}.npz"


def load_original_points_numpy(seed):
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed AC points: {path}")
    with np.load(path) as data:
        keys = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
        return {key: np.asarray(data[key], dtype=np.float64).copy() for key in keys}


def make_points_with_residual_set(original_points, x_res_np, t_res_np):
    return {
        "x_res": torch.tensor(x_res_np, dtype=DTYPE, device=DEVICE),
        "t_res": torch.tensor(t_res_np, dtype=DTYPE, device=DEVICE),
        "x_bc_l": original_points["x_bc_l"],
        "x_bc_r": original_points["x_bc_r"],
        "t_bc": original_points["t_bc"],
        "x_ic": original_points["x_ic"],
        "t_ic": original_points["t_ic"],
    }


def pde_residual(model, x, t, create_graph=False):

    u = model(x, t)
    u_t = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=create_graph, retain_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=create_graph, retain_graph=True
    )[0]
    return u_t - 0.0001 * u_xx + 5.0 * u ** 3 - 5.0 * u


def evaluate_candidate_abs_residual(model, x_np, t_np, batch_size=RAR_CANDIDATE_BATCH):
    was_training = model.training
    model.eval()
    pieces = []
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        x = torch.tensor(x_np[start:end], dtype=DTYPE, device=DEVICE, requires_grad=True)
        t = torch.tensor(t_np[start:end], dtype=DTYPE, device=DEVICE, requires_grad=True)
        with torch.enable_grad():
            residual = pde_residual(model, x, t, create_graph=False)
        pieces.append(np.abs(residual.detach().cpu().numpy()).reshape(-1))
    if was_training:
        model.train()
    return np.concatenate(pieces)


def generate_candidate_pool(seed, rar_round, n_candidates=RAR_CANDIDATES):
    rng = np.random.default_rng(int(seed) + 1000003 * int(rar_round) + 7919)
    x = rng.uniform(X_MIN, X_MAX, size=(n_candidates, 1)).astype(np.float64)
    t = rng.uniform(T_MIN, T_MAX, size=(n_candidates, 1)).astype(np.float64)
    return x, t


def select_rar_points(model, seed, rar_round, n_candidates=RAR_CANDIDATES, n_add=RAR_ADD_PER_ROUND):
    synchronize()
    start = time.time()
    x_candidates, t_candidates = generate_candidate_pool(seed, rar_round, n_candidates)
    abs_residual = evaluate_candidate_abs_residual(model, x_candidates, t_candidates)
    ids = np.argpartition(abs_residual, -n_add)[-n_add:]
    ids = ids[np.argsort(abs_residual[ids])[::-1]]
    synchronize()
    stats = {
        "candidate_count": int(n_candidates),
        "added_count": int(n_add),
        "candidate_abs_residual_mean": float(abs_residual.mean()),
        "candidate_abs_residual_std": float(abs_residual.std(ddof=1)),
        "candidate_abs_residual_max": float(abs_residual.max()),
        "selected_abs_residual_mean": float(abs_residual[ids].mean()),
        "selected_abs_residual_min": float(abs_residual[ids].min()),
        "selected_abs_residual_max": float(abs_residual[ids].max()),
        "selection_time_seconds": float(time.time() - start),
    }
    return x_candidates[ids].copy(), t_candidates[ids].copy(), abs_residual[ids].copy(), stats


def compute_fixed_monitor_loss(model, original_points):

    with torch.enable_grad():
        values = compute_loss_components(model, original_points)
    total, residual, bc, ic_u, ic_w, data = values
    return {
        "fixed_monitor_total_loss": float(total.detach().cpu()),
        "fixed_monitor_residual_loss": float(residual.detach().cpu()),
        "fixed_monitor_data_loss": float(data.detach().cpu()),
        "fixed_monitor_bc_loss": float(bc.detach().cpu()),
        "fixed_monitor_ic_loss_unweighted": float(ic_u.detach().cpu()),
        "fixed_monitor_ic_loss_weighted": float(ic_w.detach().cpu()),
    }


def run_sampling_stage(seed, overwrite=False):
    seed = int(seed)
    seed_dir = METHOD_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    point_path = selected_points_path(seed)
    summary_path = seed_dir / "sampling_stage_summary.json"

    if point_path.exists() and summary_path.exists() and not overwrite:
        print(f"Existing AC RAR point set kept: {point_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    original_np = load_original_points_numpy(seed)
    original_torch = load_training_points(seed)
    original_x, original_t = original_np["x_res"], original_np["t_res"]
    if len(original_x) != N_RES:
        raise RuntimeError(f"Original N_RES={len(original_x)}, expected {N_RES}")

    current_x = original_x.copy()
    current_t = original_t.copy()
    source_round = np.zeros(N_RES, dtype=np.int32)


    if not (np.array_equal(current_x, original_x) and np.array_equal(current_t, original_t)):
        raise RuntimeError("RAR base points differ from original fixed points.")

    set_seed(seed)
    acquisition_model = build_fresh_tanh_model()
    optimizer = torch.optim.Adam(acquisition_model.parameters(), lr=ADAM_LR)

    adam_history, rar_history = [], []
    adam_seconds = 0.0
    rar_seconds = 0.0
    synchronize()
    stage_start = time.time()

    for step in range(1, ADAM_ACQUISITION_STEPS + 1):
        points = make_points_with_residual_set(original_torch, current_x, current_t)
        synchronize()
        opt_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        total, residual, bc, ic_u, ic_w, data = compute_loss_components(acquisition_model, points)
        total.backward()
        optimizer.step()
        synchronize()
        adam_seconds += time.time() - opt_start

        adam_history.append([
            step, len(current_x), float(total.detach().cpu()), float(residual.detach().cpu()),
            float(data.detach().cpu()), float(bc.detach().cpu()), float(ic_u.detach().cpu()),
            float(ic_w.detach().cpu()), adam_seconds, rar_seconds, time.time() - stage_start,
        ])

        if step == 1 or step % 100 == 0:
            print(f"AC RAR sampling | seed {seed:4d} | Adam {step:4d}/2500 | "
                  f"N_RES={len(current_x):4d} | Loss={float(total.detach().cpu()):.6e}")

        if step % RAR_INTERVAL == 0 and step <= RAR_LAST_REFINEMENT_STEP:
            rar_round = step // RAR_INTERVAL
            x_add, t_add, _, stats = select_rar_points(acquisition_model, seed, rar_round)
            rar_seconds += stats["selection_time_seconds"]
            n_before = len(current_x)
            current_x = np.vstack([current_x, x_add])
            current_t = np.vstack([current_t, t_add])
            source_round = np.concatenate([
                source_round,
                np.full(len(x_add), rar_round, dtype=np.int32),
            ])
            rar_history.append({
                "seed": seed, "rar_round": rar_round, "adam_step": step,
                "n_res_before": n_before, "n_res_after": len(current_x), **stats,
            })
            print(f"  RAR round {rar_round}: +{len(x_add)} -> N_RES={len(current_x)} | "
                  f"selected |f| mean={stats['selected_abs_residual_mean']:.3e}")

    synchronize()
    wall = time.time() - stage_start

    if len(current_x) != FINAL_N_RES:
        raise RuntimeError(f"Final N_RES={len(current_x)}, expected {FINAL_N_RES}")
    if not (np.array_equal(current_x[:N_RES], original_x) and np.array_equal(current_t[:N_RES], original_t)):
        raise RuntimeError("The original 3000 points changed during RAR.")

    np.savez_compressed(
        point_path,
        x_res=current_x, t_res=current_t, source_round=source_round,
        original_shared_count=np.asarray([N_RES], dtype=np.int32),
        adaptive_added_count=np.asarray([RAR_ADDED_TOTAL], dtype=np.int32),
        final_n_res=np.asarray([FINAL_N_RES], dtype=np.int32),
        sampling_seed=np.asarray([seed], dtype=np.int32),
    )

    with (seed_dir / "sampling_adam_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "adam_step", "n_res", "adaptive_training_total_loss", "adaptive_training_residual_loss",
            "adaptive_training_data_loss", "adaptive_training_bc_loss",
            "adaptive_training_ic_loss_unweighted", "adaptive_training_ic_loss_weighted",
            "cumulative_adam_optimizer_seconds", "cumulative_rar_selection_seconds",
            "sampling_stage_wall_seconds",
        ])
        writer.writerows(adam_history)

    with (seed_dir / "rar_refinement_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rar_history[0].keys()))
        writer.writeheader(); writer.writerows(rar_history)

    summary = {
        "seed": seed,
        "purpose": "sampling only: original 3000 + 600 RAR points",
        "adam_acquisition_steps": ADAM_ACQUISITION_STEPS,
        "adam_lr": ADAM_LR,
        "original_shared_n_res": N_RES,
        "rar_rounds": RAR_ROUNDS,
        "rar_interval": RAR_INTERVAL,
        "rar_add_per_round": RAR_ADD_PER_ROUND,
        "rar_added_total": RAR_ADDED_TOTAL,
        "rar_candidate_count_per_round": RAR_CANDIDATES,
        "last_refinement_adam_step": RAR_LAST_REFINEMENT_STEP,
        "adam_steps_after_last_refinement": FINAL_ADAM_STEPS_AFTER_LAST_RAR,
        "final_n_res": FINAL_N_RES,
        "n_bc": N_BC, "n_ic": N_IC,
        "adam_optimizer_seconds": adam_seconds,
        "rar_selection_seconds": rar_seconds,
        "sampling_stage_wall_seconds": wall,
        "selected_points_file": str(point_path),
        "acquisition_model_saved": False,
        "acquisition_optimizer_saved": False,
        "formal_lbfgs_warm_start_from_adam": False,
    }
    save_json(summary_path, summary)

    del optimizer, acquisition_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_sampling_all_seeds(overwrite=False):
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_sampling_stage(seed, overwrite=overwrite) for seed in SEEDS]
    save_json(METHOD_DIR / "all_sampling_stage_summaries.json", {
        "method": METHOD, "method_label": METHOD_LABEL, "seeds": SEEDS, "summaries": summaries,
    })


def load_formal_points(seed):
    point_path = selected_points_path(seed)
    if not point_path.exists():
        raise FileNotFoundError(f"Missing selected RAR points: {point_path}")
    with np.load(point_path) as data:
        x_res = np.asarray(data["x_res"], dtype=np.float64).copy()
        t_res = np.asarray(data["t_res"], dtype=np.float64).copy()
    if len(x_res) != FINAL_N_RES:
        raise RuntimeError(f"Selected N_RES={len(x_res)}, expected {FINAL_N_RES}")

    original_np = load_original_points_numpy(seed)
    if not (np.array_equal(x_res[:N_RES], original_np["x_res"]) and
            np.array_equal(t_res[:N_RES], original_np["t_res"])):
        raise RuntimeError("First 3000 RAR points do not match original fixed points.")

    original_torch = load_training_points(seed)
    formal_points = make_points_with_residual_set(original_torch, x_res, t_res)
    return formal_points, original_torch, point_path


def save_formal_checkpoint(path, model, optimizer, seed, target_eval, actual_eval,
                           history, monitor_history, elapsed, monitor_seconds,
                           latest_metrics, config):
    torch.save({
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": int(seed),
        "target_checkpoint_eval": int(target_eval),
        "actual_lbfgs_eval": int(actual_eval),
        "lbfgs_eval": int(actual_eval),
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": np.asarray(history, dtype=np.float64),
        "monitor_history": monitor_history,
        "elapsed_seconds": float(elapsed),
        "monitor_diagnostic_seconds": float(monitor_seconds),
        "core_lbfgs_time_seconds_estimate": float(elapsed - monitor_seconds),
        "latest_metrics": latest_metrics,
        "config": config,
    }, path)


def run_fresh_lbfgs_single_seed(seed):
    seed = int(seed)
    seed_dir = METHOD_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    formal_points, original_points, point_path = load_formal_points(seed)


    model, initial_hash = assert_fresh_initialization_matches_standard(seed)

    config = {
        "method": METHOD, "method_label": METHOD_LABEL, "seed": seed,
        "formal_initial_state_sha256": initial_hash,
        "matches_standard_same_seed_initialization": True,
        "adam_warm_start": False,
        "original_shared_n_res": N_RES, "rar_added_n_res": RAR_ADDED_TOTAL,
        "formal_n_res": FINAL_N_RES, "formal_n_bc": N_BC, "formal_n_ic": N_IC,
        "hidden_layers": HIDDEN_LAYERS, "hidden_neurons": HIDDEN_NEURONS,
        "total_parameters": count_parameters(model),
        "lambda_res": LAMBDA_RES, "lambda_bc": LAMBDA_BC, "lambda_ic": LAMBDA_IC,
        "lbfgs_lr": LBFGS_LR, "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL, "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "selected_points_file": str(point_path),
        "fixed_monitor_points_file": str(training_points_path(seed)),
    }
    save_json(seed_dir / "formal_lbfgs_config.json", config)

    initial_monitor = compute_fixed_monitor_loss(model, original_points)
    save_json(seed_dir / "formal_initial_state.json", {
        "seed": seed, "initial_state_sha256": initial_hash,
        "matches_standard_same_seed_initialization": True,
        "adam_warm_start": False, **initial_monitor,
    })

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL, history_size=LBFGS_HISTORY_SIZE,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn=LBFGS_LINE_SEARCH_FN,
    )

    history, monitor_history, saved = [], [], set()
    eval_count = 0
    latest_metrics = None
    latest_monitor = None
    monitor_seconds = 0.0
    synchronize(); start = time.time()

    def evaluate_monitor(force=False):
        nonlocal latest_monitor, monitor_seconds
        should = (force or eval_count == 1 or eval_count % LBFGS_MONITOR_EVERY == 0 or
                  any(eval_count >= target and target not in saved for target in CHECKPOINT_EVALS))
        if not should:
            return
        synchronize(); tic = time.time()
        latest_monitor = compute_fixed_monitor_loss(model, original_points)
        synchronize(); dt = time.time() - tic
        monitor_seconds += dt
        monitor_history.append({
            "lbfgs_eval": int(eval_count), **latest_monitor,
            "monitor_evaluation_seconds": float(dt),
            "cumulative_monitor_seconds": float(monitor_seconds),
        })

    def maybe_checkpoint():
        for target in CHECKPOINT_EVALS:
            if eval_count >= target and target not in saved:
                evaluate_monitor(force=True)
                elapsed = time.time() - start
                metrics = dict(latest_metrics); metrics.update(latest_monitor)
                path = seed_dir / f"allen_cahn_{METHOD}_seed{seed}_eval{target:05d}.pth"
                save_formal_checkpoint(
                    path, model, optimizer, seed, target, eval_count,
                    history, monitor_history, elapsed, monitor_seconds, metrics, config,
                )
                saved.add(target)
                print(f"Saved passive AC RAR checkpoint: {path} (actual_eval={eval_count})")

    def closure():
        nonlocal eval_count, latest_metrics
        optimizer.zero_grad(set_to_none=True)
        total, residual, bc, ic_u, ic_w, data = compute_loss_components(model, formal_points)
        total.backward()
        eval_count += 1
        synchronize(); elapsed = time.time() - start
        latest_metrics = {
            "optimization_total_loss": float(total.detach().cpu()),
            "optimization_residual_loss": float(residual.detach().cpu()),
            "optimization_data_loss": float(data.detach().cpu()),
            "optimization_bc_loss": float(bc.detach().cpu()),
            "optimization_ic_loss_unweighted": float(ic_u.detach().cpu()),
            "optimization_ic_loss_weighted": float(ic_w.detach().cpu()),
            "elapsed_seconds": float(elapsed),
        }
        evaluate_monitor(force=False)
        history.append([
            eval_count, latest_metrics["optimization_total_loss"],
            latest_metrics["optimization_residual_loss"], latest_metrics["optimization_data_loss"],
            latest_metrics["optimization_bc_loss"], latest_metrics["optimization_ic_loss_unweighted"],
            latest_metrics["optimization_ic_loss_weighted"], elapsed, monitor_seconds,
        ])
        if eval_count == 1 or eval_count % PRINT_EVERY == 0:
            monitor_text = "NA" if latest_monitor is None else f"{latest_monitor['fixed_monitor_total_loss']:.6e}"
            print(f"AC RAR | seed {seed:4d} | L-BFGS eval {eval_count:5d} | "
                  f"Opt={latest_metrics['optimization_total_loss']:.6e} | Monitor={monitor_text}")
        maybe_checkpoint()
        return total

    optimizer.step(closure)
    synchronize(); wall = time.time() - start
    if latest_metrics is None:
        raise RuntimeError(f"No L-BFGS closure evaluation for seed={seed}")

    for target in CHECKPOINT_EVALS:
        if target not in saved:
            evaluate_monitor(force=True)
            metrics = dict(latest_metrics); metrics.update(latest_monitor)
            path = seed_dir / f"allen_cahn_{METHOD}_seed{seed}_eval{target:05d}.pth"
            save_formal_checkpoint(
                path, model, optimizer, seed, target, eval_count,
                history, monitor_history, wall, monitor_seconds, metrics, config,
            )
            saved.add(target)

    with (seed_dir / "lbfgs_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lbfgs_eval", "optimization_total_loss", "optimization_residual_loss",
            "optimization_data_loss", "optimization_bc_loss",
            "optimization_ic_loss_unweighted", "optimization_ic_loss_weighted",
            "elapsed_wall_seconds", "cumulative_monitor_seconds",
        ])
        writer.writerows(history)

    with (seed_dir / "fixed_monitor_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(monitor_history[0].keys()))
        writer.writeheader(); writer.writerows(monitor_history)

    sampling = json.loads((seed_dir / "sampling_stage_summary.json").read_text(encoding="utf-8"))
    summary = {
        "method": METHOD, "method_label": METHOD_LABEL, "seed": seed,
        "formal_initial_state_sha256": initial_hash,
        "matches_standard_same_seed_initialization": True,
        "adam_warm_start": False,
        "original_shared_n_res": N_RES, "rar_added_n_res": RAR_ADDED_TOTAL,
        "formal_n_res": FINAL_N_RES, "formal_n_bc": N_BC, "formal_n_ic": N_IC,
        "lbfgs_target_evals": LBFGS_MAX_EVAL, "lbfgs_actual_evals": eval_count,
        "lbfgs_wall_including_monitor_seconds": wall,
        "lbfgs_monitor_seconds": monitor_seconds,
        "lbfgs_core_seconds_estimate": wall - monitor_seconds,
        "sampling_stage_wall_seconds": sampling["sampling_stage_wall_seconds"],
        "sampling_adam_optimizer_seconds": sampling["adam_optimizer_seconds"],
        "sampling_rar_selection_seconds": sampling["rar_selection_seconds"],
        "end_to_end_cost_seconds": sampling["sampling_stage_wall_seconds"] + wall,
    }
    save_json(seed_dir / "formal_training_summary.json", summary)
    return summary


def run_fresh_lbfgs_all_seeds():
    summaries = [run_fresh_lbfgs_single_seed(seed) for seed in SEEDS]
    save_json(METHOD_DIR / "all_formal_training_summaries.json", {
        "method": METHOD, "method_label": METHOD_LABEL, "seeds": SEEDS, "summaries": summaries,
    })


def load_reference():

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Missing reference file: {REFERENCE_FILE}")
    data = scipy.io.loadmat(REFERENCE_FILE)
    x = np.asarray(data["x"]).flatten()
    if "tt" in data:
        t = np.asarray(data["tt"]).flatten()
    elif "t" in data:
        t = np.asarray(data["t"]).flatten()
    else:
        raise KeyError("AC.mat must contain 'tt' or 't'.")
    if "uu" in data:
        u_exact = np.asarray(data["uu"])
    elif "usol" in data:
        u_exact = np.asarray(data["usol"])
    else:
        raise KeyError("AC.mat must contain 'uu' or 'usol'.")
    target_shape = (len(x), len(t))
    if u_exact.shape != target_shape:
        if u_exact.T.shape == target_shape:
            u_exact = u_exact.T
        else:
            raise ValueError(f"Unexpected reference shape {u_exact.shape}, expected {target_shape}")
    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, u_exact


def predict(model, X, T, output_shape):
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=DTYPE, device=DEVICE)
    t_flat = torch.tensor(T.reshape(-1, 1), dtype=DTYPE, device=DEVICE)
    model.eval(); parts = []
    with torch.no_grad():
        for start in range(0, x_flat.shape[0], 20000):
            end = start + 20000
            parts.append(model(x_flat[start:end], t_flat[start:end]).detach().cpu().numpy())
    return np.vstack(parts).reshape(output_shape)


def relative_l2(exact, pred):
    return float(
        np.linalg.norm((np.asarray(exact) - np.asarray(pred)).reshape(-1), 2)
        / np.linalg.norm(np.asarray(exact).reshape(-1), 2)
    )
