import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from burgers_common_fixed_points_continuous_lbfgs import (
    SEEDS,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    REFERENCE_FILE,
    DEVICE,
    DTYPE,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    N_RES,
    N_BC,
    N_IC,
    X_MIN,
    X_MAX,
    T_MIN,
    T_MAX,
    LBFGS_LR,
    LBFGS_MAX_ITER,
    LBFGS_MAX_EVAL,
    LBFGS_HISTORY_SIZE,
    LBFGS_TOLERANCE_GRAD,
    LBFGS_TOLERANCE_CHANGE,
    LBFGS_LINE_SEARCH_FN,
    CHECKPOINT_EVALS,
    PRINT_EVERY,
    IMAGE_DPI,
    TanhPINN,
    set_seed,
    synchronize,
    load_training_points,
    training_points_path,
    compute_loss_components,
    cpu_state_dict,
    count_parameters,
    save_json,
)

METHOD = "rar_sampling_only_base5000_plus1000"
METHOD_LABEL = "RAR-guided PINN"
METHOD_DIR = RESULT_ROOT / METHOD


ADAM_ACQUISITION_STEPS = 2500
ADAM_LR = 1.0e-3

RAR_INITIAL_N_RES = 5000
RAR_PERMANENT_ANCHORS = 5000
RAR_INTERVAL = 250
RAR_ROUNDS = 8
RAR_ADD_PER_ROUND = 125
RAR_CANDIDATES = 100000
RAR_CANDIDATE_BATCH = 10000

RAR_LAST_REFINEMENT_STEP = RAR_INTERVAL * RAR_ROUNDS
FINAL_N_RES = RAR_INITIAL_N_RES + RAR_ROUNDS * RAR_ADD_PER_ROUND
LEGACY_FINAL_ADAM_STEPS = ADAM_ACQUISITION_STEPS - RAR_LAST_REFINEMENT_STEP

assert RAR_INITIAL_N_RES == 5000
assert RAR_ROUNDS == 8
assert RAR_ADD_PER_ROUND == 125
assert RAR_LAST_REFINEMENT_STEP == 2000
assert LEGACY_FINAL_ADAM_STEPS == 500
assert N_RES == 5000
assert FINAL_N_RES == 6000


SAVE_ACQUISITION_MODEL = False


LBFGS_MONITOR_EVERY = 10
assert LBFGS_MAX_ITER == 5000
assert LBFGS_MAX_EVAL == 5000
assert CHECKPOINT_EVALS == (2500, 5000)


def build_fresh_tanh_model():
    return TanhPINN(
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
    ).double().to(DEVICE)


def tensor_state_sha256(model):

    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode("utf-8"))
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        h.update(array.tobytes())
    return h.hexdigest()


def assert_fresh_initialization_matches_standard(seed):


    set_seed(seed)
    reference_standard = build_fresh_tanh_model()
    expected_hash = tensor_state_sha256(reference_standard)
    expected_state = {
        k: v.detach().cpu().clone()
        for k, v in reference_standard.state_dict().items()
    }
    del reference_standard

    set_seed(seed)
    formal_model = build_fresh_tanh_model()
    actual_hash = tensor_state_sha256(formal_model)

    if expected_hash != actual_hash:
        raise RuntimeError(
            f"Fresh initialization hash mismatch for seed={seed}: "
            f"{expected_hash} != {actual_hash}"
        )

    for key, value in formal_model.state_dict().items():
        if not torch.equal(value.detach().cpu(), expected_state[key]):
            raise RuntimeError(
                f"Fresh initialization tensor mismatch for seed={seed}, key={key}"
            )

    return formal_model, actual_hash


def pde_residual(model, x, t, create_graph):
    u = model(x, t)

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]

    return u_t + u * u_x - (0.01 / np.pi) * u_xx


def make_points(base_points, x_res_np, t_res_np):
    return {
        "x_res": torch.tensor(x_res_np, dtype=DTYPE, device=DEVICE),
        "t_res": torch.tensor(t_res_np, dtype=DTYPE, device=DEVICE),
        "x_bc": base_points["x_bc"],
        "t_bc": base_points["t_bc"],
        "x_ic": base_points["x_ic"],
        "t_ic": base_points["t_ic"],
    }


def evaluate_abs_pde_residual_candidates(
    model,
    x_np,
    t_np,
    batch_size=RAR_CANDIDATE_BATCH,
):
    was_training = model.training
    model.eval()
    pieces = []

    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))

        x = torch.tensor(
            x_np[start:end],
            dtype=DTYPE,
            device=DEVICE,
            requires_grad=True,
        )
        t = torch.tensor(
            t_np[start:end],
            dtype=DTYPE,
            device=DEVICE,
            requires_grad=True,
        )

        with torch.enable_grad():
            residual = pde_residual(
                model,
                x,
                t,
                create_graph=False,
            )

        pieces.append(
            np.abs(residual.detach().cpu().numpy()).reshape(-1)
        )

        del x, t, residual

    if was_training:
        model.train()

    return np.concatenate(pieces, axis=0)


def generate_candidate_pool(seed, rar_round, n_candidates=RAR_CANDIDATES):
    rng = np.random.default_rng(
        int(seed) + 1000003 * int(rar_round) + 7919
    )

    x = rng.uniform(
        X_MIN,
        X_MAX,
        size=(n_candidates, 1),
    ).astype(np.float64)

    t = rng.uniform(
        T_MIN,
        T_MAX,
        size=(n_candidates, 1),
    ).astype(np.float64)

    return x, t


def select_rar_points(
    model,
    seed,
    rar_round,
    n_candidates=RAR_CANDIDATES,
    n_add=RAR_ADD_PER_ROUND,
):
    synchronize()
    start = time.time()

    x_candidates, t_candidates = generate_candidate_pool(
        seed,
        rar_round,
        n_candidates=n_candidates,
    )

    abs_residual = evaluate_abs_pde_residual_candidates(
        model,
        x_candidates,
        t_candidates,
    )

    ids = np.argpartition(abs_residual, -n_add)[-n_add:]
    ids = ids[np.argsort(abs_residual[ids])[::-1]]

    x_add = x_candidates[ids].copy()
    t_add = t_candidates[ids].copy()
    r_add = abs_residual[ids].copy()

    synchronize()
    elapsed = time.time() - start

    return (
        x_add,
        t_add,
        r_add,
        {
            "candidate_count": int(n_candidates),
            "added_count": int(n_add),
            "candidate_abs_residual_mean": float(abs_residual.mean()),
            "candidate_abs_residual_std": float(abs_residual.std(ddof=1)),
            "candidate_abs_residual_max": float(abs_residual.max()),
            "selected_abs_residual_mean": float(r_add.mean()),
            "selected_abs_residual_min": float(r_add.min()),
            "selected_abs_residual_max": float(r_add.max()),
            "selection_time_seconds": float(elapsed),
        },
    )


def compute_fixed_monitor_loss(model, original_points):


    was_training = model.training
    model.eval()

    with torch.enable_grad():
        x_res = (
            original_points["x_res"]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        t_res = (
            original_points["t_res"]
            .detach()
            .clone()
            .requires_grad_(True)
        )

        residual = pde_residual(
            model,
            x_res,
            t_res,
            create_graph=False,
        )
        loss_res = torch.mean(residual.detach() ** 2)

    with torch.no_grad():
        loss_bc = torch.mean(
            model(
                original_points["x_bc"],
                original_points["t_bc"],
            ) ** 2
        )

        u_ic = model(
            original_points["x_ic"],
            original_points["t_ic"],
        )
        u_ic_exact = -torch.sin(
            np.pi * original_points["x_ic"]
        )
        loss_ic = torch.mean((u_ic - u_ic_exact) ** 2)

    if was_training:
        model.train()

    data_loss = loss_bc + loss_ic
    total_loss = loss_res + data_loss

    return {
        "fixed_monitor_total_loss": float(total_loss.detach().cpu()),
        "fixed_monitor_residual_loss": float(loss_res.detach().cpu()),
        "fixed_monitor_data_loss": float(data_loss.detach().cpu()),
        "fixed_monitor_bc_loss": float(loss_bc.detach().cpu()),
        "fixed_monitor_ic_loss": float(loss_ic.detach().cpu()),
    }


def selected_points_path(seed):
    return (
        METHOD_DIR
        / f"seed_{seed}"
        / f"burgers_rar_base5000_plus1000_points_seed{seed}.npz"
    )


def run_sampling_stage(seed, overwrite=False):


    seed = int(seed)

    seed_dir = METHOD_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    out_path = selected_points_path(seed)

    if out_path.exists() and not overwrite:
        print(
            f"Existing selected point set kept: {out_path}\n"
            "Use overwrite=True only if you intentionally want to regenerate it."
        )
        return json.loads(
            (seed_dir / "sampling_stage_summary.json").read_text(
                encoding="utf-8"
            )
        )

    set_seed(seed)

    original_points = load_training_points(seed)

    original_x = (
        original_points["x_res"]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )
    original_t = (
        original_points["t_res"]
        .detach()
        .cpu()
        .numpy()
        .copy()
    )

    if len(original_x) != N_RES:
        raise RuntimeError(
            f"Original Standard point file has {len(original_x)} PDE points; "
            f"expected {N_RES}."
        )


    current_x = original_x.copy()
    current_t = original_t.copy()

    if len(current_x) != RAR_INITIAL_N_RES:
        raise RuntimeError(
            f"RAR initial base contains {len(current_x)} points, "
            f"expected {RAR_INITIAL_N_RES}."
        )

    if not (
        np.array_equal(current_x, original_x)
        and np.array_equal(current_t, original_t)
    ):
        raise RuntimeError(
            "RAR initial 5000 PDE points are not byte-identical to the "
            "original Standard/NMAF-AAT point set."
        )

    source_round = np.zeros(
        RAR_INITIAL_N_RES,
        dtype=np.int32,
    )


    acquisition_model = build_fresh_tanh_model()

    optimizer = torch.optim.Adam(
        acquisition_model.parameters(),
        lr=ADAM_LR,
    )

    adam_history = []
    rar_history = []

    adam_optimizer_seconds = 0.0
    rar_selection_seconds = 0.0

    synchronize()
    stage_start = time.time()

    for step in range(1, ADAM_ACQUISITION_STEPS + 1):
        adaptive_points = make_points(
            original_points,
            current_x,
            current_t,
        )

        synchronize()
        opt_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        (
            total,
            residual,
            bc,
            ic,
            data,
        ) = compute_loss_components(
            acquisition_model,
            adaptive_points,
        )

        total.backward()
        optimizer.step()

        synchronize()
        adam_optimizer_seconds += time.time() - opt_start

        adam_history.append([
            int(step),
            int(len(current_x)),
            float(total.detach().cpu()),
            float(residual.detach().cpu()),
            float(data.detach().cpu()),
            float(bc.detach().cpu()),
            float(ic.detach().cpu()),
            float(adam_optimizer_seconds),
            float(rar_selection_seconds),
            float(time.time() - stage_start),
        ])

        if step == 1 or step % 100 == 0:
            print(
                f"RAR sampling | seed {seed:4d} | "
                f"Adam {step:4d}/{ADAM_ACQUISITION_STEPS} | "
                f"N_RES={len(current_x):4d} | "
                f"Loss={float(total.detach().cpu()):.6e}"
            )

        if (
            step % RAR_INTERVAL == 0
            and step <= RAR_LAST_REFINEMENT_STEP
        ):
            rar_round = step // RAR_INTERVAL

            (
                x_add,
                t_add,
                selected_r,
                stats,
            ) = select_rar_points(
                acquisition_model,
                seed,
                rar_round,
            )

            rar_selection_seconds += stats[
                "selection_time_seconds"
            ]

            n_before = len(current_x)

            current_x = np.vstack([current_x, x_add])
            current_t = np.vstack([current_t, t_add])

            source_round = np.concatenate([
                source_round,
                np.full(
                    len(x_add),
                    rar_round,
                    dtype=np.int32,
                ),
            ])

            rar_history.append({
                "seed": int(seed),
                "rar_round": int(rar_round),
                "adam_step": int(step),
                "n_res_before": int(n_before),
                "n_res_after": int(len(current_x)),
                **stats,
            })

            print(
                f"  RAR round {rar_round:2d}: "
                f"+{len(x_add)} -> N_RES={len(current_x)} | "
                f"selected |r| mean={stats['selected_abs_residual_mean']:.3e}"
            )

    synchronize()
    stage_wall_seconds = time.time() - stage_start


    if len(current_x) != FINAL_N_RES:
        raise RuntimeError(
            f"Final selected PDE count={len(current_x)}, expected {FINAL_N_RES}."
        )

    if not (
        np.array_equal(
            current_x[:RAR_PERMANENT_ANCHORS],
            original_x[:RAR_PERMANENT_ANCHORS],
        )
        and np.array_equal(
            current_t[:RAR_PERMANENT_ANCHORS],
            original_t[:RAR_PERMANENT_ANCHORS],
        )
    ):
        raise RuntimeError(
            "The 5000 permanent uniform anchors changed unexpectedly."
        )


    np.savez_compressed(
        out_path,
        x_res=current_x,
        t_res=current_t,
        source_round=source_round,
        permanent_anchor_count=np.asarray(
            [RAR_PERMANENT_ANCHORS],
            dtype=np.int32,
        ),
        final_n_res=np.asarray(
            [len(current_x)],
            dtype=np.int32,
        ),
        sampling_seed=np.asarray(
            [seed],
            dtype=np.int32,
        ),
    )

    with (seed_dir / "sampling_adam_history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "adam_step",
            "n_res",
            "adaptive_training_total_loss",
            "adaptive_training_residual_loss",
            "adaptive_training_data_loss",
            "adaptive_training_bc_loss",
            "adaptive_training_ic_loss",
            "cumulative_adam_optimizer_seconds",
            "cumulative_rar_selection_seconds",
            "sampling_stage_wall_seconds",
        ])
        writer.writerows(adam_history)

    with (seed_dir / "rar_refinement_history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rar_history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rar_history)

    summary = {
        "seed": int(seed),
        "purpose": "strong collocation design only: retain original 5000 + add 1000 RAR points",
        "adam_acquisition_steps": int(ADAM_ACQUISITION_STEPS),
        "adam_lr": float(ADAM_LR),
        "initial_uniform_anchors": int(RAR_INITIAL_N_RES),
        "rar_rounds": int(RAR_ROUNDS),
        "rar_interval": int(RAR_INTERVAL),
        "rar_add_per_round": int(RAR_ADD_PER_ROUND),
        "rar_candidates_per_round": int(RAR_CANDIDATES),
        "last_refinement_step": int(RAR_LAST_REFINEMENT_STEP),
        "legacy_final_adam_steps_after_last_refinement": int(
            LEGACY_FINAL_ADAM_STEPS
        ),
        "final_n_res": int(len(current_x)),
        "n_bc": int(N_BC),
        "n_ic": int(N_IC),
        "adam_optimizer_seconds": float(adam_optimizer_seconds),
        "rar_selection_seconds": float(rar_selection_seconds),
        "sampling_stage_wall_seconds": float(stage_wall_seconds),
        "selected_points_file": str(out_path),
        "acquisition_model_saved": False,
        "acquisition_optimizer_saved": False,
        "formal_lbfgs_warm_start_from_adam": False,
    }

    save_json(
        seed_dir / "sampling_stage_summary.json",
        summary,
    )


    del optimizer
    del acquisition_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


    forbidden = [
        seed_dir / "adam_rar_pretrained_model_final.pth",
        seed_dir / "acquisition_model.pth",
        seed_dir / "adam_model.pth",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError(
            "An acquisition-model checkpoint exists. "
            "Sampling-only protocol forbids carrying model information."
        )

    print(
        f"Sampling-only RAR seed {seed} finished. "
        f"Saved points only: {out_path}"
    )

    return summary


def run_sampling_all_seeds(overwrite=False):
    METHOD_DIR.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_sampling_stage(seed, overwrite=overwrite)
        for seed in SEEDS
    ]

    save_json(
        METHOD_DIR / "all_sampling_stage_summaries.json",
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "seeds": SEEDS,
            "summaries": summaries,
        },
    )


def load_selected_formal_points(seed):
    path = selected_points_path(seed)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing selected RAR points: {path}\n"
            "Run select_burgers_rar_points_adam2500.py first."
        )

    data = np.load(path)

    x_res = np.asarray(data["x_res"], dtype=np.float64)
    t_res = np.asarray(data["t_res"], dtype=np.float64)

    if len(x_res) != FINAL_N_RES:
        raise RuntimeError(
            f"Selected point set has N_RES={len(x_res)}, expected {FINAL_N_RES}."
        )

    base_points = load_training_points(seed)

    return (
        make_points(
            base_points,
            x_res,
            t_res,
        ),
        base_points,
        path,
    )


def save_formal_checkpoint(
    path,
    *,
    model,
    optimizer,
    seed,
    target_eval,
    actual_eval,
    history,
    monitor_history,
    elapsed_wall,
    monitor_time,
    latest_metrics,
    config,
):
    torch.save(
        {
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
            "elapsed_seconds": float(elapsed_wall),
            "monitor_diagnostic_time_seconds": float(monitor_time),
            "core_lbfgs_time_seconds_estimate": float(
                elapsed_wall - monitor_time
            ),
            "latest_metrics": latest_metrics,
            "config": config,
        },
        path,
    )


def run_fresh_lbfgs_single_seed(seed):
    seed = int(seed)

    seed_dir = METHOD_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    optimization_points, original_points, point_path = (
        load_selected_formal_points(seed)
    )


    model, initial_state_hash = (
        assert_fresh_initialization_matches_standard(seed)
    )

    config = {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": int(seed),
        "formal_network": "fresh exact TanhPINN",
        "formal_initialization": (
            "set_seed(seed) then fresh TanhPINN; "
            "byte-identical to Standard-PINN same-seed initialization"
        ),
        "formal_initial_state_sha256": initial_state_hash,
        "acquisition_model_loaded": False,
        "acquisition_optimizer_loaded": False,
        "adam_warm_start": False,
        "selected_points_file": str(point_path),
        "formal_n_res": int(FINAL_N_RES),
        "original_shared_base_n_res": int(N_RES),
        "formal_n_bc": int(N_BC),
        "formal_n_ic": int(N_IC),
        "hidden_layers": int(HIDDEN_LAYERS),
        "hidden_neurons": int(HIDDEN_NEURONS),
        "total_parameters": int(count_parameters(model)),
        "lbfgs_lr": float(LBFGS_LR),
        "lbfgs_max_iter": int(LBFGS_MAX_ITER),
        "lbfgs_max_eval": int(LBFGS_MAX_EVAL),
        "lbfgs_history_size": int(LBFGS_HISTORY_SIZE),
        "lbfgs_tolerance_grad": float(LBFGS_TOLERANCE_GRAD),
        "lbfgs_tolerance_change": float(LBFGS_TOLERANCE_CHANGE),
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "optimizer": "single continuous LBFGS",
        "optimizer_restart": False,
        "fixed_monitor_points_file": str(training_points_path(seed)),
        "lbfgs_monitor_every": int(LBFGS_MONITOR_EVERY),
    }

    save_json(
        seed_dir / "formal_lbfgs_config.json",
        config,
    )


    initial_monitor = compute_fixed_monitor_loss(
        model,
        original_points,
    )

    save_json(
        seed_dir / "formal_initial_state.json",
        {
            "seed": int(seed),
            "initial_state_sha256": initial_state_hash,
            "matches_standard_same_seed_initialization": True,
            "adam_warm_start": False,
            **initial_monitor,
        },
    )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn=LBFGS_LINE_SEARCH_FN,
    )

    history = []
    monitor_history = []
    saved = set()

    eval_count = 0
    latest_metrics = None
    latest_monitor = None
    monitor_time = 0.0

    synchronize()
    start = time.time()

    def evaluate_monitor(force=False):
        nonlocal latest_monitor, monitor_time

        should = (
            force
            or eval_count == 1
            or eval_count % LBFGS_MONITOR_EVERY == 0
            or any(
                eval_count >= target
                and target not in saved
                for target in CHECKPOINT_EVALS
            )
        )

        if not should:
            return

        synchronize()
        mon_start = time.time()

        latest_monitor = compute_fixed_monitor_loss(
            model,
            original_points,
        )

        synchronize()
        elapsed = time.time() - mon_start
        monitor_time += elapsed

        monitor_history.append({
            "lbfgs_eval": int(eval_count),
            **latest_monitor,
            "monitor_evaluation_seconds": float(elapsed),
            "cumulative_monitor_seconds": float(monitor_time),
        })

    def maybe_checkpoint():
        for target in CHECKPOINT_EVALS:
            if eval_count >= target and target not in saved:
                evaluate_monitor(force=True)

                elapsed_wall = time.time() - start

                metrics = dict(latest_metrics)
                metrics.update(latest_monitor)

                path = (
                    seed_dir
                    / f"burgers_{METHOD}_seed{seed}_eval{target:05d}.pth"
                )

                save_formal_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    seed=seed,
                    target_eval=target,
                    actual_eval=eval_count,
                    history=history,
                    monitor_history=monitor_history,
                    elapsed_wall=elapsed_wall,
                    monitor_time=monitor_time,
                    latest_metrics=metrics,
                    config=config,
                )

                saved.add(target)

                print(
                    f"Saved passive RAR formal checkpoint: {path} "
                    f"(actual_eval={eval_count})"
                )

    def closure():
        nonlocal eval_count, latest_metrics

        optimizer.zero_grad(set_to_none=True)

        (
            total,
            residual,
            bc,
            ic,
            data,
        ) = compute_loss_components(
            model,
            optimization_points,
        )

        total.backward()
        eval_count += 1

        synchronize()
        elapsed = time.time() - start

        latest_metrics = {
            "optimization_total_loss": float(total.detach().cpu()),
            "optimization_residual_loss": float(residual.detach().cpu()),
            "optimization_data_loss": float(data.detach().cpu()),
            "optimization_bc_loss": float(bc.detach().cpu()),
            "optimization_ic_loss": float(ic.detach().cpu()),
            "elapsed_seconds": float(elapsed),
        }


        evaluate_monitor(force=False)

        history.append([
            int(eval_count),
            latest_metrics["optimization_total_loss"],
            latest_metrics["optimization_residual_loss"],
            latest_metrics["optimization_data_loss"],
            latest_metrics["optimization_bc_loss"],
            latest_metrics["optimization_ic_loss"],
            float(elapsed),
            float(monitor_time),
        ])

        if eval_count == 1 or eval_count % PRINT_EVERY == 0:
            monitor_text = (
                "NA"
                if latest_monitor is None
                else f"{latest_monitor['fixed_monitor_total_loss']:.6e}"
            )

            print(
                f"RAR sampling-only | seed {seed:4d} | "
                f"L-BFGS eval {eval_count:5d} | "
                f"Opt={latest_metrics['optimization_total_loss']:.6e} | "
                f"CommonMonitor={monitor_text}"
            )

        maybe_checkpoint()

        return total

    optimizer.step(closure)

    synchronize()
    wall_seconds = time.time() - start

    if latest_metrics is None:
        raise RuntimeError(
            f"No L-BFGS closure evaluation for seed={seed}."
        )


    for target in CHECKPOINT_EVALS:
        if target not in saved:
            evaluate_monitor(force=True)

            metrics = dict(latest_metrics)
            metrics.update(latest_monitor)

            path = (
                seed_dir
                / f"burgers_{METHOD}_seed{seed}_eval{target:05d}.pth"
            )

            save_formal_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                seed=seed,
                target_eval=target,
                actual_eval=eval_count,
                history=history,
                monitor_history=monitor_history,
                elapsed_wall=wall_seconds,
                monitor_time=monitor_time,
                latest_metrics=metrics,
                config=config,
            )

    with (seed_dir / "lbfgs_history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "lbfgs_eval",
            "optimization_total_loss",
            "optimization_residual_loss",
            "optimization_data_loss",
            "optimization_bc_loss",
            "optimization_ic_loss",
            "elapsed_wall_seconds",
            "cumulative_monitor_seconds",
        ])
        writer.writerows(history)

    with (seed_dir / "fixed_monitor_history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(monitor_history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(monitor_history)

    sampling_summary = json.loads(
        (seed_dir / "sampling_stage_summary.json").read_text(
            encoding="utf-8"
        )
    )

    summary = {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": int(seed),
        "formal_initial_state_sha256": initial_state_hash,
        "matches_standard_same_seed_initialization": True,
        "adam_warm_start": False,
        "formal_n_res": int(FINAL_N_RES),
        "original_shared_base_n_res": int(N_RES),
        "formal_n_bc": int(N_BC),
        "formal_n_ic": int(N_IC),
        "lbfgs_target_evals": int(LBFGS_MAX_EVAL),
        "lbfgs_actual_evals": int(eval_count),
        "lbfgs_wall_including_monitor_seconds": float(wall_seconds),
        "lbfgs_monitor_seconds": float(monitor_time),
        "lbfgs_core_seconds_estimate": float(
            wall_seconds - monitor_time
        ),
        "sampling_stage_wall_seconds": float(
            sampling_summary["sampling_stage_wall_seconds"]
        ),
        "sampling_adam_optimizer_seconds": float(
            sampling_summary["adam_optimizer_seconds"]
        ),
        "sampling_rar_selection_seconds": float(
            sampling_summary["rar_selection_seconds"]
        ),
        "end_to_end_cost_seconds": float(
            sampling_summary["sampling_stage_wall_seconds"]
            + wall_seconds
        ),
    }

    save_json(
        seed_dir / "formal_training_summary.json",
        summary,
    )

    return summary


def run_fresh_lbfgs_all_seeds():
    summaries = [
        run_fresh_lbfgs_single_seed(seed)
        for seed in SEEDS
    ]

    save_json(
        METHOD_DIR / "all_formal_training_summaries.json",
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "seeds": SEEDS,
            "summaries": summaries,
        },
    )


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Missing reference file: {REFERENCE_FILE}."
        )

    data = scipy.io.loadmat(REFERENCE_FILE)

    x = np.asarray(data["x"]).flatten()
    t = np.asarray(data["t"]).flatten()
    exact = np.asarray(data["usol"])

    target_shape = (len(x), len(t))

    if exact.shape != target_shape:
        if exact.T.shape == target_shape:
            exact = exact.T
        else:
            raise ValueError(
                f"Unexpected reference shape: {exact.shape}; "
                f"expected {target_shape}."
            )

    X, T = np.meshgrid(
        x,
        t,
        indexing="ij",
    )

    return x, t, X, T, exact


def predict(model, X, T, output_shape):
    x_flat = torch.tensor(
        X.reshape(-1, 1),
        dtype=DTYPE,
        device=DEVICE,
    )
    t_flat = torch.tensor(
        T.reshape(-1, 1),
        dtype=DTYPE,
        device=DEVICE,
    )

    model.eval()
    parts = []

    with torch.no_grad():
        for start in range(
            0,
            x_flat.shape[0],
            20000,
        ):
            end = start + 20000
            parts.append(
                model(
                    x_flat[start:end],
                    t_flat[start:end],
                )
                .detach()
                .cpu()
                .numpy()
            )

    return np.vstack(parts).reshape(output_shape)


def relative_l2(exact, pred):
    return float(
        np.linalg.norm(
            (exact - pred).reshape(-1),
            2,
        )
        / np.linalg.norm(
            exact.reshape(-1),
            2,
        )
    )
