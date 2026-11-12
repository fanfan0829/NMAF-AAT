import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch

from nlse_common_fixed_points_continuous_lbfgs import (
    SEEDS,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    REFERENCE_FILE,
    DEVICE,
    DTYPE,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    OUTPUT_DIM,
    N_RES,
    N_BC,
    N_IC,
    X_MIN,
    X_MAX,
    T_MIN,
    T_MAX,
    LAMBDA_RES,
    LAMBDA_BC,
    LAMBDA_IC,
    LBFGS_LR,
    LBFGS_MAX_EVAL,
    LBFGS_MAX_ITER,
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


METHOD = "rar_sampling_only_base10000_plus2000"
METHOD_LABEL = "RAR-guided PINN"
METHOD_DIR = RESULT_ROOT / METHOD


ADAM_ACQUISITION_STEPS = 2500
ADAM_LR = 1.0e-3

RAR_INITIAL_N_RES = N_RES
RAR_PERMANENT_ANCHORS = N_RES
RAR_INTERVAL = 250
RAR_ROUNDS = 8
RAR_ADD_PER_ROUND = 250
RAR_CANDIDATES = 100000
RAR_CANDIDATE_BATCH = 10000

RAR_LAST_REFINEMENT_STEP = RAR_INTERVAL * RAR_ROUNDS
RAR_ADDED_TOTAL = RAR_ROUNDS * RAR_ADD_PER_ROUND
FINAL_N_RES = RAR_INITIAL_N_RES + RAR_ADDED_TOTAL
FINAL_ADAM_STEPS_AFTER_LAST_RAR = (
    ADAM_ACQUISITION_STEPS - RAR_LAST_REFINEMENT_STEP
)

assert N_RES == 10000
assert RAR_INITIAL_N_RES == 10000
assert RAR_ADDED_TOTAL == 2000
assert FINAL_N_RES == 12000
assert RAR_LAST_REFINEMENT_STEP == 2000
assert FINAL_ADAM_STEPS_AFTER_LAST_RAR == 500

SAVE_ACQUISITION_MODEL = False


LBFGS_MONITOR_EVERY = 10

assert LBFGS_MAX_ITER == 10000
assert LBFGS_MAX_EVAL == 10000
assert CHECKPOINT_EVALS == (3000, 6000, 10000)


def build_fresh_tanh_model():
    return TanhPINN(
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
    ).double().to(DEVICE)


def tensor_state_sha256(model):
    h = hashlib.sha256()

    for key, value in model.state_dict().items():
        h.update(key.encode("utf-8"))

        array = np.ascontiguousarray(
            value.detach().cpu().numpy()
        )

        h.update(array.tobytes())

    return h.hexdigest()


def assert_fresh_initialization_matches_standard(seed):


    set_seed(seed)

    reference_standard = build_fresh_tanh_model()

    expected_hash = tensor_state_sha256(
        reference_standard
    )

    expected_state = {
        key: value.detach().cpu().clone()
        for key, value
        in reference_standard.state_dict().items()
    }

    del reference_standard

    set_seed(seed)

    formal_model = build_fresh_tanh_model()

    actual_hash = tensor_state_sha256(
        formal_model
    )

    if expected_hash != actual_hash:
        raise RuntimeError(
            f"Fresh initialization hash mismatch for seed={seed}: "
            f"{expected_hash} != {actual_hash}"
        )

    for key, value in formal_model.state_dict().items():
        if not torch.equal(
            value.detach().cpu(),
            expected_state[key],
        ):
            raise RuntimeError(
                f"Fresh initialization tensor mismatch: "
                f"seed={seed}, key={key}"
            )

    return formal_model, actual_hash


def selected_points_path(seed):
    return (
        METHOD_DIR
        / f"seed_{seed}"
        / (
            f"nlse_rar_base10000_plus2000_points"
            f"_seed{seed}.npz"
        )
    )


def load_original_points_numpy(seed):
    path = training_points_path(seed)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing original NLSE fixed points: {path}\n"
            "Run generate_nlse_fixed_training_points_seeds_99_1234_2026.py first."
        )

    with np.load(path) as data:
        return {
            key: np.asarray(
                data[key],
                dtype=np.float64,
            ).copy()
            for key in [
                "x_res",
                "t_res",
                "x_bc_l",
                "x_bc_r",
                "t_bc",
                "x_ic",
                "t_ic",
            ]
        }


def make_points_with_residual_set(
    original_points_torch,
    x_res_np,
    t_res_np,
):
    return {
        "x_res": torch.tensor(
            x_res_np,
            dtype=DTYPE,
            device=DEVICE,
        ),
        "t_res": torch.tensor(
            t_res_np,
            dtype=DTYPE,
            device=DEVICE,
        ),
        "x_bc_l": original_points_torch["x_bc_l"],
        "x_bc_r": original_points_torch["x_bc_r"],
        "t_bc": original_points_torch["t_bc"],
        "x_ic": original_points_torch["x_ic"],
        "t_ic": original_points_torch["t_ic"],
    }


def pde_residual_components(
    model,
    x,
    t,
    create_graph,
):


    sol = model(
        x,
        t,
    )

    u = sol[:, 0:1]
    v = sol[:, 1:2]

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
        retain_graph=True,
    )[0]

    v_t = torch.autograd.grad(
        v,
        t,
        grad_outputs=torch.ones_like(v),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    v_x = torch.autograd.grad(
        v,
        x,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0]

    v_xx = torch.autograd.grad(
        v_x,
        x,
        grad_outputs=torch.ones_like(v_x),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    magnitude_squared = (
        u ** 2
        + v ** 2
    )

    residual_real = (
        -v_t
        + 0.5 * u_xx
        + magnitude_squared * u
    )

    residual_imag = (
        u_t
        + 0.5 * v_xx
        + magnitude_squared * v
    )

    return (
        residual_real,
        residual_imag,
    )


def evaluate_candidate_residual_score(
    model,
    x_np,
    t_np,
    batch_size=RAR_CANDIDATE_BATCH,
):


    was_training = model.training
    model.eval()

    pieces = []

    for start in range(
        0,
        len(x_np),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(x_np),
        )

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
            f_real, f_imag = (
                pde_residual_components(
                    model,
                    x,
                    t,
                    create_graph=False,
                )
            )

        score = torch.sqrt(
            f_real.detach() ** 2
            + f_imag.detach() ** 2
        )

        pieces.append(
            score.cpu().numpy().reshape(-1)
        )

        del (
            x,
            t,
            f_real,
            f_imag,
            score,
        )

    if was_training:
        model.train()

    return np.concatenate(
        pieces,
        axis=0,
    )


def generate_candidate_pool(
    seed,
    rar_round,
    n_candidates=RAR_CANDIDATES,
):


    rng = np.random.default_rng(
        int(seed)
        + 1000003 * int(rar_round)
        + 7919
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

    x_candidates, t_candidates = (
        generate_candidate_pool(
            seed,
            rar_round,
            n_candidates=n_candidates,
        )
    )

    residual_score = (
        evaluate_candidate_residual_score(
            model,
            x_candidates,
            t_candidates,
        )
    )

    ids = np.argpartition(
        residual_score,
        -n_add,
    )[-n_add:]

    ids = ids[
        np.argsort(
            residual_score[ids]
        )[::-1]
    ]

    x_add = x_candidates[
        ids
    ].copy()

    t_add = t_candidates[
        ids
    ].copy()

    score_add = residual_score[
        ids
    ].copy()

    synchronize()
    elapsed = time.time() - start

    stats = {
        "candidate_count": int(
            n_candidates
        ),
        "added_count": int(
            n_add
        ),
        "candidate_residual_score_mean": float(
            residual_score.mean()
        ),
        "candidate_residual_score_std": float(
            residual_score.std(ddof=1)
        ),
        "candidate_residual_score_max": float(
            residual_score.max()
        ),
        "selected_residual_score_mean": float(
            score_add.mean()
        ),
        "selected_residual_score_min": float(
            score_add.min()
        ),
        "selected_residual_score_max": float(
            score_add.max()
        ),
        "selection_time_seconds": float(
            elapsed
        ),
    }

    return (
        x_add,
        t_add,
        score_add,
        stats,
    )


def compute_fixed_monitor_loss(
    model,
    original_points,
):


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

        f_real, f_imag = (
            pde_residual_components(
                model,
                x_res,
                t_res,
                create_graph=False,
            )
        )

        loss_residual = (
            torch.mean(
                f_real.detach() ** 2
            )
            + torch.mean(
                f_imag.detach() ** 2
            )
        )


    with torch.enable_grad():
        x_bc_l = (
            original_points["x_bc_l"]
            .detach()
            .clone()
            .requires_grad_(True)
        )

        x_bc_r = (
            original_points["x_bc_r"]
            .detach()
            .clone()
            .requires_grad_(True)
        )

        t_bc = original_points[
            "t_bc"
        ]

        sol_l = model(
            x_bc_l,
            t_bc,
        )

        sol_r = model(
            x_bc_r,
            t_bc,
        )

        u_l = sol_l[:, 0:1]
        v_l = sol_l[:, 1:2]

        u_r = sol_r[:, 0:1]
        v_r = sol_r[:, 1:2]

        u_x_l = torch.autograd.grad(
            u_l,
            x_bc_l,
            grad_outputs=torch.ones_like(u_l),
            create_graph=False,
            retain_graph=True,
        )[0]

        u_x_r = torch.autograd.grad(
            u_r,
            x_bc_r,
            grad_outputs=torch.ones_like(u_r),
            create_graph=False,
            retain_graph=True,
        )[0]

        v_x_l = torch.autograd.grad(
            v_l,
            x_bc_l,
            grad_outputs=torch.ones_like(v_l),
            create_graph=False,
            retain_graph=True,
        )[0]

        v_x_r = torch.autograd.grad(
            v_r,
            x_bc_r,
            grad_outputs=torch.ones_like(v_r),
            create_graph=False,
            retain_graph=True,
        )[0]

        loss_bc = (
            torch.mean(
                (
                    u_l.detach()
                    - u_r.detach()
                ) ** 2
            )
            + torch.mean(
                (
                    v_l.detach()
                    - v_r.detach()
                ) ** 2
            )
            + torch.mean(
                (
                    u_x_l.detach()
                    - u_x_r.detach()
                ) ** 2
            )
            + torch.mean(
                (
                    v_x_l.detach()
                    - v_x_r.detach()
                ) ** 2
            )
        )


    with torch.no_grad():
        x_ic = original_points[
            "x_ic"
        ]

        t_ic = original_points[
            "t_ic"
        ]

        sol_ic = model(
            x_ic,
            t_ic,
        )

        u_ic_pred = sol_ic[:, 0:1]
        v_ic_pred = sol_ic[:, 1:2]

        u_ic_exact = (
            2.0
            / torch.cosh(
                x_ic
            )
        )

        v_ic_exact = torch.zeros_like(
            x_ic
        )

        loss_ic_unweighted = (
            torch.mean(
                (
                    u_ic_pred
                    - u_ic_exact
                ) ** 2
            )
            + torch.mean(
                (
                    v_ic_pred
                    - v_ic_exact
                ) ** 2
            )
        )

    loss_ic_weighted = (
        LAMBDA_IC
        * loss_ic_unweighted
    )

    data_loss = (
        LAMBDA_BC
        * loss_bc
        + loss_ic_weighted
    )

    total_loss = (
        LAMBDA_RES
        * loss_residual
        + data_loss
    )

    if was_training:
        model.train()

    return {
        "fixed_monitor_total_loss": float(
            total_loss.detach().cpu()
        ),
        "fixed_monitor_residual_loss": float(
            loss_residual.detach().cpu()
        ),
        "fixed_monitor_data_loss": float(
            data_loss.detach().cpu()
        ),
        "fixed_monitor_bc_loss": float(
            loss_bc.detach().cpu()
        ),
        "fixed_monitor_ic_loss_unweighted": float(
            loss_ic_unweighted.detach().cpu()
        ),
        "fixed_monitor_ic_loss_weighted": float(
            loss_ic_weighted.detach().cpu()
        ),
    }


def run_sampling_stage(
    seed,
    overwrite=False,
):
    seed = int(seed)

    seed_dir = (
        METHOD_DIR
        / f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    point_path = selected_points_path(
        seed
    )

    summary_path = (
        seed_dir
        / "sampling_stage_summary.json"
    )

    if (
        point_path.exists()
        and summary_path.exists()
        and not overwrite
    ):
        print(
            f"Existing NLSE RAR point set kept: {point_path}"
        )

        return json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

    original_np = (
        load_original_points_numpy(
            seed
        )
    )

    original_torch = (
        load_training_points(
            seed
        )
    )

    original_x = original_np[
        "x_res"
    ]

    original_t = original_np[
        "t_res"
    ]

    if len(
        original_x
    ) != N_RES:
        raise RuntimeError(
            f"Original NLSE residual count={len(original_x)}, "
            f"expected {N_RES}."
        )


    current_x = original_x.copy()
    current_t = original_t.copy()

    if not (
        np.array_equal(
            current_x,
            original_x,
        )
        and np.array_equal(
            current_t,
            original_t,
        )
    ):
        raise RuntimeError(
            "RAR base points are not byte-identical to "
            "the original Standard/NMAF-AAT point set."
        )

    source_round = np.zeros(
        RAR_INITIAL_N_RES,
        dtype=np.int32,
    )


    set_seed(seed)

    acquisition_model = (
        build_fresh_tanh_model()
    )

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

    for step in range(
        1,
        ADAM_ACQUISITION_STEPS + 1,
    ):
        adaptive_points = (
            make_points_with_residual_set(
                original_torch,
                current_x,
                current_t,
            )
        )

        synchronize()
        opt_start = time.time()

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            total_loss,
            loss_residual,
            loss_bc,
            loss_ic_unweighted,
            loss_ic_weighted,
            data_loss,
        ) = compute_loss_components(
            acquisition_model,
            adaptive_points,
        )

        total_loss.backward()
        optimizer.step()

        synchronize()

        adam_optimizer_seconds += (
            time.time()
            - opt_start
        )

        adam_history.append([
            int(step),
            int(len(current_x)),
            float(
                total_loss.detach().cpu()
            ),
            float(
                loss_residual.detach().cpu()
            ),
            float(
                data_loss.detach().cpu()
            ),
            float(
                loss_bc.detach().cpu()
            ),
            float(
                loss_ic_unweighted.detach().cpu()
            ),
            float(
                loss_ic_weighted.detach().cpu()
            ),
            float(
                adam_optimizer_seconds
            ),
            float(
                rar_selection_seconds
            ),
            float(
                time.time()
                - stage_start
            ),
        ])

        if (
            step == 1
            or step % 100 == 0
        ):
            print(
                f"NLSE RAR sampling | "
                f"seed {seed:4d} | "
                f"Adam {step:4d}/{ADAM_ACQUISITION_STEPS} | "
                f"N_RES={len(current_x):5d} | "
                f"Loss={float(total_loss.detach().cpu()):.6e}"
            )

        if (
            step % RAR_INTERVAL == 0
            and step
            <= RAR_LAST_REFINEMENT_STEP
        ):
            rar_round = (
                step
                // RAR_INTERVAL
            )

            (
                x_add,
                t_add,
                selected_score,
                stats,
            ) = select_rar_points(
                acquisition_model,
                seed,
                rar_round,
            )

            rar_selection_seconds += (
                stats[
                    "selection_time_seconds"
                ]
            )

            n_before = len(
                current_x
            )

            current_x = np.vstack([
                current_x,
                x_add,
            ])

            current_t = np.vstack([
                current_t,
                t_add,
            ])

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
                "rar_round": int(
                    rar_round
                ),
                "adam_step": int(
                    step
                ),
                "n_res_before": int(
                    n_before
                ),
                "n_res_after": int(
                    len(current_x)
                ),
                **stats,
            })

            print(
                f"  RAR round {rar_round:2d}: "
                f"+{len(x_add)} -> "
                f"N_RES={len(current_x)} | "
                f"selected rho mean="
                f"{stats['selected_residual_score_mean']:.3e}"
            )

    synchronize()

    sampling_wall_seconds = (
        time.time()
        - stage_start
    )


    if len(
        current_x
    ) != FINAL_N_RES:
        raise RuntimeError(
            f"Final NLSE RAR N_RES={len(current_x)}, "
            f"expected {FINAL_N_RES}."
        )

    if not (
        np.array_equal(
            current_x[
                :N_RES
            ],
            original_x,
        )
        and np.array_equal(
            current_t[
                :N_RES
            ],
            original_t,
        )
    ):
        raise RuntimeError(
            "The first 10000 RAR points are not identical "
            "to the original shared NLSE residual points."
        )

    np.savez_compressed(
        point_path,
        x_res=current_x,
        t_res=current_t,
        source_round=source_round,
        original_shared_count=np.asarray(
            [N_RES],
            dtype=np.int32,
        ),
        adaptive_added_count=np.asarray(
            [RAR_ADDED_TOTAL],
            dtype=np.int32,
        ),
        final_n_res=np.asarray(
            [FINAL_N_RES],
            dtype=np.int32,
        ),
        sampling_seed=np.asarray(
            [seed],
            dtype=np.int32,
        ),
    )

    with (
        seed_dir
        / "sampling_adam_history.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow([
            "adam_step",
            "n_res",
            "adaptive_training_total_loss",
            "adaptive_training_residual_loss",
            "adaptive_training_data_loss",
            "adaptive_training_bc_loss",
            "adaptive_training_ic_loss_unweighted",
            "adaptive_training_ic_loss_weighted",
            "cumulative_adam_optimizer_seconds",
            "cumulative_rar_selection_seconds",
            "sampling_stage_wall_seconds",
        ])

        writer.writerows(
            adam_history
        )

    with (
        seed_dir
        / "rar_refinement_history.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rar_history[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rar_history
        )

    summary = {
        "seed": int(seed),
        "purpose": (
            "sampling design only; original 10000 retained "
            "+ 2000 residual-based points"
        ),
        "adam_acquisition_steps": int(
            ADAM_ACQUISITION_STEPS
        ),
        "adam_lr": float(
            ADAM_LR
        ),
        "original_shared_n_res": int(
            N_RES
        ),
        "rar_rounds": int(
            RAR_ROUNDS
        ),
        "rar_interval": int(
            RAR_INTERVAL
        ),
        "rar_add_per_round": int(
            RAR_ADD_PER_ROUND
        ),
        "rar_added_total": int(
            RAR_ADDED_TOTAL
        ),
        "rar_candidate_count_per_round": int(
            RAR_CANDIDATES
        ),
        "rar_score": (
            "sqrt(f_real^2 + f_imag^2)"
        ),
        "last_refinement_adam_step": int(
            RAR_LAST_REFINEMENT_STEP
        ),
        "adam_steps_after_last_refinement": int(
            FINAL_ADAM_STEPS_AFTER_LAST_RAR
        ),
        "final_n_res": int(
            FINAL_N_RES
        ),
        "n_bc": int(
            N_BC
        ),
        "n_ic": int(
            N_IC
        ),
        "adam_optimizer_seconds": float(
            adam_optimizer_seconds
        ),
        "rar_selection_seconds": float(
            rar_selection_seconds
        ),
        "sampling_stage_wall_seconds": float(
            sampling_wall_seconds
        ),
        "selected_points_file": str(
            point_path
        ),
        "acquisition_model_saved": False,
        "acquisition_optimizer_saved": False,
        "formal_lbfgs_warm_start_from_adam": False,
    }

    save_json(
        summary_path,
        summary,
    )


    del optimizer
    del acquisition_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"NLSE RAR sampling-only stage finished for seed={seed}. "
        f"Saved points only: {point_path}"
    )

    return summary


def run_sampling_all_seeds(
    overwrite=False,
):
    METHOD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = [
        run_sampling_stage(
            seed,
            overwrite=overwrite,
        )
        for seed in SEEDS
    ]

    save_json(
        METHOD_DIR
        / "all_sampling_stage_summaries.json",
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "seeds": SEEDS,
            "summaries": summaries,
        },
    )


def load_formal_points(
    seed,
):
    point_path = selected_points_path(
        seed
    )

    if not point_path.exists():
        raise FileNotFoundError(
            f"Missing selected NLSE RAR points: {point_path}\n"
            "Run select_nlse_rar_points_adam2500.py first."
        )

    with np.load(
        point_path
    ) as data:
        x_res = np.asarray(
            data[
                "x_res"
            ],
            dtype=np.float64,
        ).copy()

        t_res = np.asarray(
            data[
                "t_res"
            ],
            dtype=np.float64,
        ).copy()

    if len(
        x_res
    ) != FINAL_N_RES:
        raise RuntimeError(
            f"Selected NLSE RAR set has N_RES={len(x_res)}, "
            f"expected {FINAL_N_RES}."
        )

    original_np = (
        load_original_points_numpy(
            seed
        )
    )

    if not (
        np.array_equal(
            x_res[
                :N_RES
            ],
            original_np[
                "x_res"
            ],
        )
        and np.array_equal(
            t_res[
                :N_RES
            ],
            original_np[
                "t_res"
            ],
        )
    ):
        raise RuntimeError(
            "The first 10000 selected RAR points do not match "
            "the original fixed points."
        )

    original_torch = (
        load_training_points(
            seed
        )
    )

    formal_points = (
        make_points_with_residual_set(
            original_torch,
            x_res,
            t_res,
        )
    )

    return (
        formal_points,
        original_torch,
        point_path,
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
    monitor_seconds,
    latest_metrics,
    config,
):
    torch.save(
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "seed": int(seed),
            "target_checkpoint_eval": int(
                target_eval
            ),
            "actual_lbfgs_eval": int(
                actual_eval
            ),
            "lbfgs_eval": int(
                actual_eval
            ),
            "model_state_dict": cpu_state_dict(
                model
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "loss_history": np.asarray(
                history,
                dtype=np.float64,
            ),
            "monitor_history": (
                monitor_history
            ),
            "elapsed_seconds": float(
                elapsed_wall
            ),
            "monitor_diagnostic_seconds": float(
                monitor_seconds
            ),
            "core_lbfgs_time_seconds_estimate": float(
                elapsed_wall
                - monitor_seconds
            ),
            "latest_metrics": (
                latest_metrics
            ),
            "config": (
                config
            ),
        },
        path,
    )


def run_fresh_lbfgs_single_seed(
    seed,
):
    seed = int(seed)

    seed_dir = (
        METHOD_DIR
        / f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        formal_points,
        original_points,
        point_path,
    ) = load_formal_points(
        seed
    )


    model, initial_hash = (
        assert_fresh_initialization_matches_standard(
            seed
        )
    )

    config = {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": int(seed),
        "formal_network": (
            "fresh exact TanhPINN"
        ),
        "formal_initialization": (
            "set_seed(seed) then fresh TanhPINN; "
            "identical to Standard-PINN same-seed initialization"
        ),
        "formal_initial_state_sha256": (
            initial_hash
        ),
        "adam_warm_start": False,
        "acquisition_model_loaded": False,
        "acquisition_optimizer_loaded": False,
        "original_shared_n_res": int(
            N_RES
        ),
        "rar_added_n_res": int(
            RAR_ADDED_TOTAL
        ),
        "formal_n_res": int(
            FINAL_N_RES
        ),
        "formal_n_bc": int(
            N_BC
        ),
        "formal_n_ic": int(
            N_IC
        ),
        "hidden_layers": int(
            HIDDEN_LAYERS
        ),
        "hidden_neurons": int(
            HIDDEN_NEURONS
        ),
        "output_dim": int(
            OUTPUT_DIM
        ),
        "total_parameters": int(
            count_parameters(
                model
            )
        ),
        "lambda_res": float(
            LAMBDA_RES
        ),
        "lambda_bc": float(
            LAMBDA_BC
        ),
        "lambda_ic": float(
            LAMBDA_IC
        ),
        "lbfgs_lr": float(
            LBFGS_LR
        ),
        "lbfgs_max_iter": int(
            LBFGS_MAX_ITER
        ),
        "lbfgs_max_eval": int(
            LBFGS_MAX_EVAL
        ),
        "lbfgs_history_size": int(
            LBFGS_HISTORY_SIZE
        ),
        "lbfgs_tolerance_grad": float(
            LBFGS_TOLERANCE_GRAD
        ),
        "lbfgs_tolerance_change": float(
            LBFGS_TOLERANCE_CHANGE
        ),
        "lbfgs_line_search_fn": (
            LBFGS_LINE_SEARCH_FN
        ),
        "checkpoint_evals": list(
            CHECKPOINT_EVALS
        ),
        "optimizer": (
            "single continuous LBFGS"
        ),
        "optimizer_restart": False,
        "selected_points_file": str(
            point_path
        ),
        "fixed_monitor_points_file": str(
            training_points_path(
                seed
            )
        ),
        "lbfgs_monitor_every": int(
            LBFGS_MONITOR_EVERY
        ),
    }

    save_json(
        seed_dir
        / "formal_lbfgs_config.json",
        config,
    )

    initial_monitor = (
        compute_fixed_monitor_loss(
            model,
            original_points,
        )
    )

    save_json(
        seed_dir
        / "formal_initial_state.json",
        {
            "seed": int(seed),
            "initial_state_sha256": (
                initial_hash
            ),
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
    monitor_seconds = 0.0

    synchronize()
    start_time = time.time()

    def evaluate_monitor(
        force=False,
    ):
        nonlocal latest_monitor
        nonlocal monitor_seconds

        should = (
            force
            or eval_count == 1
            or eval_count
            % LBFGS_MONITOR_EVERY
            == 0
            or any(
                eval_count
                >= target
                and target
                not in saved
                for target
                in CHECKPOINT_EVALS
            )
        )

        if not should:
            return

        synchronize()
        monitor_start = time.time()

        latest_monitor = (
            compute_fixed_monitor_loss(
                model,
                original_points,
            )
        )

        synchronize()
        dt = (
            time.time()
            - monitor_start
        )

        monitor_seconds += dt

        monitor_history.append({
            "lbfgs_eval": int(
                eval_count
            ),
            **latest_monitor,
            "monitor_evaluation_seconds": float(
                dt
            ),
            "cumulative_monitor_seconds": float(
                monitor_seconds
            ),
        })

    def maybe_save_checkpoint():
        for target in CHECKPOINT_EVALS:
            if (
                eval_count
                >= target
                and target
                not in saved
            ):
                evaluate_monitor(
                    force=True
                )

                elapsed_wall = (
                    time.time()
                    - start_time
                )

                metrics = dict(
                    latest_metrics
                )

                metrics.update(
                    latest_monitor
                )

                path = (
                    seed_dir
                    / (
                        f"nlse_{METHOD}"
                        f"_seed{seed}"
                        f"_eval{target:05d}.pth"
                    )
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
                    monitor_seconds=monitor_seconds,
                    latest_metrics=metrics,
                    config=config,
                )

                saved.add(
                    target
                )

                print(
                    f"Saved passive NLSE RAR checkpoint: "
                    f"{path} "
                    f"(actual_eval={eval_count})"
                )

    def closure():
        nonlocal eval_count
        nonlocal latest_metrics

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            total_loss,
            loss_residual,
            loss_bc,
            loss_ic_unweighted,
            loss_ic_weighted,
            data_loss,
        ) = compute_loss_components(
            model,
            formal_points,
        )

        total_loss.backward()

        eval_count += 1

        synchronize()

        elapsed = (
            time.time()
            - start_time
        )

        latest_metrics = {
            "optimization_total_loss": float(
                total_loss.detach().cpu()
            ),
            "optimization_residual_loss": float(
                loss_residual.detach().cpu()
            ),
            "optimization_data_loss": float(
                data_loss.detach().cpu()
            ),
            "optimization_bc_loss": float(
                loss_bc.detach().cpu()
            ),
            "optimization_ic_loss_unweighted": float(
                loss_ic_unweighted.detach().cpu()
            ),
            "optimization_ic_loss_weighted": float(
                loss_ic_weighted.detach().cpu()
            ),
            "elapsed_seconds": float(
                elapsed
            ),
        }

        evaluate_monitor(
            force=False
        )

        history.append([
            int(eval_count),
            latest_metrics[
                "optimization_total_loss"
            ],
            latest_metrics[
                "optimization_residual_loss"
            ],
            latest_metrics[
                "optimization_data_loss"
            ],
            latest_metrics[
                "optimization_bc_loss"
            ],
            latest_metrics[
                "optimization_ic_loss_unweighted"
            ],
            latest_metrics[
                "optimization_ic_loss_weighted"
            ],
            float(
                elapsed
            ),
            float(
                monitor_seconds
            ),
        ])

        if (
            eval_count == 1
            or eval_count
            % PRINT_EVERY
            == 0
        ):
            monitor_text = (
                "NA"
                if latest_monitor
                is None
                else (
                    f"{latest_monitor['fixed_monitor_total_loss']:.6e}"
                )
            )

            print(
                f"NLSE RAR sampling-only | "
                f"seed {seed:4d} | "
                f"L-BFGS eval {eval_count:5d} | "
                f"Opt={latest_metrics['optimization_total_loss']:.6e} | "
                f"Monitor={monitor_text}"
            )

        maybe_save_checkpoint()

        return total_loss

    optimizer.step(
        closure
    )

    synchronize()

    total_wall_seconds = (
        time.time()
        - start_time
    )

    if latest_metrics is None:
        raise RuntimeError(
            f"No L-BFGS closure evaluation for NLSE RAR seed={seed}."
        )


    for target in CHECKPOINT_EVALS:
        if target not in saved:
            evaluate_monitor(
                force=True
            )

            metrics = dict(
                latest_metrics
            )

            metrics.update(
                latest_monitor
            )

            path = (
                seed_dir
                / (
                    f"nlse_{METHOD}"
                    f"_seed{seed}"
                    f"_eval{target:05d}.pth"
                )
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
                elapsed_wall=total_wall_seconds,
                monitor_seconds=monitor_seconds,
                latest_metrics=metrics,
                config=config,
            )

            saved.add(
                target
            )

    with (
        seed_dir
        / "lbfgs_history.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow([
            "lbfgs_eval",
            "optimization_total_loss",
            "optimization_residual_loss",
            "optimization_data_loss",
            "optimization_bc_loss",
            "optimization_ic_loss_unweighted",
            "optimization_ic_loss_weighted",
            "elapsed_wall_seconds",
            "cumulative_monitor_seconds",
        ])

        writer.writerows(
            history
        )

    with (
        seed_dir
        / "fixed_monitor_history.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                monitor_history[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            monitor_history
        )

    sampling_summary = json.loads(
        (
            seed_dir
            / "sampling_stage_summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    formal_summary = {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": int(seed),
        "formal_initial_state_sha256": (
            initial_hash
        ),
        "matches_standard_same_seed_initialization": True,
        "adam_warm_start": False,
        "original_shared_n_res": int(
            N_RES
        ),
        "rar_added_n_res": int(
            RAR_ADDED_TOTAL
        ),
        "formal_n_res": int(
            FINAL_N_RES
        ),
        "formal_n_bc": int(
            N_BC
        ),
        "formal_n_ic": int(
            N_IC
        ),
        "lbfgs_target_evals": int(
            LBFGS_MAX_EVAL
        ),
        "lbfgs_actual_evals": int(
            eval_count
        ),
        "lbfgs_wall_including_monitor_seconds": float(
            total_wall_seconds
        ),
        "lbfgs_monitor_seconds": float(
            monitor_seconds
        ),
        "lbfgs_core_seconds_estimate": float(
            total_wall_seconds
            - monitor_seconds
        ),
        "sampling_stage_wall_seconds": float(
            sampling_summary[
                "sampling_stage_wall_seconds"
            ]
        ),
        "sampling_adam_optimizer_seconds": float(
            sampling_summary[
                "adam_optimizer_seconds"
            ]
        ),
        "sampling_rar_selection_seconds": float(
            sampling_summary[
                "rar_selection_seconds"
            ]
        ),
        "end_to_end_cost_seconds": float(
            sampling_summary[
                "sampling_stage_wall_seconds"
            ]
            + total_wall_seconds
        ),
    }

    save_json(
        seed_dir
        / "formal_training_summary.json",
        formal_summary,
    )

    return formal_summary


def run_fresh_lbfgs_all_seeds():
    summaries = [
        run_fresh_lbfgs_single_seed(
            seed
        )
        for seed in SEEDS
    ]

    save_json(
        METHOD_DIR
        / "all_formal_training_summaries.json",
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
            f"Missing reference file: {REFERENCE_FILE}. "
            "Place NLS.mat in the working directory."
        )

    data = scipy.io.loadmat(
        REFERENCE_FILE
    )

    x = data[
        "x"
    ].flatten()

    if "tt" in data:
        t = data[
            "tt"
        ].flatten()

    elif "t" in data:
        t = data[
            "t"
        ].flatten()

    else:
        raise KeyError(
            "NLS.mat must contain 'tt' or 't'."
        )

    h_exact = data[
        "uu"
    ]

    target_shape = (
        len(x),
        len(t),
    )

    if h_exact.shape != target_shape:
        if (
            h_exact.T.shape
            == target_shape
        ):
            h_exact = (
                h_exact.T
            )

        else:
            raise ValueError(
                f"Unexpected reference shape: "
                f"{h_exact.shape}, "
                f"expected {target_shape}."
            )

    X, T = np.meshgrid(
        x,
        t,
        indexing="ij",
    )

    return (
        x,
        t,
        X,
        T,
        h_exact,
    )


def predict(
    model,
    X,
    T,
    output_shape,
):
    x_flat = torch.tensor(
        X.reshape(
            -1,
            1,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )

    t_flat = torch.tensor(
        T.reshape(
            -1,
            1,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )

    model.eval()

    parts = []

    with torch.no_grad():
        for start in range(
            0,
            x_flat.shape[
                0
            ],
            20000,
        ):
            end = (
                start
                + 20000
            )

            parts.append(
                model(
                    x_flat[
                        start:end
                    ],
                    t_flat[
                        start:end
                    ],
                )
                .detach()
                .cpu()
                .numpy()
            )

    solution = np.vstack(
        parts
    )

    u_pred = solution[
        :,
        0,
    ].reshape(
        output_shape
    )

    v_pred = solution[
        :,
        1,
    ].reshape(
        output_shape
    )

    h_pred = (
        u_pred
        + 1j * v_pred
    )

    return (
        h_pred,
        u_pred,
        v_pred,
    )


def relative_l2(
    h_exact,
    h_pred,
):
    exact_flat = np.asarray(
        h_exact
    ).reshape(
        -1
    )

    pred_flat = np.asarray(
        h_pred
    ).reshape(
        -1
    )

    return float(
        np.linalg.norm(
            exact_flat
            - pred_flat,
            2,
        )
        / np.linalg.norm(
            exact_flat,
            2,
        )
    )
