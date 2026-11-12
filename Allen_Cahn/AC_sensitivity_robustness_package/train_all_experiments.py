from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ac_sensitivity_common import (
    RESULT_ROOT,
    TRAIN_EXPERIMENTS,
    SEEDS,
    METHODS,
    DEVICE,
    DTYPE,
    LAMBDA_RES,
    LAMBDA_BC,
    LAMBDA_IC,
    LBFGS_LR,
    LBFGS_MAX_ITER,
    LBFGS_MAX_EVAL,
    LBFGS_HISTORY_SIZE,
    LBFGS_TOLERANCE_GRAD,
    LBFGS_TOLERANCE_CHANGE,
    LBFGS_LINE_SEARCH_FN,
    CHECKPOINT_EVAL,
    PRINT_EVERY,
    build_model,
    count_parameters,
    count_adaptive_parameters,
    data_path_for_spec,
    load_points_for_spec,
    method_label,
    nominal_training_count,
    physical_coordinate_count,
    prepare_all_generated_datasets,
    sample_counts,
    set_seed,
    spec_to_dict,
    synchronize,
    verify_required_baseline_files,
    write_experiment_plan,
)


def compute_loss_components(model, points):

    x_res = points["x_res"].detach().clone().requires_grad_(True)
    t_res = points["t_res"].detach().clone().requires_grad_(True)

    u = model(x_res, t_res)
    u_t = torch.autograd.grad(
        u, t_res, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x_res, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_res, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]

    residual = u_t - 0.0001 * u_xx + 5.0 * u ** 3 - 5.0 * u
    loss_residual = torch.mean(residual ** 2)


    x_bc_l = points["x_bc_l"].detach().clone().requires_grad_(True)
    x_bc_r = points["x_bc_r"].detach().clone().requires_grad_(True)
    t_bc = points["t_bc"].detach().clone()

    u_l = model(x_bc_l, t_bc)
    u_r = model(x_bc_r, t_bc)
    u_x_l = torch.autograd.grad(
        u_l, x_bc_l, torch.ones_like(u_l), create_graph=True, retain_graph=True
    )[0]
    u_x_r = torch.autograd.grad(
        u_r, x_bc_r, torch.ones_like(u_r), create_graph=True, retain_graph=True
    )[0]

    loss_bc = torch.mean((u_l - u_r) ** 2) + torch.mean((u_x_l - u_x_r) ** 2)


    u_ic_pred = model(points["x_ic"], points["t_ic"])
    loss_ic_unweighted = torch.mean(
        (u_ic_pred - points["u_ic_target"]) ** 2
    )
    loss_ic_weighted = LAMBDA_IC * loss_ic_unweighted

    data_loss = LAMBDA_BC * loss_bc + loss_ic_weighted
    total_loss = LAMBDA_RES * loss_residual + data_loss

    return (
        total_loss,
        loss_residual,
        loss_bc,
        loss_ic_unweighted,
        loss_ic_weighted,
        data_loss,
    )


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run_directory(spec, method, seed):
    return RESULT_ROOT / spec.name / method / f"seed_{seed}"


def run_is_complete(spec, method, seed):
    d = run_directory(spec, method, seed)
    return (d / "final_checkpoint.pth").is_file() and (d / "training_summary.json").is_file()


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def train_one(spec, method, seed):
    out_dir = run_directory(spec, method, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_is_complete(spec, method, seed):
        print(f"[SKIP] {spec.name} | {method_label(method)} | seed {seed} already complete.")
        return

    set_seed(seed)
    model = build_model(method, spec.depth, spec.width)
    points = load_points_for_spec(seed, spec)

    n_res, n_bc, n_ic = sample_counts(spec)
    config = {
        "experiment": spec_to_dict(spec),
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "total_parameters": int(count_parameters(model)),
        "adaptive_parameters": int(count_adaptive_parameters(model)),
        "parameter_to_nominal_point_ratio": float(
            count_parameters(model) / nominal_training_count(spec)
        ),
        "parameter_to_coordinate_point_ratio": float(
            count_parameters(model) / physical_coordinate_count(spec)
        ),
        "training_points_file": str(data_path_for_spec(seed, spec)),
        "n_res": int(n_res),
        "n_bc": int(n_bc),
        "n_ic": int(n_ic),
        "optimizer": "single continuous LBFGS",
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": float(LBFGS_TOLERANCE_CHANGE),
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "lambda_res": LAMBDA_RES,
        "lambda_bc": LAMBDA_BC,
        "lambda_ic": LAMBDA_IC,
        "noise_note": (
            "Noise is added only to IC target values. PDE residual and periodic BC "
            "constraints remain exact."
        ),
    }
    save_json(out_dir / "config.json", config)

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
    eval_count = 0
    checkpoint_saved = False
    latest_metrics = None

    synchronize()
    start_time = time.time()

    def closure():
        nonlocal eval_count, checkpoint_saved, latest_metrics

        optimizer.zero_grad(set_to_none=True)

        (
            total_loss,
            loss_residual,
            loss_bc,
            loss_ic_unweighted,
            loss_ic_weighted,
            data_loss,
        ) = compute_loss_components(model, points)

        total_loss.backward()

        eval_count += 1
        synchronize()
        elapsed = time.time() - start_time

        latest_metrics = {
            "total_loss": float(total_loss.detach().cpu()),
            "residual_loss": float(loss_residual.detach().cpu()),
            "bc_loss": float(loss_bc.detach().cpu()),
            "ic_loss_unweighted": float(loss_ic_unweighted.detach().cpu()),
            "ic_loss_weighted": float(loss_ic_weighted.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "elapsed_seconds": float(elapsed),
        }
        history.append(
            [
                eval_count,
                latest_metrics["total_loss"],
                latest_metrics["residual_loss"],
                latest_metrics["data_loss"],
                latest_metrics["bc_loss"],
                latest_metrics["ic_loss_unweighted"],
                latest_metrics["ic_loss_weighted"],
                latest_metrics["elapsed_seconds"],
            ]
        )

        if eval_count == 1 or eval_count % PRINT_EVERY == 0:
            print(
                f"{spec.name:12s} | {method_label(method):13s} | seed {seed:4d} | "
                f"eval {eval_count:5d} | Total={latest_metrics['total_loss']:.6e}"
            )

        if eval_count >= CHECKPOINT_EVAL and not checkpoint_saved:
            torch.save(
                {
                    "experiment": spec_to_dict(spec),
                    "method": method,
                    "seed": int(seed),
                    "actual_eval": int(eval_count),
                    "model_state_dict": cpu_state_dict(model),
                    "latest_metrics": latest_metrics,
                    "config": config,
                },
                out_dir / f"checkpoint_eval{CHECKPOINT_EVAL}.pth",
            )
            checkpoint_saved = True

        return total_loss

    optimizer.step(closure)
    synchronize()
    total_time = time.time() - start_time

    if latest_metrics is None:
        raise RuntimeError("L-BFGS closure was never evaluated.")

    history_array = np.asarray(history, dtype=np.float64)
    np.savetxt(
        out_dir / "loss_history.txt",
        history_array,
        header=(
            "lbfgs_eval\tTotalLoss\tResidualLoss\tDataLoss\tBCLoss\t"
            "ICLossUnweighted\tICLossWeighted\tElapsedSeconds"
        ),
        fmt="%.10e",
        delimiter="\t",
    )

    with (out_dir / "loss_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lbfgs_eval",
                "total_loss",
                "residual_loss",
                "data_loss",
                "bc_loss",
                "ic_loss_unweighted",
                "ic_loss_weighted",
                "elapsed_seconds",
            ]
        )
        writer.writerows(history)

    final_ckpt = {
        "experiment": spec_to_dict(spec),
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "actual_eval": int(eval_count),
        "target_eval": int(LBFGS_MAX_EVAL),
        "elapsed_seconds": float(total_time),
        "model_state_dict": cpu_state_dict(model),
        "latest_metrics": latest_metrics,
        "config": config,
    }
    torch.save(final_ckpt, out_dir / "final_checkpoint.pth")

    summary = {
        "experiment": spec_to_dict(spec),
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "actual_eval": int(eval_count),
        "target_eval": int(LBFGS_MAX_EVAL),
        "total_time_seconds": float(total_time),
        "final_metrics": latest_metrics,
        "total_parameters": int(count_parameters(model)),
        "adaptive_parameters": int(count_adaptive_parameters(model)),
        "parameter_to_nominal_point_ratio": float(
            count_parameters(model) / nominal_training_count(spec)
        ),
        "parameter_to_coordinate_point_ratio": float(
            count_parameters(model) / physical_coordinate_count(spec)
        ),
    }
    save_json(out_dir / "training_summary.json", summary)

    print(
        f"[DONE] {spec.name} | {method_label(method)} | seed {seed} | "
        f"eval={eval_count} | time={total_time:.1f}s | "
        f"loss={latest_metrics['total_loss']:.6e}"
    )

    del optimizer, model, points
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 100)
    print("Allen-Cahn sensitivity / robustness training")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print("Baseline 7x128 clean case will NOT be retrained.")
    print("=" * 100)

    verify_required_baseline_files(require_reference=False)
    plan_path = write_experiment_plan()
    print(f"Experiment plan: {plan_path}")


    prepare_all_generated_datasets(overwrite=False)

    total_new_runs = len(TRAIN_EXPERIMENTS) * len(METHODS) * len(SEEDS)
    print(f"Planned new runs: {total_new_runs}")

    run_index = 0
    for spec in TRAIN_EXPERIMENTS:
        print("\n" + "#" * 100)
        print(
            f"Experiment: {spec.name} | study={spec.study} | level={spec.level} | "
            f"depth={spec.depth} | width={spec.width} | "
            f"data_factor={spec.data_factor} | noise={spec.noise_pct}%"
        )
        print("#" * 100)


        for seed in SEEDS:
            for method in METHODS:
                run_index += 1
                print(f"\n[{run_index}/{total_new_runs}]")
                train_one(spec, method, seed)

    print("\nAll requested NEW experiments are complete.")


if __name__ == "__main__":
    main()
