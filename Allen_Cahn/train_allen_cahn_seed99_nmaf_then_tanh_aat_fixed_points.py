from __future__ import annotations

import csv
import gc
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn

import allen_cahn_common_fixed_points_continuous_lbfgs as C


SEED = 99
TRAIN_ORDER = ("nmaf", "tanh_aat")

METHOD_LABELS = {
    "nmaf": "NMAF",
    "tanh_aat": "tanh-AAT",
}


class FixedNMAF(nn.Module):


    def forward(self, z):
        return -z * (torch.exp(-0.5 * z ** 2) - torch.tanh(z))


class NMAFPINN(nn.Module):


    def __init__(
        self,
        hidden_layers=C.HIDDEN_LAYERS,
        hidden_neurons=C.HIDDEN_NEURONS,
    ):
        super().__init__()

        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)


        self.activation = FixedNMAF()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))

        for layer in self.hidden_layers:
            out = self.activation(layer(out))

        return self.output_layer(out)


class SharedTanhAAT(nn.Module):


    def __init__(
        self,
        in_features,
        initial_alpha=C.INITIAL_ALPHA,
        initial_beta=C.INITIAL_BETA,
    ):
        super().__init__()

        self.alpha = nn.Parameter(
            torch.ones(in_features, dtype=C.DTYPE) * initial_alpha
        )
        self.beta = nn.Parameter(
            torch.ones(in_features, dtype=C.DTYPE) * initial_beta
        )

    def forward(self, z):
        X = self.alpha * z + self.beta
        return torch.tanh(X)


class TanhAATPINN(nn.Module):


    def __init__(
        self,
        hidden_layers=C.HIDDEN_LAYERS,
        hidden_neurons=C.HIDDEN_NEURONS,
    ):
        super().__init__()

        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)

        self.activation = SharedTanhAAT(hidden_neurons)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))

        for layer in self.hidden_layers:
            out = self.activation(layer(out))

        return self.output_layer(out)


def build_model(method):
    if method == "nmaf":
        return NMAFPINN().double().to(C.DEVICE)

    if method == "tanh_aat":
        return TanhAATPINN().double().to(C.DEVICE)

    raise ValueError(f"Unknown method: {method}")


def method_label(method):
    return METHOD_LABELS[method]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_adaptive_parameters(model):
    if hasattr(model, "activation") and hasattr(model.activation, "alpha"):
        return (
            model.activation.alpha.numel()
            + model.activation.beta.numel()
        )
    return 0


def adaptive_parameter_summary(model):
    if not (
        hasattr(model, "activation")
        and hasattr(model.activation, "alpha")
    ):
        return {}

    alpha = model.activation.alpha.detach().cpu().numpy()
    beta = model.activation.beta.detach().cpu().numpy()

    return {
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha, ddof=1))
        if alpha.size > 1 else 0.0,
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "beta_mean": float(np.mean(beta)),
        "beta_std": float(np.std(beta, ddof=1))
        if beta.size > 1 else 0.0,
        "beta_min": float(np.min(beta)),
        "beta_max": float(np.max(beta)),
        "delta_alpha_l2": float(
            np.linalg.norm(alpha - C.INITIAL_ALPHA, 2)
        ),
        "delta_beta_l2": float(
            np.linalg.norm(beta - C.INITIAL_BETA, 2)
        ),
    }


def method_config(method, seed, model):
    cfg = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "device": str(C.DEVICE),
        "dtype": str(C.DTYPE),

        "hidden_layers": C.HIDDEN_LAYERS,
        "hidden_neurons": C.HIDDEN_NEURONS,

        "n_res": C.N_RES,
        "n_bc": C.N_BC,
        "n_ic": C.N_IC,

        "training_points_file": str(C.training_points_path(seed)),
        "training_points_policy":
            "load existing fixed points only; never regenerate in this script",

        "optimizer": "single continuous LBFGS",
        "lbfgs_max_iter": C.LBFGS_MAX_ITER,
        "lbfgs_max_eval": C.LBFGS_MAX_EVAL,
        "lbfgs_lr": C.LBFGS_LR,
        "lbfgs_history_size": C.LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": C.LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": C.LBFGS_TOLERANCE_CHANGE,
        "lbfgs_line_search_fn": C.LBFGS_LINE_SEARCH_FN,

        "checkpoint_evals": list(C.CHECKPOINT_EVALS),
        "checkpoint_policy":
            "passive save inside closure; optimizer is not restarted",

        "lambda_res": C.LAMBDA_RES,
        "lambda_bc": C.LAMBDA_BC,
        "lambda_ic": C.LAMBDA_IC,

        "pde": "u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0",
        "ic": "u(0,x)=x^2*cos(pi*x)",
        "bc": "periodic u and u_x at x=-1 and x=1",

        "total_parameters": count_parameters(model),
        "adaptive_parameters": count_adaptive_parameters(model),

        "backbone_initialization":
            "set_seed(seed) is reset before each method; Linear layers are "
            "constructed in the same order as the existing baselines",
    }

    if method == "nmaf":
        cfg.update({
            "activation": "fixed NMAF",
            "nmaf":
                "phi(z) = -z * (exp(-0.5*z^2) - tanh(z))",
            "aat": None,
            "trainable_activation_parameters": 0,
        })

    elif method == "tanh_aat":
        cfg.update({
            "activation":
                "tanh with shared neuron-wise adaptive affine transformation",
            "base_activation": "tanh",
            "aat": "X_j = alpha_j * z_j + beta_j",
            "activation_formula":
                "phi(z_j) = tanh(alpha_j*z_j + beta_j)",
            "aat_parameter_sharing":
                "one shared alpha vector and one shared beta vector reused "
                "across all hidden activation locations",
            "initial_alpha": C.INITIAL_ALPHA,
            "initial_beta": C.INITIAL_BETA,
        })

    return cfg


def save_checkpoint(
    path,
    method,
    model,
    optimizer,
    seed,
    target_eval,
    actual_eval,
    history,
    elapsed_seconds,
    latest_metrics,
    config,
):
    checkpoint = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),

        "target_checkpoint_eval": int(target_eval),
        "actual_lbfgs_eval": int(actual_eval),
        "lbfgs_eval": int(actual_eval),

        "model_state_dict": C.cpu_state_dict(model),
        "optimizer_state_dict":
            optimizer.state_dict() if optimizer is not None else None,

        "loss_history":
            np.asarray(history, dtype=np.float64),

        "elapsed_seconds": float(elapsed_seconds),
        "latest_metrics": latest_metrics,
        "config": config,
    }

    if method == "tanh_aat":
        checkpoint["adaptive_parameter_summary"] = (
            adaptive_parameter_summary(model)
        )

    torch.save(checkpoint, path)


def verify_existing_points(seed):


    path = C.training_points_path(seed)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing existing fixed training points: {path}\n"
            "This ablation script deliberately DOES NOT generate new points. "
            "Please restore the original seed-99 point file first."
        )

    print(f"Using existing fixed training points only: {path}")
    return path


def train_single_method(method, seed=SEED):
    if method not in TRAIN_ORDER:
        raise ValueError(f"Unsupported method: {method}")

    verify_existing_points(seed)


    C.set_seed(seed)

    method_dir = C.RESULT_ROOT / method
    seed_dir = method_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(method)


    points = C.load_training_points(seed)

    cfg = method_config(method, seed, model)
    C.save_json(seed_dir / "method_config.json", cfg)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=C.LBFGS_LR,
        max_iter=C.LBFGS_MAX_ITER,
        max_eval=C.LBFGS_MAX_EVAL,
        history_size=C.LBFGS_HISTORY_SIZE,
        tolerance_grad=C.LBFGS_TOLERANCE_GRAD,
        tolerance_change=C.LBFGS_TOLERANCE_CHANGE,
        line_search_fn=C.LBFGS_LINE_SEARCH_FN,
    )

    history = []
    saved = set()
    eval_count = 0
    latest_metrics = None

    C.synchronize()
    start_time = time.time()

    def maybe_save_checkpoint():
        for checkpoint_eval in C.CHECKPOINT_EVALS:
            if (
                eval_count >= checkpoint_eval
                and checkpoint_eval not in saved
            ):
                elapsed = time.time() - start_time

                checkpoint_path = (
                    seed_dir
                    / f"allen_cahn_{method}_seed{seed}_"
                      f"eval{checkpoint_eval:05d}.pth"
                )

                save_checkpoint(
                    path=checkpoint_path,
                    method=method,
                    model=model,
                    optimizer=optimizer,
                    seed=seed,
                    target_eval=checkpoint_eval,
                    actual_eval=eval_count,
                    history=history,
                    elapsed_seconds=elapsed,
                    latest_metrics=latest_metrics,
                    config=cfg,
                )

                saved.add(checkpoint_eval)

                print(
                    f"Saved passive checkpoint: {checkpoint_path} "
                    f"(actual_eval={eval_count})"
                )

    def closure():
        nonlocal eval_count, latest_metrics

        optimizer.zero_grad(set_to_none=True)

        (
            total_loss,
            loss_residual,
            loss_bc,
            loss_ic_unweighted,
            loss_ic_weighted,
            data_loss,
        ) = C.compute_loss_components(
            model,
            points,
        )

        total_loss.backward()

        eval_count += 1

        C.synchronize()
        elapsed = time.time() - start_time

        latest_metrics = {
            "total_loss": float(total_loss.detach().cpu()),
            "residual_loss": float(loss_residual.detach().cpu()),
            "bc_loss": float(loss_bc.detach().cpu()),
            "ic_loss_unweighted":
                float(loss_ic_unweighted.detach().cpu()),
            "ic_loss_weighted":
                float(loss_ic_weighted.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "elapsed_seconds": float(elapsed),
        }

        history.append([
            eval_count,
            latest_metrics["total_loss"],
            latest_metrics["residual_loss"],
            latest_metrics["data_loss"],
            latest_metrics["bc_loss"],
            latest_metrics["ic_loss_unweighted"],
            latest_metrics["ic_loss_weighted"],
            latest_metrics["elapsed_seconds"],
        ])

        if (
            eval_count == 1
            or eval_count % C.PRINT_EVERY == 0
        ):
            print(
                f"{method_label(method):10s} | "
                f"seed {seed:4d} | "
                f"L-BFGS eval {eval_count:5d} | "
                f"Total={latest_metrics['total_loss']:.6e} | "
                f"Residual={latest_metrics['residual_loss']:.6e} | "
                f"Data={latest_metrics['data_loss']:.6e}"
            )

        maybe_save_checkpoint()
        return total_loss

    optimizer.step(closure)

    C.synchronize()
    total_time = time.time() - start_time

    if latest_metrics is None:
        raise RuntimeError(
            f"No L-BFGS closure evaluation was performed for "
            f"method={method}, seed={seed}."
        )


    for checkpoint_eval in C.CHECKPOINT_EVALS:
        if checkpoint_eval not in saved:
            checkpoint_path = (
                seed_dir
                / f"allen_cahn_{method}_seed{seed}_"
                  f"eval{checkpoint_eval:05d}.pth"
            )

            save_checkpoint(
                path=checkpoint_path,
                method=method,
                model=model,
                optimizer=optimizer,
                seed=seed,
                target_eval=checkpoint_eval,
                actual_eval=eval_count,
                history=history,
                elapsed_seconds=total_time,
                latest_metrics=latest_metrics,
                config=cfg,
            )

            saved.add(checkpoint_eval)

            print(
                f"Saved fallback checkpoint: {checkpoint_path} "
                f"(actual_eval={eval_count})"
            )

    history_array = np.asarray(
        history,
        dtype=np.float64,
    )

    np.savetxt(
        seed_dir / "allen_cahn_loss_history.txt",
        history_array,
        header=(
            "lbfgs_eval\tTotalLoss\tResidualLoss\tDataLoss\t"
            "BCLoss\tICLossUnweighted\tICLossWeighted\tElapsedSeconds"
        ),
        fmt="%.10e",
        delimiter="\t",
    )

    scipy.io.savemat(
        seed_dir / "allen_cahn_loss_history.mat",
        {
            "loss_history": history_array,
            "column_names": [
                "lbfgs_eval",
                "TotalLoss",
                "ResidualLoss",
                "DataLoss",
                "BCLoss",
                "ICLossUnweighted",
                "ICLossWeighted",
                "ElapsedSeconds",
            ],
        },
    )

    with (
        seed_dir / "lbfgs_history.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "lbfgs_eval",
            "total_loss",
            "residual_loss",
            "data_loss",
            "bc_loss",
            "ic_loss_unweighted",
            "ic_loss_weighted",
            "elapsed_seconds",
        ])

        writer.writerows(history)

    summary = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),

        "final_actual_lbfgs_eval": int(eval_count),
        "target_lbfgs_eval": int(C.LBFGS_MAX_EVAL),
        "saved_checkpoints": list(C.CHECKPOINT_EVALS),

        "total_time_seconds": float(total_time),
        "final_metrics": latest_metrics,

        "training_points_file":
            str(C.training_points_path(seed)),
        "training_points_reused": True,
        "training_points_regenerated": False,
    }

    if method == "tanh_aat":
        summary["final_adaptive_parameter_summary"] = (
            adaptive_parameter_summary(model)
        )

    C.save_json(
        seed_dir / "training_summary.json",
        summary,
    )

    print("-" * 96)
    print(
        f"{method_label(method)} seed {seed} "
        f"continuous training completed."
    )
    print(
        f"Actual L-BFGS evaluations: {eval_count}"
    )
    print(
        f"Total time: {total_time:.2f} s"
    )
    print(
        f"Final total loss: "
        f"{latest_metrics['total_loss']:.6e}"
    )
    print("-" * 96)

    return summary


def cleanup_gpu_memory():


    gc.collect()

    if C.DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        torch.cuda.synchronize()

        print(
            "CUDA cache cleared before the next ablation run. "
            f"Allocated={torch.cuda.memory_allocated()/1024**2:.2f} MiB, "
            f"Reserved={torch.cuda.memory_reserved()/1024**2:.2f} MiB"
        )
    else:
        print(
            "CPU device in use; garbage collection completed."
        )


def main():
    print("=" * 96)
    print("Allen-Cahn seed=99 four-way activation/AAT ablation")
    print("Existing: tanh, NMAF-AAT")
    print("Training now: fixed NMAF -> tanh-AAT")
    print("=" * 96)

    point_path = verify_existing_points(SEED)

    print(
        "\nThe SAME existing seed-99 point file will be used "
        "for both new methods:"
    )
    print(f"  {point_path}")

    summaries = []


    summaries.append(
        train_single_method(
            "nmaf",
            SEED,
        )
    )


    cleanup_gpu_memory()


    summaries.append(
        train_single_method(
            "tanh_aat",
            SEED,
        )
    )

    cleanup_gpu_memory()

    combined = {
        "experiment":
            "Allen-Cahn seed99 four-way activation/AAT ablation",
        "seed": SEED,

        "existing_methods_not_retrained": [
            "tanh",
            "nmaf_aat",
        ],

        "methods_trained_by_this_script": [
            "nmaf",
            "tanh_aat",
        ],

        "training_order": list(TRAIN_ORDER),

        "fixed_training_points_file":
            str(C.training_points_path(SEED)),
        "training_points_reused": True,
        "training_points_regenerated": False,

        "shared_training_setup": {
            "hidden_layers": C.HIDDEN_LAYERS,
            "hidden_neurons": C.HIDDEN_NEURONS,
            "dtype": str(C.DTYPE),
            "device": str(C.DEVICE),
            "lbfgs_max_eval": C.LBFGS_MAX_EVAL,
            "lbfgs_history_size": C.LBFGS_HISTORY_SIZE,
            "lambda_res": C.LAMBDA_RES,
            "lambda_bc": C.LAMBDA_BC,
            "lambda_ic": C.LAMBDA_IC,
        },

        "summaries": summaries,
    }

    C.save_json(
        C.RESULT_ROOT
        / "seed99_activation_aat_ablation_new_methods_summary.json",
        combined,
    )

    print("\nAll requested seed-99 runs completed.")
    print("Existing results remain untouched:")
    print("  tanh")
    print("  nmaf_aat")
    print("New results written under:")
    print(f"  {C.RESULT_ROOT / 'nmaf' / 'seed_99'}")
    print(f"  {C.RESULT_ROOT / 'tanh_aat' / 'seed_99'}")


if __name__ == "__main__":
    main()
