import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn


SEEDS = [99, 1234, 2026]
REPRESENTATIVE_SEED = 99

RESULT_ROOT = Path("burgers_multiseed_results")
POINTS_DIR = RESULT_ROOT / "fixed_training_points"
REFERENCE_FILE = Path("burgers_shock.mat")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 4
HIDDEN_NEURONS = 64

N_RES = 5000
N_BC = 200
N_IC = 300

X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0

LBFGS_LR = 1.0
LBFGS_MAX_ITER = 5000
LBFGS_MAX_EVAL = 5000
LBFGS_HISTORY_SIZE = 50
LBFGS_TOLERANCE_GRAD = 1.0e-9
LBFGS_TOLERANCE_CHANGE = np.finfo(float).eps
LBFGS_LINE_SEARCH_FN = "strong_wolfe"

CHECKPOINT_EVALS = (2500, 5000)
PRINT_EVERY = 100
IMAGE_DPI = 800

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class TanhPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super(TanhPINN, self).__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)


class StanActivation(nn.Module):


    def __init__(self, in_features, initial_beta=STAN_INITIAL_BETA):
        super(StanActivation, self).__init__()
        self.beta = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * initial_beta
        )

    def forward(self, z):
        tanh_z = torch.tanh(z)
        return tanh_z + self.beta * z * tanh_z


class StanPINN(nn.Module):


    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super(StanPINN, self).__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)


        self.stan_activations = nn.ModuleList([
            StanActivation(hidden_neurons)
            for _ in range(hidden_layers + 1)
        ])

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.stan_activations[0](self.input_layer(inputs))
        for activation_index, layer in enumerate(self.hidden_layers, start=1):
            out = self.stan_activations[activation_index](layer(out))
        return self.output_layer(out)


class AdaptiveActivation(nn.Module):
    def __init__(self, in_features, initial_alpha=INITIAL_ALPHA, initial_beta=INITIAL_BETA):
        super(AdaptiveActivation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_alpha)
        self.beta = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_beta)

    def forward(self, x):
        X = self.alpha * x + self.beta
        return -X * (torch.exp(-0.5 * X ** 2) - torch.tanh(X))


class NMAFAATPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super(NMAFAATPINN, self).__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = AdaptiveActivation(hidden_neurons)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)


def build_model(method):
    if method == "tanh":
        return TanhPINN().double().to(DEVICE)
    if method == "nmaf_aat":
        return NMAFAATPINN().double().to(DEVICE)
    if method == "stan":
        return StanPINN().double().to(DEVICE)
    raise ValueError(f"Unknown method: {method}")


def method_label(method):
    return {
        "tanh": "Standard PINN",
        "nmaf_aat": "NMAF-AAT",
        "stan": "Stan-PINN",
    }[method]


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def count_adaptive_parameters(model):
    if hasattr(model, "activation") and hasattr(model.activation, "alpha"):
        return model.activation.alpha.numel() + model.activation.beta.numel()
    if hasattr(model, "stan_activations"):
        return sum(activation.beta.numel() for activation in model.stan_activations)
    return 0


def adaptive_parameter_summary(model):
    if not (hasattr(model, "activation") and hasattr(model.activation, "alpha")):
        return {}
    alpha = model.activation.alpha.detach().cpu().numpy()
    beta = model.activation.beta.detach().cpu().numpy()
    return {
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha, ddof=1)) if alpha.size > 1 else 0.0,
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "beta_mean": float(np.mean(beta)),
        "beta_std": float(np.std(beta, ddof=1)) if beta.size > 1 else 0.0,
        "beta_min": float(np.min(beta)),
        "beta_max": float(np.max(beta)),
        "delta_alpha_l2": float(np.linalg.norm(alpha - INITIAL_ALPHA, 2)),
        "delta_beta_l2": float(np.linalg.norm(beta - INITIAL_BETA, 2)),
    }


def stan_parameter_summary(model):
    if not hasattr(model, "stan_activations"):
        return {}
    beta = torch.cat([
        activation.beta.detach().reshape(-1).cpu()
        for activation in model.stan_activations
    ]).numpy()
    return {
        "beta_mean": float(np.mean(beta)),
        "beta_std": float(np.std(beta, ddof=1)) if beta.size > 1 else 0.0,
        "beta_min": float(np.min(beta)),
        "beta_max": float(np.max(beta)),
        "delta_beta_l2": float(np.linalg.norm(beta - STAN_INITIAL_BETA, 2)),
        "number_of_beta_parameters": int(beta.size),
        "number_of_activation_locations": int(len(model.stan_activations)),
    }


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def training_points_path(seed):
    return POINTS_DIR / f"burgers_training_points_seed{seed}.npz"


def generate_points_for_seed(seed):
    rng = np.random.default_rng(seed)
    x_res = rng.random((N_RES, 1)) * (X_MAX - X_MIN) + X_MIN
    t_res = rng.random((N_RES, 1)) * (T_MAX - T_MIN) + T_MIN
    x_bc = np.vstack([
        np.full((N_BC // 2, 1), X_MIN),
        np.full((N_BC - N_BC // 2, 1), X_MAX),
    ])
    t_bc = rng.random((N_BC, 1)) * (T_MAX - T_MIN) + T_MIN
    x_ic = rng.random((N_IC, 1)) * (X_MAX - X_MIN) + X_MIN
    t_ic = np.zeros((N_IC, 1))
    return {
        "x_res": x_res.astype(np.float64),
        "t_res": t_res.astype(np.float64),
        "x_bc": x_bc.astype(np.float64),
        "t_bc": t_bc.astype(np.float64),
        "x_ic": x_ic.astype(np.float64),
        "t_ic": t_ic.astype(np.float64),
    }


def generate_and_save_all_training_points(overwrite=False):
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": SEEDS,
        "n_res": N_RES,
        "n_bc": N_BC,
        "n_ic": N_IC,
        "domain": {"x_min": X_MIN, "x_max": X_MAX, "t_min": T_MIN, "t_max": T_MAX},
        "sampling": "fixed numpy default_rng points",
        "files": [],
    }
    for seed in SEEDS:
        path = training_points_path(seed)
        if path.exists() and not overwrite:
            print(f"Existing fixed point file kept: {path}")
        else:
            np.savez_compressed(path, **generate_points_for_seed(seed))
            print(f"Saved fixed training points: {path}")
        manifest["files"].append(str(path))
    save_json(POINTS_DIR / "training_points_manifest.json", manifest)


def load_training_points(seed):
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fixed training points: {path}\n"
            "Run generate_burgers_fixed_training_points_99_1234_2026.py first."
        )
    data = np.load(path)
    keys = ["x_res", "t_res", "x_bc", "t_bc", "x_ic", "t_ic"]
    return {key: torch.tensor(data[key], dtype=DTYPE, device=DEVICE) for key in keys}


def compute_loss_components(model, points):
    x_res = points["x_res"].detach().clone().requires_grad_(True)
    t_res = points["t_res"].detach().clone().requires_grad_(True)
    x_bc = points["x_bc"]
    t_bc = points["t_bc"]
    x_ic = points["x_ic"]
    t_ic = points["t_ic"]

    u = model(x_res, t_res)
    u_t = torch.autograd.grad(u, t_res, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x_res, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    residual = u_t + u * u_x - (0.01 / np.pi) * u_xx
    loss_residual = torch.mean(residual ** 2)

    loss_bc = torch.mean(model(x_bc, t_bc) ** 2)
    u_ic_pred = model(x_ic, t_ic)
    u_ic_exact = -torch.sin(np.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

    data_loss = loss_bc + loss_ic
    total_loss = loss_residual + data_loss
    return total_loss, loss_residual, loss_bc, loss_ic, data_loss


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def method_config(method, seed, model):
    cfg = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_neurons": HIDDEN_NEURONS,
        "n_res": N_RES,
        "n_bc": N_BC,
        "n_ic": N_IC,
        "training_points_file": str(training_points_path(seed)),
        "optimizer": "single continuous LBFGS",
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "checkpoint_policy": "passive save inside closure; optimizer is not restarted",
        "pde": "u_t + u*u_x - (0.01/pi)*u_xx = 0",
        "ic": "u(x,0) = -sin(pi*x)",
        "bc": "u(-1,t)=u(1,t)=0",
        "total_parameters": count_parameters(model),
        "adaptive_parameters": count_adaptive_parameters(model),
    }
    if method == "nmaf_aat":
        cfg.update({
            "activation": "NMAF with shared neuron-wise AAT",
            "nmaf": "phi(X) = -X * (exp(-0.5*X^2) - tanh(X))",
            "aat": "X_j = alpha_j * z_j + beta_j",
            "initial_alpha": INITIAL_ALPHA,
            "initial_beta": INITIAL_BETA,
        })
    elif method == "stan":
        cfg.update({
            "activation": "Self-scalable tanh (Stan)",
            "stan": "sigma_k^i(z) = tanh(z) + beta_k^i * z * tanh(z)",
            "stan_parameterization": "independent trainable beta_k^i for every neuron at every hidden activation location",
            "initial_beta": STAN_INITIAL_BETA,
            "beta_parameters": count_adaptive_parameters(model),
        })
    else:
        cfg["activation"] = "tanh"
    return cfg


def save_checkpoint(path, method, model, optimizer, seed, target_eval, actual_eval, history, elapsed_seconds, latest_metrics, config):
    checkpoint = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "target_checkpoint_eval": int(target_eval),
        "actual_lbfgs_eval": int(actual_eval),
        "lbfgs_eval": int(actual_eval),
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss_history": np.asarray(history, dtype=np.float64),
        "elapsed_seconds": float(elapsed_seconds),
        "latest_metrics": latest_metrics,
        "config": config,
    }
    if method == "nmaf_aat":
        checkpoint["adaptive_parameter_summary"] = adaptive_parameter_summary(model)
    elif method == "stan":
        checkpoint["stan_parameter_summary"] = stan_parameter_summary(model)
    torch.save(checkpoint, path)


def train_single_seed(method, seed):
    set_seed(seed)
    method_dir = RESULT_ROOT / method
    seed_dir = method_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(method)
    points = load_training_points(seed)
    cfg = method_config(method, seed, model)
    save_json(seed_dir / "method_config.json", cfg)

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
    saved = set()
    eval_count = 0
    latest_metrics = None

    synchronize()
    start_time = time.time()

    def maybe_save_checkpoint():
        for checkpoint_eval in CHECKPOINT_EVALS:
            if eval_count >= checkpoint_eval and checkpoint_eval not in saved:
                elapsed = time.time() - start_time
                checkpoint_path = seed_dir / f"burgers_{method}_seed{seed}_eval{checkpoint_eval:05d}.pth"
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
                print(f"Saved passive checkpoint: {checkpoint_path} (actual_eval={eval_count})")

    def closure():
        nonlocal eval_count, latest_metrics
        optimizer.zero_grad(set_to_none=True)
        total_loss, loss_res, loss_bc, loss_ic, data_loss = compute_loss_components(model, points)
        total_loss.backward()
        eval_count += 1
        synchronize()
        elapsed = time.time() - start_time
        latest_metrics = {
            "total_loss": float(total_loss.detach().cpu()),
            "residual_loss": float(loss_res.detach().cpu()),
            "bc_loss": float(loss_bc.detach().cpu()),
            "ic_loss": float(loss_ic.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "elapsed_seconds": float(elapsed),
        }
        history.append([
            eval_count,
            latest_metrics["total_loss"],
            latest_metrics["residual_loss"],
            latest_metrics["data_loss"],
            latest_metrics["bc_loss"],
            latest_metrics["ic_loss"],
            latest_metrics["elapsed_seconds"],
        ])
        if eval_count == 1 or eval_count % PRINT_EVERY == 0:
            print(
                f"{method_label(method):13s} | seed {seed:4d} | "
                f"L-BFGS eval {eval_count:5d} | "
                f"Total={latest_metrics['total_loss']:.6e} | "
                f"Residual={latest_metrics['residual_loss']:.6e} | "
                f"Data={latest_metrics['data_loss']:.6e}"
            )
        maybe_save_checkpoint()
        return total_loss

    optimizer.step(closure)
    total_time = time.time() - start_time

    if latest_metrics is None:
        raise RuntimeError(f"No L-BFGS closure evaluation was performed for method={method}, seed={seed}.")

    for checkpoint_eval in CHECKPOINT_EVALS:
        if checkpoint_eval not in saved:
            checkpoint_path = seed_dir / f"burgers_{method}_seed{seed}_eval{checkpoint_eval:05d}.pth"
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
            print(f"Saved fallback checkpoint: {checkpoint_path} (actual_eval={eval_count})")

    history_array = np.asarray(history, dtype=np.float64)
    np.savetxt(
        seed_dir / "burgers_loss_history.txt",
        history_array,
        header="lbfgs_eval\tTotalLoss\tResidualLoss\tDataLoss\tBCLoss\tICLoss\tElapsedSeconds",
        fmt="%.10e",
        delimiter="\t",
    )
    scipy.io.savemat(
        seed_dir / "burgers_loss_history.mat",
        {"loss_history": history_array, "column_names": ["lbfgs_eval", "TotalLoss", "ResidualLoss", "DataLoss", "BCLoss", "ICLoss", "ElapsedSeconds"]},
    )
    with (seed_dir / "lbfgs_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lbfgs_eval", "total_loss", "residual_loss", "data_loss", "bc_loss", "ic_loss", "elapsed_seconds"])
        writer.writerows(history)

    summary = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "final_actual_lbfgs_eval": int(eval_count),
        "target_lbfgs_eval": int(LBFGS_MAX_EVAL),
        "saved_checkpoints": list(CHECKPOINT_EVALS),
        "total_time_seconds": float(total_time),
        "final_metrics": latest_metrics,
        "training_points_file": str(training_points_path(seed)),
    }
    if method == "nmaf_aat":
        summary["final_adaptive_parameter_summary"] = adaptive_parameter_summary(model)
    elif method == "stan":
        summary["final_stan_parameter_summary"] = stan_parameter_summary(model)
    save_json(seed_dir / "training_summary.json", summary)

    print("-" * 96)
    print(f"{method_label(method)} seed {seed} continuous training completed.")
    print(f"Actual L-BFGS evaluations: {eval_count}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Final total loss: {latest_metrics['total_loss']:.6e}")
    print("-" * 96)
    return summary


def train_all_seeds(method):
    method_dir = RESULT_ROOT / method
    method_dir.mkdir(parents=True, exist_ok=True)
    summaries = [train_single_seed(method, seed) for seed in SEEDS]
    save_json(method_dir / "all_training_summaries.json", {"method": method, "method_label": method_label(method), "seeds": SEEDS, "summaries": summaries})
