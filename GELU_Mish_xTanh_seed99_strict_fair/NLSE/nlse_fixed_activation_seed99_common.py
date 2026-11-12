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
from scipy.stats import qmc


SEEDS = [99]
REPRESENTATIVE_SEED = 99

RESULT_ROOT = Path("nlse_activation_ablation_seed99_results")
FORMAL_RESULT_ROOT = Path("nlse_multiseed_results")
POINTS_DIR = FORMAL_RESULT_ROOT / "fixed_training_points"
REFERENCE_FILE = Path("NLS.mat")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 100
OUTPUT_DIM = 2

N_RES = 10000
N_BC = 200
N_IC = 200

X_MIN = -5.0
X_MAX = 5.0
T_MIN = 0.0
T_MAX = np.pi / 2.0

LAMBDA_RES = 1.0
LAMBDA_BC = 1.0
LAMBDA_IC = 100.0

LBFGS_LR = 1.0
LBFGS_MAX_EVAL = 10000
LBFGS_MAX_ITER = 10000
LBFGS_HISTORY_SIZE = 50
LBFGS_TOLERANCE_GRAD = 1.0e-7
LBFGS_TOLERANCE_CHANGE = np.finfo(float).eps
LBFGS_LINE_SEARCH_FN = "strong_wolfe"

CHECKPOINT_EVALS = (3000, 6000, 10000)

PRINT_EVERY = 500
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
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)

        out = self.activation(self.input_layer(inputs))

        for layer in self.hidden_layers:
            out = self.activation(layer(out))

        return self.output_layer(out)


class AdaptiveActivation(nn.Module):
    def __init__(
        self,
        in_features,
        initial_alpha=INITIAL_ALPHA,
        initial_beta=INITIAL_BETA,
    ):
        super(AdaptiveActivation, self).__init__()

        self.alpha = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * initial_alpha
        )
        self.beta = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * initial_beta
        )

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
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)


        self.activation = AdaptiveActivation(
            hidden_neurons,
            initial_alpha=INITIAL_ALPHA,
            initial_beta=INITIAL_BETA,
        )

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
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)


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


class FixedActivationPINN(nn.Module):

    def __init__(self, activation_name, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.activation_name = activation_name
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)
        if activation_name == "gelu":
            self.activation = nn.GELU(approximate="none")
        elif activation_name == "mish":
            self.activation = nn.Mish()
        elif activation_name == "xtanh":
            self.activation = None
        else:
            raise ValueError(f"Unknown fixed activation: {activation_name}")

    def _activate(self, z):
        if self.activation_name == "xtanh":
            return z * torch.tanh(z)
        return self.activation(z)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self._activate(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self._activate(layer(out))
        return self.output_layer(out)


def build_model(method):
    if method in {"gelu", "mish", "xtanh"}:
        return FixedActivationPINN(method).double().to(DEVICE)
    raise ValueError(f"Unknown method: {method}")


def method_label(method):
    return {
        "gelu": "GELU",
        "mish": "Mish",
        "xtanh": r"x*tanh(x)",
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
    if not hasattr(model, "activation") or not hasattr(model.activation, "alpha"):
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


def training_points_path(seed: int):
    return POINTS_DIR / f"nlse_training_points_seed{seed}.npz"


def generate_points_for_seed(seed: int):


    sampler_res = qmc.LatinHypercube(d=2, seed=seed + 1000)
    sample_res = sampler_res.random(n=N_RES)
    sample_res = qmc.scale(
        sample_res,
        np.array([X_MIN, T_MIN]),
        np.array([X_MAX, T_MAX]),
    )

    x_res = sample_res[:, 0:1]
    t_res = sample_res[:, 1:2]

    sampler_t = qmc.LatinHypercube(d=1, seed=seed + 2000)
    t_bc = sampler_t.random(n=N_BC) * (T_MAX - T_MIN) + T_MIN
    x_bc_l = np.full((N_BC, 1), X_MIN)
    x_bc_r = np.full((N_BC, 1), X_MAX)

    sampler_x = qmc.LatinHypercube(d=1, seed=seed + 3000)
    x_ic = sampler_x.random(n=N_IC) * (X_MAX - X_MIN) + X_MIN
    t_ic = np.zeros((N_IC, 1))

    return {
        "x_res": x_res.astype(np.float64),
        "t_res": t_res.astype(np.float64),
        "x_bc_l": x_bc_l.astype(np.float64),
        "x_bc_r": x_bc_r.astype(np.float64),
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
        "domain": {
            "x_min": X_MIN,
            "x_max": X_MAX,
            "t_min": T_MIN,
            "t_max": T_MAX,
        },
        "sampling": "fixed LatinHypercube with explicit sampler seeds",
        "files": [],
    }

    for seed in SEEDS:
        path = training_points_path(seed)

        if path.exists() and not overwrite:
            print(f"Existing fixed point file kept: {path}")
        else:
            points = generate_points_for_seed(seed)
            np.savez_compressed(path, **points)
            print(f"Saved fixed training points: {path}")

        manifest["files"].append(str(path))

    save_json(POINTS_DIR / "training_points_manifest.json", manifest)


def load_training_points(seed: int):
    path = training_points_path(seed)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing fixed training points: {path}\n"
            "Run generate_nlse_fixed_training_points_seeds_99_1234_2026.py first."
        )

    data = np.load(path)

    required = {
        "x_res",
        "t_res",
        "x_bc_l",
        "x_bc_r",
        "t_bc",
        "x_ic",
        "t_ic",
    }

    if not required.issubset(data.files):
        missing = required.difference(set(data.files))
        raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")

    return {
        name: torch.tensor(data[name], dtype=DTYPE, device=DEVICE)
        for name in required
    }


def compute_loss_components(model, points):
    x_res = points["x_res"].detach().clone().requires_grad_(True)
    t_res = points["t_res"].detach().clone().requires_grad_(True)

    x_bc_l = points["x_bc_l"].detach().clone().requires_grad_(True)
    x_bc_r = points["x_bc_r"].detach().clone().requires_grad_(True)
    t_bc = points["t_bc"]

    x_ic = points["x_ic"]
    t_ic = points["t_ic"]

    sol = model(x_res, t_res)
    u = sol[:, 0:1]
    v = sol[:, 1:2]

    u_t = torch.autograd.grad(
        u,
        t_res,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x = torch.autograd.grad(
        u,
        x_res,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        u_x,
        x_res,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    v_t = torch.autograd.grad(
        v,
        t_res,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0]
    v_x = torch.autograd.grad(
        v,
        x_res,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0]
    v_xx = torch.autograd.grad(
        v_x,
        x_res,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual_real = -v_t + 0.5 * u_xx + (u ** 2 + v ** 2) * u
    residual_imag = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v

    loss_residual = torch.mean(residual_real ** 2) + torch.mean(residual_imag ** 2)

    sol_l = model(x_bc_l, t_bc)
    sol_r = model(x_bc_r, t_bc)

    u_l = sol_l[:, 0:1]
    v_l = sol_l[:, 1:2]
    u_r = sol_r[:, 0:1]
    v_r = sol_r[:, 1:2]

    u_x_l = torch.autograd.grad(
        u_l,
        x_bc_l,
        grad_outputs=torch.ones_like(u_l),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_x_r = torch.autograd.grad(
        u_r,
        x_bc_r,
        grad_outputs=torch.ones_like(u_r),
        create_graph=True,
        retain_graph=True,
    )[0]
    v_x_l = torch.autograd.grad(
        v_l,
        x_bc_l,
        grad_outputs=torch.ones_like(v_l),
        create_graph=True,
        retain_graph=True,
    )[0]
    v_x_r = torch.autograd.grad(
        v_r,
        x_bc_r,
        grad_outputs=torch.ones_like(v_r),
        create_graph=True,
        retain_graph=True,
    )[0]

    loss_bc = (
        torch.mean((u_l - u_r) ** 2)
        + torch.mean((v_l - v_r) ** 2)
        + torch.mean((u_x_l - u_x_r) ** 2)
        + torch.mean((v_x_l - v_x_r) ** 2)
    )

    sol_ic = model(x_ic, t_ic)
    u_ic_pred = sol_ic[:, 0:1]
    v_ic_pred = sol_ic[:, 1:2]

    u_ic_exact = 2.0 / torch.cosh(x_ic)
    v_ic_exact = torch.zeros_like(x_ic)

    loss_ic_unweighted = (
        torch.mean((u_ic_pred - u_ic_exact) ** 2)
        + torch.mean((v_ic_pred - v_ic_exact) ** 2)
    )
    loss_ic_weighted = LAMBDA_IC * loss_ic_unweighted

    data_loss = LAMBDA_BC * loss_bc + loss_ic_weighted
    total_loss = LAMBDA_RES * loss_residual + data_loss

    return total_loss, loss_residual, loss_bc, loss_ic_unweighted, loss_ic_weighted, data_loss


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def method_config(method, seed, model):
    config = {
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_neurons": HIDDEN_NEURONS,
        "output_dim": OUTPUT_DIM,
        "n_res": N_RES,
        "n_bc": N_BC,
        "n_ic": N_IC,
        "domain": {
            "x_min": X_MIN,
            "x_max": X_MAX,
            "t_min": T_MIN,
            "t_max": T_MAX,
        },
        "lambda_res": LAMBDA_RES,
        "lambda_bc": LAMBDA_BC,
        "lambda_ic": LAMBDA_IC,
        "optimizer": "single continuous LBFGS",
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "checkpoint_policy": "passive save inside closure; optimizer is not restarted",
        "training_points_file": str(training_points_path(seed)),
        "pde": "i*h_t + 0.5*h_xx + |h|^2*h = 0",
        "ic": "h(0,x)=2*sech(x), Im(h)=0",
        "bc": "periodic h(t,-5)=h(t,5), h_x(t,-5)=h_x(t,5)",
        "total_parameters": count_parameters(model),
        "adaptive_parameters": count_adaptive_parameters(model),
    }

    config["activation"] = {
        "gelu": "GELU exact: x*Phi(x), PyTorch nn.GELU(approximate='none')",
        "mish": "Mish: x*tanh(softplus(x)), PyTorch nn.Mish()",
        "xtanh": "x*tanh(x)",
    }[method]
    config["comparison_role"] = "fixed activation only; no trainable activation parameters"
    return config


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

    config = method_config(method, seed, model)
    save_json(seed_dir / "method_config.json", config)

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

                checkpoint_path = (
                    seed_dir
                    / f"nlse_{method}_seed{seed}_eval{checkpoint_eval:05d}.pth"
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
                    config=config,
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
        raise RuntimeError(
            f"No L-BFGS closure evaluation was performed for method={method}, seed={seed}."
        )

    for checkpoint_eval in CHECKPOINT_EVALS:
        if checkpoint_eval not in saved:
            checkpoint_path = (
                seed_dir
                / f"nlse_{method}_seed{seed}_eval{checkpoint_eval:05d}.pth"
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
                config=config,
            )
            saved.add(checkpoint_eval)
            print(
                f"Saved fallback checkpoint: {checkpoint_path} "
                f"(actual_eval={eval_count}, target_eval={checkpoint_eval})"
            )

    history_array = np.asarray(history, dtype=np.float64)

    np.savetxt(
        seed_dir / "nlse_loss_history.txt",
        history_array,
        header=(
            "lbfgs_eval\tTotalLoss\tResidualLoss\tDataLoss\t"
            "BCLoss\tICLossUnweighted\tICLossWeighted\tElapsedSeconds"
        ),
        fmt="%.10e",
        delimiter="\t",
    )

    scipy.io.savemat(
        seed_dir / "nlse_loss_history.mat",
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

    with (seed_dir / "lbfgs_history.csv").open("w", newline="", encoding="utf-8") as f:
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

    all_summaries = []

    for seed in SEEDS:
        all_summaries.append(
            train_single_seed(method, seed)
        )

    save_json(
        method_dir / "all_training_summaries.json",
        {
            "method": method,
            "method_label": method_label(method),
            "seeds": SEEDS,
            "summaries": all_summaries,
        },
    )
