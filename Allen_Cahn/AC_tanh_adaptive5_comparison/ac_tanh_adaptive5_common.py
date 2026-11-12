from __future__ import annotations

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


EXPERIMENT_DIR = Path(__file__).resolve().parent
AC_ROOT = EXPERIMENT_DIR.parent
REFERENCE_FILE = AC_ROOT / "AC.mat"
FORMAL_RESULT_ROOT = AC_ROOT / "allen_cahn_multiseed_results"
FIXED_POINTS_DIR = FORMAL_RESULT_ROOT / "fixed_training_points"

NEW_FIXED_RESULT_ROOT = EXPERIMENT_DIR / "results_fixed_20000_tanhbase"
NEW_TIME_RESULT_ROOT = EXPERIMENT_DIR / "results_seedwise_tanh_aat_time"
TARGET_EXPORT = EXPERIMENT_DIR / "tanh_aat_seedwise_targets.json"

SEEDS = [2026, 99, 1234]
REPRESENTATIVE_SEED = 99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 128
N_ACTIVATION_SITES = HIDDEN_LAYERS + 1

N_RES = 3000
N_BC = 200
N_IC = 500

LAMBDA_RES = 1.0
LAMBDA_BC = 1.0
LAMBDA_IC = 100.0

LBFGS_LR = 1.0
LBFGS_MAX_ITER = 20000
LBFGS_MAX_EVAL = 20000
LBFGS_HISTORY_SIZE = 200
LBFGS_TOLERANCE_GRAD = 1.0e-7
LBFGS_TOLERANCE_CHANGE = np.finfo(float).eps
LBFGS_LINE_SEARCH_FN = "strong_wolfe"
CHECKPOINT_EVALS = (10000, 20000)
PRINT_EVERY = 100


TIME_CONTROLLED_LBFGS_MAX_ITER = 1
TIME_CONTROLLED_LBFGS_MAX_EVAL = 25
PRINT_EVERY_TIME_OUTER = 100

INITIAL_SCALE = 1.0
INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
N_PRED_BATCH = 65536
PACKAGE_VERSION = "AC_ADAPTIVE5_TANH_BASE_V1_20260907"

METHODS = ["tanh", "gaaf", "llaaf", "nlaaf", "tanh_aat"]
NEW_FIXED_METHODS = ["gaaf", "llaaf", "nlaaf", "tanh_aat"]
TIME_COMPETITORS = ["tanh", "gaaf", "llaaf", "nlaaf"]


def set_seed(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def warmup_cuda() -> None:
    if DEVICE.type != "cuda":
        return
    torch.cuda.init()
    a = torch.ones((128, 128), dtype=DTYPE, device=DEVICE)
    b = torch.ones((128, 128), dtype=DTYPE, device=DEVICE)
    _ = a @ b
    torch.cuda.synchronize()
    del a, b


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def method_label(method: str) -> str:
    return {
        "tanh": "Standard PINN",
        "gaaf": "tanh-GAAF",
        "llaaf": "tanh-L-LAAF",
        "nlaaf": "tanh-N-LAAF",
        "tanh_aat": "tanh-AAT",
    }[method]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_backbone_parameters(model: nn.Module) -> int:
    total = 0
    for module in [model.input_layer, *list(model.hidden_layers), model.output_layer]:
        total += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total


def count_adaptive_parameters(model: nn.Module) -> int:
    return count_parameters(model) - count_backbone_parameters(model)


class TanhPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class GAAFActivation(nn.Module):

    def __init__(self, initial_scale=INITIAL_SCALE):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(initial_scale), dtype=DTYPE))

    def forward(self, z):
        return torch.tanh(self.a * z)


class GAAFPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = GAAFActivation()

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class LLAAFActivation(nn.Module):

    def __init__(self, initial_scale=INITIAL_SCALE):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, dtype=DTYPE) * initial_scale)

    def forward(self, z):
        return torch.tanh(self.a * z)


class LLAAFPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activations = nn.ModuleList([
            LLAAFActivation() for _ in range(hidden_layers + 1)
        ])

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activations[0](self.input_layer(h))
        for i, layer in enumerate(self.hidden_layers, start=1):
            h = self.activations[i](layer(h))
        return self.output_layer(h)


class NLAAFActivation(nn.Module):

    def __init__(self, n_neurons=HIDDEN_NEURONS, initial_scale=INITIAL_SCALE):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_neurons, dtype=DTYPE) * initial_scale)

    def forward(self, z):
        return torch.tanh(self.a * z)


class NLAAFPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activations = nn.ModuleList([
            NLAAFActivation(hidden_neurons) for _ in range(hidden_layers + 1)
        ])

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activations[0](self.input_layer(h))
        for i, layer in enumerate(self.hidden_layers, start=1):
            h = self.activations[i](layer(h))
        return self.output_layer(h)


class TanhAATActivation(nn.Module):

    def __init__(self, in_features=HIDDEN_NEURONS,
                 initial_alpha=INITIAL_ALPHA, initial_beta=INITIAL_BETA):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_alpha)
        self.beta = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_beta)

    def forward(self, z):
        return torch.tanh(self.alpha * z + self.beta)


class TanhAATPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = TanhAATActivation(hidden_neurons)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


def build_model(method: str, device=DEVICE) -> nn.Module:
    if method == "tanh":
        model = TanhPINN()
    elif method == "gaaf":
        model = GAAFPINN()
    elif method == "llaaf":
        model = LLAAFPINN()
    elif method == "nlaaf":
        model = NLAAFPINN()
    elif method == "tanh_aat":
        model = TanhAATPINN()
    else:
        raise ValueError(f"Unknown method: {method}")
    return model.double().to(device)


def adaptive_parameter_summary(model: nn.Module, method: str) -> dict:
    if method == "gaaf":
        return {"global_scale": float(model.activation.a.detach().cpu())}
    if method == "llaaf":
        a = np.array([float(act.a.detach().cpu()) for act in model.activations])
        return {
            "layer_scales": a.tolist(),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()),
        }
    if method == "nlaaf":
        per_layer = [act.a.detach().cpu().numpy().copy() for act in model.activations]
        all_a = np.concatenate(per_layer)
        return {
            "mean": float(all_a.mean()),
            "std": float(all_a.std(ddof=1)) if all_a.size > 1 else 0.0,
            "min": float(all_a.min()), "max": float(all_a.max()),
            "number_of_scales": int(all_a.size),
            "layer_means": [float(v.mean()) for v in per_layer],
            "layer_stds": [float(v.std(ddof=1)) for v in per_layer],
        }
    if method == "tanh_aat":
        alpha = model.activation.alpha.detach().cpu().numpy()
        beta = model.activation.beta.detach().cpu().numpy()
        return {
            "alpha_mean": float(alpha.mean()),
            "alpha_std": float(alpha.std(ddof=1)),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "beta_mean": float(beta.mean()),
            "beta_std": float(beta.std(ddof=1)),
            "beta_min": float(beta.min()),
            "beta_max": float(beta.max()),
        }
    return {}


def training_points_path(seed: int) -> Path:
    return FIXED_POINTS_DIR / f"allen_cahn_training_points_seed{int(seed)}.npz"


def load_training_points_np(seed: int) -> dict:
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ORIGINAL fixed training points: {path}\n"
            "This comparison must reuse the exact formal AC points."
        )
    data = np.load(path)
    keys = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
    return {k: np.asarray(data[k], dtype=np.float64) for k in keys}


def make_points(np_points: dict, device=DEVICE) -> dict:
    return {k: torch.tensor(v, dtype=DTYPE, device=device) for k, v in np_points.items()}


def load_training_points(seed: int, device=DEVICE) -> dict:
    return make_points(load_training_points_np(seed), device=device)


def compute_loss_components(model: nn.Module, points: dict) -> dict:
    x_res = points["x_res"].detach().clone().requires_grad_(True)
    t_res = points["t_res"].detach().clone().requires_grad_(True)
    u = model(x_res, t_res)
    u_t = torch.autograd.grad(u, t_res, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x_res, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    residual = u_t - 0.0001 * u_xx + 5.0 * u ** 3 - 5.0 * u
    loss_residual = torch.mean(residual ** 2)

    x_bc_l = points["x_bc_l"].detach().clone().requires_grad_(True)
    x_bc_r = points["x_bc_r"].detach().clone().requires_grad_(True)
    t_bc = points["t_bc"].detach().clone()
    u_l = model(x_bc_l, t_bc)
    u_r = model(x_bc_r, t_bc)
    u_x_l = torch.autograd.grad(u_l, x_bc_l, torch.ones_like(u_l), create_graph=True, retain_graph=True)[0]
    u_x_r = torch.autograd.grad(u_r, x_bc_r, torch.ones_like(u_r), create_graph=True, retain_graph=True)[0]
    loss_bc = torch.mean((u_l - u_r) ** 2) + torch.mean((u_x_l - u_x_r) ** 2)

    x_ic = points["x_ic"]
    t_ic = points["t_ic"]
    u_ic_pred = model(x_ic, t_ic)
    u_ic_exact = (x_ic ** 2) * torch.cos(np.pi * x_ic)
    loss_ic_unweighted = torch.mean((u_ic_pred - u_ic_exact) ** 2)
    loss_ic_weighted = LAMBDA_IC * loss_ic_unweighted

    data_loss = LAMBDA_BC * loss_bc + loss_ic_weighted
    total_loss = LAMBDA_RES * loss_residual + data_loss
    return {
        "total_loss": total_loss,
        "residual_loss": loss_residual,
        "data_loss": data_loss,
        "bc_loss": loss_bc,
        "ic_loss_unweighted": loss_ic_unweighted,
        "ic_loss_weighted": loss_ic_weighted,
    }


def float_losses(losses: dict) -> dict:
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Missing reference file: {REFERENCE_FILE}")
    data = scipy.io.loadmat(REFERENCE_FILE)
    x = np.asarray(data["x"]).reshape(-1)
    t = np.asarray(data["tt"]).reshape(-1)
    exact = np.asarray(data["uu"], dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")
    if exact.shape != X.shape:
        if exact.T.shape == X.shape:
            exact = exact.T
        else:
            raise ValueError(f"AC.mat uu shape {exact.shape} does not match mesh {X.shape}.")
    return x, t, X, T, exact


def predict(model: nn.Module, X: np.ndarray, T: np.ndarray, shape, device=DEVICE) -> np.ndarray:
    model.eval()
    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xf), N_PRED_BATCH):
            end = min(start + N_PRED_BATCH, len(xf))
            xt = torch.tensor(xf[start:end], dtype=DTYPE, device=device)
            tt = torch.tensor(tf[start:end], dtype=DTYPE, device=device)
            chunks.append(model(xt, tt).detach().cpu().numpy())
    return np.vstack(chunks).reshape(shape)


def relative_l2(exact: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(exact.reshape(-1) - pred.reshape(-1)) / np.linalg.norm(exact.reshape(-1)))


def evaluate_model(model: nn.Module, device=DEVICE) -> dict:
    x, t, X, T, exact = load_reference()
    pred = predict(model, X, T, exact.shape, device=device)
    abs_error = np.abs(exact - pred)
    return {
        "relative_l2_error": relative_l2(exact, pred),
        "maximum_absolute_error": float(abs_error.max()),
        "x": x, "t": t, "X": X, "T": T,
        "u_exact": exact, "u_pred": pred, "abs_error": abs_error,
    }


def formal_seed_dir(method: str, seed: int) -> Path:
    return FORMAL_RESULT_ROOT / method / f"seed_{int(seed)}"


def formal_standard_final_checkpoint(seed: int) -> Path:
    seed_dir = formal_seed_dir("tanh", seed)
    preferred = seed_dir / f"allen_cahn_tanh_seed{int(seed)}_eval20000.pth"
    if preferred.exists():
        return preferred
    candidates = list(seed_dir.glob("*eval20000*.pth")) + list(seed_dir.glob("*20000*.pth"))
    unique, seen = [], set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            unique.append(p); seen.add(key)
    if len(unique) == 1:
        return unique[0]
    raise FileNotFoundError(f"Cannot uniquely locate formal Standard final checkpoint for seed={seed}: {unique}")


def formal_standard_history(seed: int) -> Path:
    p = formal_seed_dir("tanh", seed) / "lbfgs_history.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def fixed_result_dir(method: str, seed: int) -> Path:
    return NEW_FIXED_RESULT_ROOT / method / f"seed_{int(seed)}"


def fixed_checkpoint_path(method: str, seed: int, target_eval=20000) -> Path:
    return fixed_result_dir(method, seed) / f"allen_cahn_{method}_seed{int(seed)}_eval{int(target_eval):05d}.pth"


def method_config(method: str, seed: int, model: nn.Module) -> dict:
    desc = {
        "gaaf": "tanh(a*z), one global trainable scalar a shared by all activation sites",
        "llaaf": "tanh(a_l*z), one trainable scalar per activation site/layer",
        "nlaaf": "tanh(a_lj*z_lj), one trainable scale per neuron per activation site/layer",
        "tanh_aat": "tanh(alpha_j*z_j+beta_j), one shared neuron-wise alpha/beta vector reused at all activation sites",
    }[method]
    return {
        "package_version": PACKAGE_VERSION,
        "method": method,
        "method_label": method_label(method),
        "seed": int(seed),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "hidden_layers_constant": HIDDEN_LAYERS,
        "actual_activation_sites": N_ACTIVATION_SITES,
        "hidden_neurons": HIDDEN_NEURONS,
        "n_res": N_RES, "n_bc": N_BC, "n_ic": N_IC,
        "fixed_training_points_file": str(training_points_path(seed)),
        "optimizer": "single continuous PyTorch L-BFGS",
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "lambda_res": LAMBDA_RES, "lambda_bc": LAMBDA_BC, "lambda_ic": LAMBDA_IC,
        "pde": "u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0",
        "bc": "periodic u and u_x at x=-1 and x=1",
        "ic": "u(0,x)=x^2*cos(pi*x)",
        "base_activation": "tanh",
        "adaptive_definition": desc,
        "initial_scale": INITIAL_SCALE,
        "initial_alpha": INITIAL_ALPHA if method == "tanh_aat" else None,
        "initial_beta": INITIAL_BETA if method == "tanh_aat" else None,
        "slope_recovery_term": False,
        "total_trainable_parameters": count_parameters(model),
        "adaptive_parameters": count_adaptive_parameters(model),
        "fairness_note": "same tanh base activation, same fixed points, same Linear construction order, same optimizer; adaptive parameterization is the intended variable",
    }


def save_fixed_checkpoint(path: Path, method: str, model: nn.Module, optimizer, seed: int,
                          target_eval: int, actual_eval: int, history, elapsed_seconds: float,
                          latest_metrics: dict, config: dict) -> None:
    torch.save({
        "package_version": PACKAGE_VERSION,
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
        "adaptive_parameter_summary": adaptive_parameter_summary(model, method),
        "config": config,
    }, path)


def load_checkpoint_model(method: str, checkpoint_path: Path, device=DEVICE):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(method, device=device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model, ck


def train_fixed_single_seed(method: str, seed: int) -> dict:
    if method not in NEW_FIXED_METHODS:
        raise ValueError(f"Fixed new-method training expects one of {NEW_FIXED_METHODS}")
    seed = int(seed)
    set_seed(seed)
    out_dir = fixed_result_dir(method, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(method)
    points = load_training_points(seed)
    cfg = method_config(method, seed, model)
    save_json(out_dir / "method_config.json", cfg)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER, max_eval=LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE, tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE, line_search_fn=LBFGS_LINE_SEARCH_FN,
    )

    history, saved = [], set()
    eval_count = 0
    latest_metrics = None
    sync(); start_time = time.perf_counter()

    def maybe_save():
        for target in CHECKPOINT_EVALS:
            if eval_count >= target and target not in saved:
                elapsed = time.perf_counter() - start_time
                p = fixed_checkpoint_path(method, seed, target)
                save_fixed_checkpoint(p, method, model, optimizer, seed, target, eval_count,
                                      history, elapsed, latest_metrics, cfg)
                saved.add(target)
                print(f"Saved passive checkpoint: {p} (actual_eval={eval_count})")

    def closure():
        nonlocal eval_count, latest_metrics
        optimizer.zero_grad(set_to_none=True)
        losses = compute_loss_components(model, points)
        losses["total_loss"].backward()
        eval_count += 1
        sync()
        elapsed = time.perf_counter() - start_time
        latest_metrics = float_losses(losses)
        latest_metrics["elapsed_seconds"] = float(elapsed)
        history.append([
            eval_count, latest_metrics["total_loss"], latest_metrics["residual_loss"],
            latest_metrics["data_loss"], latest_metrics["bc_loss"],
            latest_metrics["ic_loss_unweighted"], latest_metrics["ic_loss_weighted"],
            latest_metrics["elapsed_seconds"],
        ])
        if eval_count == 1 or eval_count % PRINT_EVERY == 0:
            print(f"{method_label(method):14s} | seed {seed:4d} | eval {eval_count:5d} | Total={latest_metrics['total_loss']:.6e}")
        maybe_save()
        return losses["total_loss"]

    optimizer.step(closure)
    sync(); returned_time = time.perf_counter() - start_time

    hist = np.asarray(history, dtype=np.float64)
    with (out_dir / "lbfgs_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lbfgs_eval", "total_loss", "residual_loss", "data_loss", "bc_loss",
                    "ic_loss_unweighted", "ic_loss_weighted", "elapsed_seconds"])
        w.writerows(hist)

    if 20000 not in saved:
        early_path = out_dir / f"allen_cahn_{method}_seed{seed}_EARLY_STOP_actual_eval{eval_count:05d}.pth"
        save_fixed_checkpoint(early_path, method, model, optimizer, seed, 20000, eval_count,
                              history, returned_time, latest_metrics or {}, cfg)
        early_summary = {
            "package_version": PACKAGE_VERSION, "method": method, "seed": seed,
            "status": "INCOMPLETE_EVAL_BUDGET", "actual_lbfgs_eval": int(eval_count),
            "required_lbfgs_eval": 20000, "returned_time_seconds": float(returned_time),
            "message": "L-BFGS returned before 20000 closure evaluations. No eval20000 checkpoint was fabricated.",
        }
        save_json(out_dir / "training_summary_INCOMPLETE.json", early_summary)
        raise RuntimeError(
            f"{method_label(method)} seed={seed} stopped at {eval_count} evaluations (<20000). "
            "No valid fixed-20000 result was created."
        )


    target_path = fixed_checkpoint_path(method, seed, 20000)
    target_model, target_ck = load_checkpoint_model(method, target_path)
    target_time = float(target_ck["elapsed_seconds"])
    target_latest = target_ck.get("latest_metrics", {})
    metrics = evaluate_model(target_model)
    np.savez_compressed(
        out_dir / f"allen_cahn_{method}_seed{seed}_prediction_eval20000.npz",
        x=metrics["x"], t=metrics["t"], u_exact=metrics["u_exact"],
        u_pred=metrics["u_pred"], abs_error=metrics["abs_error"],
    )
    summary = {
        "package_version": PACKAGE_VERSION,
        "method": method, "method_label": method_label(method), "seed": seed,
        "status": "COMPLETE",
        "target_lbfgs_eval": 20000,
        "checkpoint_actual_lbfgs_eval": int(target_ck.get("actual_lbfgs_eval", 20000)),
        "optimizer_returned_actual_lbfgs_eval": int(eval_count),
        "total_time_seconds": target_time,
        "optimizer_returned_time_seconds": float(returned_time),
        "final_metrics": target_latest,
        "relative_l2_error": float(metrics["relative_l2_error"]),
        "maximum_absolute_error": float(metrics["maximum_absolute_error"]),
        "metric_evaluation_outside_training_timer": True,
        "adaptive_parameter_summary": target_ck.get("adaptive_parameter_summary", {}),
        "fixed_training_points_file": str(training_points_path(seed)),
    }
    save_json(out_dir / "training_summary.json", summary)
    return summary


def train_fixed_all_seeds(method: str):
    summaries = [train_fixed_single_seed(method, seed) for seed in SEEDS]
    save_json(NEW_FIXED_RESULT_ROOT / method / "all_training_summaries.json", {
        "method": method, "method_label": method_label(method), "seeds": SEEDS, "summaries": summaries,
    })
    return summaries


def fixed_final_checkpoint(method: str, seed: int) -> Path:
    if method == "tanh":
        return formal_standard_final_checkpoint(seed)
    p = fixed_checkpoint_path(method, seed, 20000)
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def fixed_history_path(method: str, seed: int) -> Path:
    if method == "tanh":
        return formal_standard_history(seed)
    p = fixed_result_dir(method, seed) / "lbfgs_history.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_fixed_model(method: str, seed: int, device=DEVICE):
    p = fixed_final_checkpoint(method, seed)
    ck = torch.load(p, map_location=device, weights_only=False)
    model = build_model(method, device=device)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(state, strict=True)
    return model, ck


def read_tanh_aat_target(seed: int) -> tuple[float, str]:
    seed = int(seed)
    summary = fixed_result_dir("tanh_aat", seed) / "training_summary.json"
    if not summary.exists():
        raise FileNotFoundError(
            f"Missing tanh-AAT fixed-20000 summary for seed={seed}: {summary}\n"
            "Run 05_train_tanh_aat_fixed_20000.py before wall-clock experiments."
        )
    obj = json.loads(summary.read_text(encoding="utf-8"))
    if obj.get("status") != "COMPLETE":
        raise RuntimeError(f"Incomplete tanh-AAT reference: {summary}")
    value = float(obj["total_time_seconds"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid tanh-AAT time in {summary}: {value}")
    return value, str(summary)


def load_tanh_aat_reference(seed: int) -> dict:
    seed = int(seed)
    target, source = read_tanh_aat_target(seed)
    model, ck = load_fixed_model("tanh_aat", seed)
    latest = ck.get("latest_metrics", {})
    loss = latest.get("total_loss")
    if loss is None:
        import pandas as pd
        hist = pd.read_csv(fixed_history_path("tanh_aat", seed))
        row = hist.loc[hist["lbfgs_eval"] <= 20000].iloc[-1]
        loss = float(row["total_loss"])
    metrics = evaluate_model(model)
    return {
        "seed": seed,
        "target_wall_time_seconds": float(target),
        "tanh_aat_total_loss": float(loss),
        "tanh_aat_relative_l2_error": float(metrics["relative_l2_error"]),
        "tanh_aat_maximum_absolute_error": float(metrics["maximum_absolute_error"]),
        "time_source": source,
        "metric_source": str(fixed_final_checkpoint("tanh_aat", seed)),
    }


def export_tanh_aat_seedwise_targets() -> dict:
    refs = {str(seed): load_tanh_aat_reference(seed) for seed in SEEDS}
    info = {
        "package_version": PACKAGE_VERSION,
        "protocol": "same-seed tanh-AAT matched wall-clock",
        "reference_method": "tanh_aat",
        "seeds": SEEDS,
        "average_target_used": False,
        "target_wall_time_seconds_by_seed": {str(seed): refs[str(seed)]["target_wall_time_seconds"] for seed in SEEDS},
        "tanh_aat_reference_by_seed": refs,
    }
    save_json(TARGET_EXPORT, info)
    return info


def time_result_dir(method: str, seed: int) -> Path:
    return NEW_TIME_RESULT_ROOT / method / f"seed_{int(seed)}"


def _build_time_lbfgs(model):
    return torch.optim.LBFGS(
        model.parameters(), lr=LBFGS_LR,
        max_iter=TIME_CONTROLLED_LBFGS_MAX_ITER, max_eval=TIME_CONTROLLED_LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE, tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE, line_search_fn=LBFGS_LINE_SEARCH_FN,
    )


def train_seedwise_tanh_aat_time(method: str, seed: int) -> dict:
    if method not in TIME_COMPETITORS:
        raise ValueError(f"Time-controlled competitor must be one of {TIME_COMPETITORS}")
    seed = int(seed)
    target_seconds, target_source = read_tanh_aat_target(seed)
    points = load_training_points(seed)
    out_dir = time_result_dir(method, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_cuda()
    set_seed(seed)
    model = build_model(method)
    optimizer = _build_time_lbfgs(model)

    outer_steps = 0
    closure_evals = 0
    sync(); t0 = time.perf_counter()
    print("=" * 118)
    print(f"AC TANH-AAT-MATCHED TIME | {method_label(method)} | seed={seed} | target={target_seconds:.6f}s")
    print("=" * 118)

    while True:
        sync()
        if time.perf_counter() - t0 >= target_seconds:
            break

        def closure():
            nonlocal closure_evals
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss_components(model, points)
            losses["total_loss"].backward()
            closure_evals += 1
            return losses["total_loss"]

        optimizer.step(closure)
        outer_steps += 1
        sync(); elapsed = time.perf_counter() - t0
        if outer_steps == 1 or outer_steps % PRINT_EVERY_TIME_OUTER == 0:
            print(f"{method_label(method):14s} | seed={seed:4d} | outer={outer_steps:6d} | closures={closure_evals:7d} | {elapsed:.3f}/{target_seconds:.3f}s")
        if elapsed >= target_seconds:
            break

    sync(); actual_time = time.perf_counter() - t0
    with torch.enable_grad():
        losses = float_losses(compute_loss_components(model, points))
    metrics = evaluate_model(model)

    summary = {
        "package_version": PACKAGE_VERSION,
        "protocol": "seed-wise matched wall-clock budget referenced to same-seed tanh-AAT fixed-20000 time",
        "method": method, "method_label": method_label(method), "seed": seed,
        "reference_method": "tanh_aat", "reference_seed": seed,
        "target_wall_time_seconds": float(target_seconds), "target_source": target_source,
        "average_target_used": False, "tanh_aat_retrained_in_time_stage": False,
        "actual_stage_training_time_seconds": float(actual_time),
        "equivalent_end_to_end_time_seconds": float(actual_time),
        "time_overshoot_seconds": float(actual_time - target_seconds),
        "lbfgs_outer_steps_completed": int(outer_steps),
        "lbfgs_closure_evals_completed": int(closure_evals),
        "trainable_parameters": int(count_parameters(model)),
        "adaptive_parameters": int(count_adaptive_parameters(model)),
        "final_n_res_training": N_RES,
        "fixed_training_points_file": str(training_points_path(seed)),
        "monitor_total_loss": float(losses["total_loss"]),
        "monitor_residual_loss": float(losses["residual_loss"]),
        "monitor_data_loss": float(losses["data_loss"]),
        "monitor_bc_loss": float(losses["bc_loss"]),
        "monitor_ic_loss_unweighted": float(losses["ic_loss_unweighted"]),
        "monitor_ic_loss_weighted": float(losses["ic_loss_weighted"]),
        "optimization_total_loss": float(losses["total_loss"]),
        "relative_l2_error": float(metrics["relative_l2_error"]),
        "maximum_absolute_error": float(metrics["maximum_absolute_error"]),
        "metric_evaluation_outside_training_timer": True,
        "adaptive_parameter_summary": adaptive_parameter_summary(model, method),
        "slope_recovery_term": False if method in {"gaaf", "llaaf", "nlaaf"} else None,
        "base_activation": "tanh",
    }
    save_json(out_dir / "seedwise_matched_summary.json", summary)
    torch.save({"summary": summary, "model_state_dict": cpu_state_dict(model)},
               out_dir / f"allen_cahn_{method}_seed{seed}_seedwise_tanh_aat_time_final.pth")
    np.savez_compressed(
        out_dir / f"allen_cahn_{method}_seed{seed}_seedwise_tanh_aat_time_prediction.npz",
        x=metrics["x"], t=metrics["t"], u_exact=metrics["u_exact"],
        u_pred=metrics["u_pred"], abs_error=metrics["abs_error"],
    )
    return summary


def train_all_seeds_seedwise_tanh_aat_time(method: str):
    export_tanh_aat_seedwise_targets()
    summaries = [train_seedwise_tanh_aat_time(method, seed) for seed in SEEDS]
    save_json(NEW_TIME_RESULT_ROOT / method / "all_seedwise_matched_summaries.json", {
        "method": method, "method_label": method_label(method), "seeds": SEEDS,
        "reference_method": "tanh_aat", "summaries": summaries,
    })
    return summaries
