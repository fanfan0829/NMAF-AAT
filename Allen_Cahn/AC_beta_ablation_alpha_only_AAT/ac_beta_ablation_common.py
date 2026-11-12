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


PACKAGE_DIR = Path(__file__).resolve().parent


def _candidate_roots():
    raw = [
        PACKAGE_DIR.parent,
        PACKAGE_DIR,
        Path.cwd(),
        Path.cwd().parent,
    ]
    raw += list(PACKAGE_DIR.parents[:3])
    raw += list(Path.cwd().resolve().parents[:3])
    seen = set()
    out = []
    for p in raw:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def discover_ac_root() -> Path:
    for root in _candidate_roots():
        points = root / "allen_cahn_multiseed_results" / "fixed_training_points"
        if points.is_dir():
            return root

    return PACKAGE_DIR.parent


AC_ROOT = discover_ac_root()
FORMAL_RESULT_ROOT = AC_ROOT / "allen_cahn_multiseed_results"
FIXED_POINTS_DIR = FORMAL_RESULT_ROOT / "fixed_training_points"
FULL_AAT_RESULT_DIR = FORMAL_RESULT_ROOT / "nmaf_aat"
REFERENCE_FILE = AC_ROOT / "AC.mat"

NEW_RESULT_ROOT = PACKAGE_DIR / "results_alpha_only_aat"
ANALYSIS_DIR = PACKAGE_DIR / "analysis_beta_ablation"

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

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
N_PRED_BATCH = 65536
PACKAGE_VERSION = "AC_BETA_ABLATION_ALPHA_ONLY_AAT_V1_20260907"


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


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_backbone_parameters(model: nn.Module) -> int:
    total = 0
    for module in [model.input_layer, *list(model.hidden_layers), model.output_layer]:
        total += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total


def count_adaptive_parameters(model: nn.Module) -> int:
    return count_parameters(model) - count_backbone_parameters(model)


def nmaf(X: torch.Tensor) -> torch.Tensor:
    return -X * (torch.exp(-0.5 * X ** 2) - torch.tanh(X))


class AlphaOnlyActivation(nn.Module):

    def __init__(self, in_features=HIDDEN_NEURONS, initial_alpha=INITIAL_ALPHA):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * float(initial_alpha)
        )

    def forward(self, z):
        return nmaf(self.alpha * z)


class AlphaOnlyAATPINN(nn.Module):

    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()

        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)

        self.activation = AlphaOnlyActivation(hidden_neurons)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class FullAATActivation(nn.Module):

    def __init__(self, in_features=HIDDEN_NEURONS,
                 initial_alpha=INITIAL_ALPHA, initial_beta=INITIAL_BETA):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * float(initial_alpha)
        )
        self.beta = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * float(initial_beta)
        )

    def forward(self, z):
        return nmaf(self.alpha * z + self.beta)


class FullAATPINN(nn.Module):

    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = FullAATActivation(hidden_neurons)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


def build_alpha_only_model(device=DEVICE):
    return AlphaOnlyAATPINN().double().to(device)


def build_full_aat_model(device=DEVICE):
    return FullAATPINN().double().to(device)


def backbone_state(model: nn.Module) -> dict:
    out = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("input_layer.") or name.startswith("hidden_layers.") or name.startswith("output_layer."):
            out[name] = tensor.detach().cpu().clone()
    return out


def alpha_summary(model: AlphaOnlyAATPINN) -> dict:
    a = model.activation.alpha.detach().cpu().numpy()
    return {
        "alpha_mean": float(a.mean()),
        "alpha_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "alpha_min": float(a.min()),
        "alpha_max": float(a.max()),
        "delta_alpha_l2": float(np.linalg.norm(a - INITIAL_ALPHA, 2)),
    }


def training_points_path(seed: int) -> Path:
    return FIXED_POINTS_DIR / f"allen_cahn_training_points_seed{int(seed)}.npz"


def load_training_points(seed: int, device=DEVICE) -> dict:
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing formal fixed training points: {path}\n"
            "Place this package directly under the Allen-Cahn project root, next to "
            "allen_cahn_multiseed_results/."
        )
    data = np.load(path)
    keys = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
    return {
        key: torch.tensor(np.asarray(data[key], dtype=np.float64), dtype=DTYPE, device=device)
        for key in keys
    }


def compute_loss_components(model: nn.Module, points: dict):
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
    return total_loss, loss_residual, loss_bc, loss_ic_unweighted, loss_ic_weighted, data_loss


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Missing reference field: {REFERENCE_FILE}")
    data = scipy.io.loadmat(REFERENCE_FILE)
    x = np.asarray(data["x"]).reshape(-1)
    t = np.asarray(data["tt"]).reshape(-1)
    exact = np.asarray(data["uu"], dtype=np.float64)
    X, T = np.meshgrid(x, t, indexing="ij")
    if exact.shape != X.shape:
        if exact.T.shape == X.shape:
            exact = exact.T
        else:
            raise ValueError(f"AC.mat uu shape {exact.shape} does not match mesh {X.shape}")
    return x, t, X, T, exact


def predict(model: nn.Module, X: np.ndarray, T: np.ndarray, shape, device=DEVICE):
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


def evaluate_model(model: nn.Module, device=DEVICE) -> dict:
    x, t, X, T, exact = load_reference()
    pred = predict(model, X, T, exact.shape, device=device)
    abs_error = np.abs(exact - pred)
    return {
        "relative_l2_error": float(np.linalg.norm((exact - pred).reshape(-1)) / np.linalg.norm(exact.reshape(-1))),
        "maximum_absolute_error": float(abs_error.max()),
        "x": x,
        "t": t,
        "u_exact": exact,
        "u_pred": pred,
        "abs_error": abs_error,
    }


def alpha_seed_dir(seed: int) -> Path:
    return NEW_RESULT_ROOT / f"seed_{int(seed)}"


def alpha_history_path(seed: int) -> Path:
    return alpha_seed_dir(seed) / "lbfgs_history.csv"


def alpha_checkpoint_path(seed: int, target_eval: int = 20000) -> Path:
    return alpha_seed_dir(seed) / f"allen_cahn_nmaf_aat_alpha_only_seed{int(seed)}_eval{int(target_eval):05d}.pth"


def alpha_summary_path(seed: int) -> Path:
    return alpha_seed_dir(seed) / "training_summary.json"


def full_seed_dir(seed: int) -> Path:
    return FULL_AAT_RESULT_DIR / f"seed_{int(seed)}"


def full_history_path(seed: int) -> Path:
    return full_seed_dir(seed) / "lbfgs_history.csv"


def full_checkpoint_path(seed: int, target_eval: int = 20000) -> Path:
    return full_seed_dir(seed) / f"allen_cahn_nmaf_aat_seed{int(seed)}_eval{int(target_eval):05d}.pth"


def full_summary_path(seed: int) -> Path:
    return full_seed_dir(seed) / "training_summary.json"


def method_config(seed: int, model: nn.Module) -> dict:
    return {
        "package_version": PACKAGE_VERSION,
        "method": "nmaf_aat_alpha_only",
        "method_label": "NMAF-AAT (alpha only)",
        "ablation_target": "beta / shifting parameter",
        "seed": int(seed),
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "hidden_layers_constant": HIDDEN_LAYERS,
        "actual_hidden_activation_sites": N_ACTIVATION_SITES,
        "hidden_neurons": HIDDEN_NEURONS,
        "n_res": N_RES,
        "n_bc": N_BC,
        "n_ic": N_IC,
        "training_points_file": str(training_points_path(seed)),
        "optimizer": "single continuous PyTorch L-BFGS",
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "lbfgs_tolerance_grad": LBFGS_TOLERANCE_GRAD,
        "lbfgs_tolerance_change": LBFGS_TOLERANCE_CHANGE,
        "lbfgs_line_search_fn": LBFGS_LINE_SEARCH_FN,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "lambda_res": LAMBDA_RES,
        "lambda_bc": LAMBDA_BC,
        "lambda_ic": LAMBDA_IC,
        "base_activation": "NMAF: phi(X)=-X*(exp(-0.5*X^2)-tanh(X))",
        "adaptive_definition": "X_j = alpha_j * z_j; one shared neuron-wise alpha vector reused at all activation sites; beta absent",
        "initial_alpha": INITIAL_ALPHA,
        "beta_present": False,
        "slope_recovery_term": False,
        "total_trainable_parameters": count_parameters(model),
        "adaptive_parameters": count_adaptive_parameters(model),
        "fairness_note": "Compared with formal NMAF-AAT, only beta is removed; fixed points, Linear construction order, alpha sharing, network, loss and optimizer settings are unchanged.",
    }


def save_alpha_checkpoint(path: Path, model: nn.Module, optimizer, seed: int,
                          target_eval: int, actual_eval: int, history,
                          elapsed_seconds: float, latest_metrics: dict, config: dict):
    torch.save({
        "method": "nmaf_aat_alpha_only",
        "method_label": "NMAF-AAT (alpha only)",
        "seed": int(seed),
        "target_checkpoint_eval": int(target_eval),
        "actual_lbfgs_eval": int(actual_eval),
        "lbfgs_eval": int(actual_eval),
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss_history": np.asarray(history, dtype=np.float64),
        "elapsed_seconds": float(elapsed_seconds),
        "latest_metrics": latest_metrics,
        "adaptive_parameter_summary": alpha_summary(model),
        "config": config,
    }, path)


def train_alpha_only_single_seed(seed: int) -> dict:
    set_seed(seed)
    seed_dir = alpha_seed_dir(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    model = build_alpha_only_model()
    points = load_training_points(seed)
    cfg = method_config(seed, model)
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

    sync()
    start_time = time.time()

    def maybe_save_checkpoint():
        for target in CHECKPOINT_EVALS:
            if eval_count >= target and target not in saved:
                elapsed = time.time() - start_time
                path = alpha_checkpoint_path(seed, target)
                save_alpha_checkpoint(
                    path, model, optimizer, seed, target, eval_count,
                    history, elapsed, latest_metrics, cfg,
                )
                saved.add(target)
                print(f"Saved passive checkpoint: {path} (actual_eval={eval_count})")

    def closure():
        nonlocal eval_count, latest_metrics
        optimizer.zero_grad(set_to_none=True)
        total, residual, bc, ic_unweighted, ic_weighted, data = compute_loss_components(model, points)
        total.backward()

        eval_count += 1
        sync()
        elapsed = time.time() - start_time
        latest_metrics = {
            "total_loss": float(total.detach().cpu()),
            "residual_loss": float(residual.detach().cpu()),
            "bc_loss": float(bc.detach().cpu()),
            "ic_loss_unweighted": float(ic_unweighted.detach().cpu()),
            "ic_loss_weighted": float(ic_weighted.detach().cpu()),
            "data_loss": float(data.detach().cpu()),
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
                f"alpha-only AAT | seed {seed:4d} | eval {eval_count:5d} | "
                f"Total={latest_metrics['total_loss']:.6e} | "
                f"Residual={latest_metrics['residual_loss']:.6e} | "
                f"Data={latest_metrics['data_loss']:.6e}"
            )
        maybe_save_checkpoint()
        return total

    optimizer.step(closure)
    sync()
    total_time = time.time() - start_time

    if latest_metrics is None:
        raise RuntimeError(f"No L-BFGS closure evaluation was performed for seed={seed}")


    for target in CHECKPOINT_EVALS:
        if target not in saved and eval_count >= target:
            path = alpha_checkpoint_path(seed, target)
            save_alpha_checkpoint(
                path, model, optimizer, seed, target, eval_count,
                history, total_time, latest_metrics, cfg,
            )
            saved.add(target)

    if eval_count < LBFGS_MAX_EVAL:
        final_actual = seed_dir / f"allen_cahn_nmaf_aat_alpha_only_seed{seed}_actual{eval_count:05d}.pth"
        save_alpha_checkpoint(
            final_actual, model, optimizer, seed, eval_count, eval_count,
            history, total_time, latest_metrics, cfg,
        )
        print(
            f"WARNING: L-BFGS terminated before {LBFGS_MAX_EVAL} closure evaluations "
            f"(actual={eval_count}). No fake eval20000 checkpoint was created."
        )

    with alpha_history_path(seed).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lbfgs_eval", "total_loss", "residual_loss", "data_loss",
            "bc_loss", "ic_loss_unweighted", "ic_loss_weighted", "elapsed_seconds",
        ])
        writer.writerows(history)

    summary = {
        "method": "nmaf_aat_alpha_only",
        "method_label": "NMAF-AAT (alpha only)",
        "seed": int(seed),
        "target_lbfgs_eval": LBFGS_MAX_EVAL,
        "final_actual_lbfgs_eval": int(eval_count),
        "reached_20000": bool(eval_count >= 20000),
        "total_time_seconds": float(total_time),
        "final_metrics": latest_metrics,
        "final_adaptive_parameter_summary": alpha_summary(model),
        "training_points_file": str(training_points_path(seed)),
    }
    save_json(alpha_summary_path(seed), summary)
    return summary


def train_alpha_only_all_seeds():
    NEW_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [train_alpha_only_single_seed(seed) for seed in SEEDS]
    save_json(
        NEW_RESULT_ROOT / "all_training_summaries.json",
        {"package_version": PACKAGE_VERSION, "seeds": SEEDS, "summaries": summaries},
    )
    return summaries
