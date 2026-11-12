from __future__ import annotations
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SEED = 99

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "allen_cahn_multiseed_results"
POINTS_DIR = RESULT_ROOT / "fixed_training_points"

METHOD = "siren"
METHOD_LABEL = "SIREN"
SEED_DIR = RESULT_ROOT / METHOD / f"seed_{SEED}"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 128

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

OMEGA_0 = 30.0


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sha256_state(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


class Sine(nn.Module):
    def __init__(self, omega_0=OMEGA_0):
        super().__init__()
        self.omega_0 = float(omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


def sine_init(layer):
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.size(1)
        bound = np.sqrt(6.0 / fan_in) / OMEGA_0
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)


def first_layer_sine_init(layer):
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.size(1)
        bound = 1.0 / fan_in
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)


class SirenPINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(2, HIDDEN_NEURONS)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            for _ in range(HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)

        self.sine_activations = nn.ModuleList([
            Sine(OMEGA_0) for _ in range(HIDDEN_LAYERS + 1)
        ])


        self.input_layer.apply(first_layer_sine_init)
        for layer in self.hidden_layers:
            layer.apply(sine_init)


        self.output_layer.apply(sine_init)

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.sine_activations[0](self.input_layer(h))
        for i, layer in enumerate(self.hidden_layers, start=1):
            h = self.sine_activations[i](layer(h))
        return self.output_layer(h)


def load_training_points():
    p = POINTS_DIR / f"allen_cahn_training_points_seed{SEED}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing fixed Allen-Cahn points:\n{p}\n"
            "Do not generate a new point set for this comparison."
        )
    with np.load(p) as d:
        keys = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
        return {k: torch.tensor(d[k], dtype=DTYPE, device=DEVICE) for k in keys}


def compute_loss(model, points):
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

    residual = u_t - 0.0001 * u_xx + 5.0 * u**3 - 5.0 * u
    loss_residual = torch.mean(residual**2)

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

    loss_bc = torch.mean((u_l - u_r)**2) + torch.mean((u_x_l - u_x_r)**2)

    x_ic = points["x_ic"]
    t_ic = points["t_ic"]
    u_ic = model(x_ic, t_ic)
    u_ic_exact = (x_ic**2) * torch.cos(np.pi * x_ic)

    loss_ic_unweighted = torch.mean((u_ic - u_ic_exact)**2)
    loss_ic_weighted = LAMBDA_IC * loss_ic_unweighted

    data_loss = LAMBDA_BC * loss_bc + loss_ic_weighted
    total = LAMBDA_RES * loss_residual + data_loss
    return total, loss_residual, loss_bc, loss_ic_unweighted, loss_ic_weighted, data_loss


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(path, model, optimizer, eval_count, history, elapsed, latest, config):
    torch.save({
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": SEED,
        "target_checkpoint_eval": int(path.stem.split("eval")[-1]),
        "actual_lbfgs_eval": int(eval_count),
        "lbfgs_eval": int(eval_count),
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": np.asarray(history, dtype=np.float64),
        "elapsed_seconds": float(elapsed),
        "latest_metrics": latest,
        "config": config,
    }, path)


def train():
    set_seed(SEED)
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    model = SirenPINN().double().to(DEVICE)
    points = load_training_points()

    config = {
        "method": METHOD,
        "seed": SEED,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_neurons": HIDDEN_NEURONS,
        "n_res": N_RES, "n_bc": N_BC, "n_ic": N_IC,
        "lambda_res": LAMBDA_RES, "lambda_bc": LAMBDA_BC, "lambda_ic": LAMBDA_IC,
        "optimizer": "single continuous LBFGS",
        "lbfgs_lr": LBFGS_LR,
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "lbfgs_max_eval": LBFGS_MAX_EVAL,
        "lbfgs_history_size": LBFGS_HISTORY_SIZE,
        "activation": "SIREN sine",
        "omega_0": OMEGA_0,
        "first_layer_weight_init": "U(-1/fan_in, +1/fan_in)",
        "subsequent_linear_weight_init": "U(-sqrt(6/fan_in)/omega0, +sqrt(6/fan_in)/omega0)",
        "output_layer": "linear",
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "initial_state_sha256": sha256_state(model),
    }
    save_json(SEED_DIR / "method_config.json", config)

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
    saved = set()
    latest = None

    sync()
    start = time.time()

    def maybe_save():
        for target in CHECKPOINT_EVALS:
            if eval_count >= target and target not in saved:
                p = SEED_DIR / f"allen_cahn_{METHOD}_seed{SEED}_eval{target:05d}.pth"
                save_checkpoint(p, model, optimizer, eval_count, history,
                                time.time() - start, latest, config)
                saved.add(target)
                print("Saved:", p)

    def closure():
        nonlocal eval_count, latest
        optimizer.zero_grad(set_to_none=True)
        total, lr, lbc, lic, licw, data = compute_loss(model, points)
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite SIREN loss at closure evaluation {eval_count+1}."
            )
        total.backward()
        eval_count += 1
        sync()
        elapsed = time.time() - start
        latest = {
            "total_loss": float(total.detach().cpu()),
            "residual_loss": float(lr.detach().cpu()),
            "bc_loss": float(lbc.detach().cpu()),
            "ic_loss_unweighted": float(lic.detach().cpu()),
            "ic_loss_weighted": float(licw.detach().cpu()),
            "data_loss": float(data.detach().cpu()),
            "elapsed_seconds": float(elapsed),
        }
        history.append([
            eval_count, latest["total_loss"], latest["residual_loss"],
            latest["data_loss"], latest["bc_loss"], latest["ic_loss_unweighted"],
            latest["ic_loss_weighted"], latest["elapsed_seconds"]
        ])
        if eval_count == 1 or eval_count % PRINT_EVERY == 0:
            print(
                f"{METHOD_LABEL:12s} | seed {SEED} | eval {eval_count:5d} | "
                f"Total={latest['total_loss']:.6e} | Residual={latest['residual_loss']:.6e}"
            )
        maybe_save()
        return total

    optimizer.step(closure)
    total_time = time.time() - start
    sync()

    if latest is None:
        raise RuntimeError("No L-BFGS closure evaluation was performed.")

    for target in CHECKPOINT_EVALS:
        if target not in saved:
            p = SEED_DIR / f"allen_cahn_{METHOD}_seed{SEED}_eval{target:05d}.pth"
            save_checkpoint(p, model, optimizer, eval_count, history,
                            total_time, latest, config)

    np.savetxt(
        SEED_DIR / "lbfgs_history.csv",
        np.asarray(history),
        delimiter=",",
        header="lbfgs_eval,total_loss,residual_loss,data_loss,bc_loss,ic_loss_unweighted,ic_loss_weighted,elapsed_seconds",
        comments=""
    )

    save_json(SEED_DIR / "training_summary.json", {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "seed": SEED,
        "final_actual_lbfgs_eval": int(eval_count),
        "total_time_seconds": float(total_time),
        "final_metrics": latest,
        "initial_state_sha256": config["initial_state_sha256"],
    })

    print("=" * 100)
    print("SIREN completed.")
    print("Final eval:", eval_count)
    print("Final loss:", latest["total_loss"])
    print("Time (s):", total_time)
    print("=" * 100)


if __name__ == "__main__":
    train()
