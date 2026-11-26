from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
POINTS_DIR = SCRIPT_DIR / "allen_cahn_multiseed_results" / "fixed_training_points"
RESULT_ROOT = SCRIPT_DIR / "ac_nmaf_sensitivity_results"

SEEDS = [99, 1234, 2026]
SENSITIVITY_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0]

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
PRINT_EVERY = 100


SKIP_COMPLETED = True


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def synchronize() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_value(v: float) -> str:
    return (f"{v:g}").replace(".", "p")


def case_name(amplitude: float, sigma: float) -> str:
    return f"A_{fmt_value(amplitude)}_sigma_{fmt_value(sigma)}"


class NMAFActivation(nn.Module):


    def __init__(self, amplitude: float = 1.0, sigma: float = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.amplitude = float(amplitude)
        self.sigma = float(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = self.amplitude * x * torch.exp(
            -(x ** 2) / (2.0 * self.sigma ** 2)
        )
        return x * torch.tanh(x) - correction


class NMAFPINN(nn.Module):
    def __init__(
        self,
        amplitude: float,
        sigma: float,
        hidden_layers: int = HIDDEN_LAYERS,
        hidden_neurons: int = HIDDEN_NEURONS,
    ):
        super().__init__()

        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = NMAFActivation(amplitude=amplitude, sigma=sigma)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, t], dim=1)
        z = self.activation(self.input_layer(z))
        for layer in self.hidden_layers:
            z = self.activation(layer(z))
        return self.output_layer(z)


def training_points_path(seed: int) -> Path:
    return POINTS_DIR / f"allen_cahn_training_points_seed{seed}.npz"


def verify_existing_points() -> None:
    missing = [training_points_path(s) for s in SEEDS if not training_points_path(s).exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "The sensitivity study is configured to READ EXISTING training points only.\n"
            "Missing file(s):\n" + msg + "\n\n"
            "Place this package beside the existing 'allen_cahn_multiseed_results' folder."
        )


def load_training_points(seed: int):
    data = np.load(training_points_path(seed))
    required = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing arrays in {training_points_path(seed)}: {missing}")
    return {
        k: torch.tensor(data[k], dtype=DTYPE, device=DEVICE)
        for k in required
    }


def compute_loss_components(model: nn.Module, points):
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

    x_ic = points["x_ic"]
    t_ic = points["t_ic"]
    u_ic_pred = model(x_ic, t_ic)
    u_ic_exact = (x_ic ** 2) * torch.cos(np.pi * x_ic)
    loss_ic_unweighted = torch.mean((u_ic_pred - u_ic_exact) ** 2)
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


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_one(amplitude: float, sigma: float, seed: int):
    cname = case_name(amplitude, sigma)
    seed_dir = RESULT_ROOT / cname / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    final_ckpt = seed_dir / "final_checkpoint.pth"
    summary_file = seed_dir / "training_summary.json"
    if SKIP_COMPLETED and final_ckpt.exists() and summary_file.exists():
        print(f"[SKIP completed] {cname}, seed={seed}")
        return json.loads(summary_file.read_text(encoding="utf-8"))


    set_seed(seed)
    model = NMAFPINN(amplitude=amplitude, sigma=sigma).double().to(DEVICE)
    points = load_training_points(seed)

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
    latest_metrics = None

    synchronize()
    start_time = time.time()

    def closure():
        nonlocal eval_count, latest_metrics
        optimizer.zero_grad(set_to_none=True)
        vals = compute_loss_components(model, points)
        total_loss, loss_residual, loss_bc, loss_ic_unweighted, loss_ic_weighted, data_loss = vals
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
                f"{cname:22s} | seed {seed:4d} | eval {eval_count:5d} | "
                f"Total={latest_metrics['total_loss']:.6e}"
            )
        return total_loss

    optimizer.step(closure)
    synchronize()
    total_time = time.time() - start_time

    if latest_metrics is None:
        raise RuntimeError(f"No L-BFGS evaluation for {cname}, seed={seed}.")

    history_arr = np.asarray(history, dtype=np.float64)
    np.savetxt(
        seed_dir / "loss_history.txt",
        history_arr,
        delimiter="\t",
        fmt="%.10e",
        header=(
            "lbfgs_eval\tTotalLoss\tResidualLoss\tDataLoss\tBCLoss\t"
            "ICLossUnweighted\tICLossWeighted\tElapsedSeconds"
        ),
    )

    checkpoint = {
        "model_state_dict": cpu_state_dict(model),
        "seed": int(seed),
        "amplitude": float(amplitude),
        "sigma": float(sigma),
        "actual_lbfgs_eval": int(eval_count),
        "loss_history": history_arr,
        "final_metrics": latest_metrics,
        "config": {
            "activation": "phi(x;A,sigma)=x*tanh(x)-A*x*exp(-x^2/(2*sigma^2))",
            "AAT": False,
            "hidden_layers": HIDDEN_LAYERS,
            "hidden_neurons": HIDDEN_NEURONS,
            "training_points_file": str(training_points_path(seed)),
            "optimizer": "single continuous L-BFGS",
            "max_eval": LBFGS_MAX_EVAL,
            "history_size": LBFGS_HISTORY_SIZE,
            "tolerance_grad": LBFGS_TOLERANCE_GRAD,
            "line_search_fn": LBFGS_LINE_SEARCH_FN,
        },
    }
    torch.save(checkpoint, final_ckpt)

    summary = {
        "case": cname,
        "seed": int(seed),
        "amplitude": float(amplitude),
        "sigma": float(sigma),
        "actual_lbfgs_eval": int(eval_count),
        "final_total_loss": latest_metrics["total_loss"],
        "training_time_seconds": float(total_time),
        "training_points_file": str(training_points_path(seed)),
        "checkpoint": str(final_ckpt),
    }
    save_json(summary_file, summary)
    print(
        f"[DONE] {cname}, seed={seed}, evals={eval_count}, "
        f"loss={latest_metrics['total_loss']:.6e}, time={total_time:.2f}s"
    )
    return summary


def unique_cases():

    cases = []
    for a in SENSITIVITY_VALUES:
        cases.append((float(a), 1.0, "amplitude"))
    for s in SENSITIVITY_VALUES:
        if float(s) != 1.0:
            cases.append((1.0, float(s), "width"))
    return cases


def main():
    verify_existing_points()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    cases = unique_cases()
    manifest = {
        "equation": "Allen-Cahn",
        "AAT": False,
        "activation_family": "NMAF correction-term sensitivity",
        "formula": "phi(x;A,sigma)=x*tanh(x)-A*x*exp(-x^2/(2*sigma^2))",
        "original_NMAF": {"A": 1.0, "sigma": 1.0},
        "study": "one-factor-at-a-time (OFAT)",
        "seeds": SEEDS,
        "sensitivity_values": SENSITIVITY_VALUES,
        "cases": [
            {"A": a, "sigma": s, "branch": branch, "case": case_name(a, s)}
            for a, s, branch in cases
        ],
        "number_of_unique_activation_cases": len(cases),
        "number_of_runs": len(cases) * len(SEEDS),
        "fixed_points_only": True,
        "points_directory": str(POINTS_DIR),
        "device": str(DEVICE),
    }
    save_json(RESULT_ROOT / "training_manifest.json", manifest)

    print("=" * 90)
    print("Allen-Cahn NMAF amplitude/width sensitivity — NO AAT")
    print(f"Device: {DEVICE}")
    print(f"Unique activation cases: {len(cases)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(cases) * len(SEEDS)}")
    print("Existing fixed data points are read directly; none are regenerated.")
    print("=" * 90)

    all_rows = []
    for a, s, branch in cases:
        for seed in SEEDS:
            row = train_one(a, s, seed)
            row["branch"] = branch
            all_rows.append(row)

    csv_path = RESULT_ROOT / "all_training_summaries.csv"
    fieldnames = [
        "branch", "case", "seed", "amplitude", "sigma",
        "actual_lbfgs_eval", "final_total_loss", "training_time_seconds",
        "training_points_file", "checkpoint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("\nAll sensitivity runs completed.")
    print(f"Summary: {csv_path}")
    print("Next: run 02_analyze_AC_NMAF_amplitude_width_sensitivity.py")


if __name__ == "__main__":
    main()
