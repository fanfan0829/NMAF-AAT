from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
WAVE_ROOT = EXPERIMENT_DIR.parent
OLD_RESULT_ROOT = WAVE_ROOT / "wave_multiseed_results"
OLD_POINTS_DIR = OLD_RESULT_ROOT / "fixed_training_points"
OLD_NMAF_DIR = OLD_RESULT_ROOT / "nmaf_aat"

NEW_RESULT_ROOT = EXPERIMENT_DIR / "results_seedwise_matched_to_nmaf"
TARGET_EXPORT = EXPERIMENT_DIR / "nmaf_seedwise_targets.json"

SEEDS = [2026, 99, 1234]
REPRESENTATIVE_SEED = 99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_ADAM = torch.float32
DTYPE_LBFGS = torch.float64


HIDDEN_LAYERS = 4
ACTUAL_HIDDEN_LAYERS = HIDDEN_LAYERS + 1
HIDDEN_NEURONS = 128
N_RES = 6400
N_BC = 512
N_IC1 = 256
N_IC2 = 256

ADAM_EPOCHS = 500000
ADAM_LR = 1.0e-4

LBFGS_LR = 1.0
LBFGS_HISTORY_SIZE = 50
LBFGS_TOLERANCE_GRAD = 1.0e-9
LBFGS_TOLERANCE_CHANGE = 1.0e-11
LBFGS_LINE_SEARCH_FN = "strong_wolfe"


TIME_CONTROLLED_LBFGS_MAX_ITER = 1
TIME_CONTROLLED_LBFGS_MAX_EVAL = 25

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0


RAR_INTERVAL = 50000
RAR_ROUNDS = 10
RAR_SELECTION_ADAM_STEPS = tuple(
    RAR_INTERVAL * i for i in range(1, RAR_ROUNDS + 1)
)

RAR_SELECTION_ADAM_STEP = RAR_SELECTION_ADAM_STEPS[0]
RAR_CANDIDATES = 100000
RAR_CANDIDATE_BATCH = 10000
RAR_ADD_PER_ROUND = 128
RAR_ADD_N_RES = RAR_ROUNDS * RAR_ADD_PER_ROUND
RAR_FINAL_N_RES = N_RES + RAR_ADD_N_RES
RAR_RNG_OFFSET = 910271

assert N_RES == 6400
assert ADAM_EPOCHS == 500000
assert RAR_SELECTION_ADAM_STEPS == (
    50000, 100000, 150000, 200000, 250000,
    300000, 350000, 400000, 450000, 500000,
)
assert RAR_ADD_N_RES == 1280
assert RAR_FINAL_N_RES == 7680

N_TEST = 200
PRINT_EVERY_ADAM = 1000
PRINT_EVERY_LBFGS_OUTER = 100

FINAL_CHECKPOINT_NAME = "total0700000"
PACKAGE_VERSION = "SEEDWISE_NMAF_MATCHED_V2_FULL10_RAR_20260904"


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


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_dict_to_dtype(state_dict, dtype):
    return {
        k: (v.to(dtype=dtype) if torch.is_tensor(v) else v)
        for k, v in state_dict.items()
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def method_label(method: str) -> str:
    return {
        "tanh": "Standard PINN",
        "rar": "RAR-guided PINN",
        "stan": "Stan-PINN",
        "nmaf_aat": "NMAF-AAT",
    }[method]


def exact_solution(x, t):
    return (
        torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * t)
        + torch.sin(10.0 * np.pi * x) * torch.sin(10.0 * np.pi * t)
    )


class TanhPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(z))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)


class AdaptiveActivation(nn.Module):
    def __init__(self, in_features, dtype,
                 initial_alpha=INITIAL_ALPHA, initial_beta=INITIAL_BETA):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_features, dtype=dtype) * initial_alpha)
        self.beta = nn.Parameter(torch.ones(in_features, dtype=dtype) * initial_beta)

    def forward(self, z):
        X = self.alpha * z + self.beta
        return -X * (torch.exp(-0.5 * X ** 2) - torch.tanh(X))


class NMAFAATPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS,
                 hidden_neurons=HIDDEN_NEURONS, dtype=DTYPE_LBFGS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)

        self.activation = AdaptiveActivation(hidden_neurons, dtype=dtype)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(z))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)


class StanActivation(nn.Module):
    def __init__(self, in_features, dtype, initial_beta=STAN_INITIAL_BETA):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(in_features, dtype=dtype) * initial_beta)

    def forward(self, z):
        tanh_z = torch.tanh(z)
        return tanh_z + self.beta * z * tanh_z


class StanPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS,
                 hidden_neurons=HIDDEN_NEURONS, dtype=DTYPE_ADAM):
        super().__init__()


        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, 1)
        self.stan_activations = nn.ModuleList([
            StanActivation(hidden_neurons, dtype=dtype)
            for _ in range(hidden_layers + 1)
        ])

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        out = self.stan_activations[0](self.input_layer(z))
        for i, layer in enumerate(self.hidden_layers):
            out = self.stan_activations[i + 1](layer(out))
        return self.output_layer(out)


def build_model(method: str, dtype):
    if method in ("tanh", "rar"):
        model = TanhPINN()
    elif method == "stan":
        model = StanPINN(dtype=dtype)
    elif method == "nmaf_aat":
        model = NMAFAATPINN(dtype=dtype)
    else:
        raise ValueError(method)
    return model.to(dtype=dtype, device=DEVICE)


def nmaf_seed_dir(seed: int) -> Path:
    return OLD_NMAF_DIR / f"seed_{int(seed)}"


def nmaf_final_checkpoint_path(seed: int) -> Path:
    return nmaf_seed_dir(seed) / (
        f"wave_nmaf_aat_seed{int(seed)}_{FINAL_CHECKPOINT_NAME}.pth"
    )


def _read_original_nmaf_time(seed: int) -> tuple[float, str]:


    seed_dir = nmaf_seed_dir(seed)
    summary_path = seed_dir / "training_summary.json"
    ckpt_path = nmaf_final_checkpoint_path(seed)

    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        if "total_time_seconds" in obj:
            target = float(obj["total_time_seconds"])
            source = str(summary_path)
        elif isinstance(obj.get("final_metrics"), dict) and "elapsed_seconds" in obj["final_metrics"]:
            target = float(obj["final_metrics"]["elapsed_seconds"])
            source = str(summary_path)
        else:
            target = None
            source = None
        if target is not None:
            if not np.isfinite(target) or target <= 0:
                raise ValueError(f"Invalid NMAF-AAT target time for seed={seed}: {target}")
            return float(target), source

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "elapsed_seconds" in ck:
            target = float(ck["elapsed_seconds"])
        elif isinstance(ck.get("latest_metrics"), dict) and "elapsed_seconds" in ck["latest_metrics"]:
            target = float(ck["latest_metrics"]["elapsed_seconds"])
        else:
            raise KeyError(f"No elapsed time found in {ckpt_path}")
        if not np.isfinite(target) or target <= 0:
            raise ValueError(f"Invalid NMAF-AAT target time for seed={seed}: {target}")
        return float(target), str(ckpt_path)

    raise FileNotFoundError(
        f"Cannot determine original NMAF-AAT final time for seed={seed}.\n"
        f"Expected either:\n  {summary_path}\nor:\n  {ckpt_path}"
    )


def _final_row_from_seed_summary(path: Path):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty NMAF-AAT seed summary: {path}")
    if "total_iteration" in frame.columns:
        vals = pd.to_numeric(frame["total_iteration"], errors="coerce")
        if vals.notna().any():
            return frame.loc[vals.idxmax()]
    if "phase" in frame.columns:
        mask = frame["phase"].astype(str).str.lower().eq("adam_lbfgs")
        if mask.any():
            return frame.loc[mask].iloc[-1]
    return frame.iloc[-1]


@torch.no_grad()
def predict_full_field(model, n_test=N_TEST):
    x = torch.linspace(0.0, 1.0, n_test, dtype=DTYPE_LBFGS, device=DEVICE)
    t = torch.linspace(0.0, 1.0, n_test, dtype=DTYPE_LBFGS, device=DEVICE)
    X, T = torch.meshgrid(x, t, indexing="ij")
    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)
    model.eval()
    pieces = []
    for s in range(0, xf.shape[0], 20000):
        pieces.append(model(xf[s:s+20000], tf[s:s+20000]).cpu().numpy())
    pred = np.vstack(pieces).reshape(n_test, n_test)
    exact = exact_solution(xf, tf).cpu().numpy().reshape(n_test, n_test)
    return (
        x.cpu().numpy(), t.cpu().numpy(), X.cpu().numpy(), T.cpu().numpy(),
        exact, pred,
    )


def relative_l2(u_exact, u_pred):
    ue = np.asarray(u_exact).reshape(-1)
    up = np.asarray(u_pred).reshape(-1)
    return float(np.linalg.norm(ue - up, 2) / np.linalg.norm(ue, 2))


def load_original_nmaf_seed_reference(seed: int) -> dict:


    seed = int(seed)
    target_time, time_source = _read_original_nmaf_time(seed)
    seed_dir = nmaf_seed_dir(seed)
    summary_csv = seed_dir / "seed_summary.csv"
    ckpt_path = nmaf_final_checkpoint_path(seed)

    loss = np.nan
    l2 = np.nan
    max_abs = np.nan
    metric_source = None


    if summary_csv.exists():
        row = _final_row_from_seed_summary(summary_csv)
        if "total_loss" in row.index and pd.notna(row["total_loss"]):
            loss = float(row["total_loss"])
        if "relative_l2_error" in row.index and pd.notna(row["relative_l2_error"]):
            l2 = float(row["relative_l2_error"])
        if "maximum_absolute_error" in row.index and pd.notna(row["maximum_absolute_error"]):
            max_abs = float(row["maximum_absolute_error"])
        metric_source = str(summary_csv)


    if not np.isfinite(loss) or not np.isfinite(l2):
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Need NMAF-AAT final metrics for seed={seed}, but neither a complete\n"
                f"{summary_csv}\nnor final checkpoint\n{ckpt_path}\nis available."
            )
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        latest = ck.get("latest_metrics", {})
        if not np.isfinite(loss):
            if "total_loss" not in latest:
                raise KeyError(f"No total_loss in final NMAF checkpoint for seed={seed}")
            loss = float(latest["total_loss"])
        if not np.isfinite(l2) or not np.isfinite(max_abs):
            model = build_model("nmaf_aat", DTYPE_LBFGS)
            sd = state_dict_to_dtype(ck["model_state_dict"], DTYPE_LBFGS)
            model.load_state_dict(sd, strict=True)
            _, _, _, _, ue, up = predict_full_field(model)
            ae = np.abs(ue - up)
            if not np.isfinite(l2):
                l2 = relative_l2(ue, up)
            if not np.isfinite(max_abs):
                max_abs = float(ae.max())
        metric_source = str(ckpt_path)

    return {
        "seed": seed,
        "reference_method": "nmaf_aat",
        "reference_method_label": "NMAF-AAT",
        "target_wall_time_seconds": float(target_time),
        "nmaf_total_loss": float(loss),
        "nmaf_relative_l2_error": float(l2),
        "nmaf_maximum_absolute_error": float(max_abs),
        "time_source": time_source,
        "metric_source": metric_source,
        "original_nmaf_seed_dir": str(seed_dir),
    }


def load_nmaf_seedwise_targets() -> dict:
    refs = {str(seed): load_original_nmaf_seed_reference(seed) for seed in SEEDS}
    targets = {str(seed): refs[str(seed)]["target_wall_time_seconds"] for seed in SEEDS}
    return {
        "protocol": "seed-wise matched wall-clock budget referenced exclusively to original NMAF-AAT",
        "reference_method": "nmaf_aat",
        "reference_method_label": "NMAF-AAT",
        "seeds": SEEDS,
        "target_wall_time_seconds_by_seed": targets,
        "nmaf_reference_by_seed": refs,
        "IMPORTANT": "There is no averaged stopping target. Every training run uses the same-seed original NMAF-AAT final time.",
    }


def export_nmaf_seedwise_targets() -> dict:
    info = load_nmaf_seedwise_targets()
    save_json(TARGET_EXPORT, info)
    return info


def target_seconds_for_seed(seed: int) -> float:


    ref = load_original_nmaf_seed_reference(int(seed))
    return float(ref["target_wall_time_seconds"])


def training_points_path(seed: int) -> Path:
    return OLD_POINTS_DIR / f"wave_training_points_seed{int(seed)}.npz"


def load_training_points_np(seed: int) -> dict:
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ORIGINAL fixed Wave training points: {path}\n"
            "Do not regenerate points for the efficiency supplement."
        )
    with np.load(path) as data:
        keys = ["x_res", "t_res", "x_bc", "t_bc", "x_ic1", "t_ic1", "x_ic2", "t_ic2"]
        return {k: np.asarray(data[k], dtype=np.float64).copy() for k in keys}


def make_points(base_np: dict, dtype, x_res=None, t_res=None) -> dict:
    x_res = base_np["x_res"] if x_res is None else x_res
    t_res = base_np["t_res"] if t_res is None else t_res
    return {
        "x_res": torch.tensor(x_res, dtype=dtype, device=DEVICE),
        "t_res": torch.tensor(t_res, dtype=dtype, device=DEVICE),
        "x_bc": torch.tensor(base_np["x_bc"], dtype=dtype, device=DEVICE),
        "t_bc": torch.tensor(base_np["t_bc"], dtype=dtype, device=DEVICE),
        "x_ic1": torch.tensor(base_np["x_ic1"], dtype=dtype, device=DEVICE),
        "t_ic1": torch.tensor(base_np["t_ic1"], dtype=dtype, device=DEVICE),
        "x_ic2": torch.tensor(base_np["x_ic2"], dtype=dtype, device=DEVICE),
        "t_ic2": torch.tensor(base_np["t_ic2"], dtype=dtype, device=DEVICE),
    }


def residual(model, x, t, training_graph=True):
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(
        u_t, t, torch.ones_like(u_t), create_graph=training_graph, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, torch.ones_like(u_x), create_graph=training_graph, retain_graph=training_graph
    )[0]
    return u_tt - u_xx


def data_losses(model, pts):
    lbc = torch.mean(model(pts["x_bc"], pts["t_bc"]) ** 2)
    lic1 = torch.mean(model(pts["x_ic1"], pts["t_ic1"]) ** 2)
    x2 = pts["x_ic2"].detach().clone().requires_grad_(True)
    t2 = pts["t_ic2"].detach().clone().requires_grad_(True)
    u2 = model(x2, t2)
    ut2 = torch.autograd.grad(u2, t2, torch.ones_like(u2), create_graph=True, retain_graph=True)[0]
    target = 2*np.pi*torch.sin(2*np.pi*x2) + 10*np.pi*torch.sin(10*np.pi*x2)
    lic2 = torch.mean((ut2 - target) ** 2)
    return lbc, lic1, lic2


def compute_standard_loss(model, pts):
    x = pts["x_res"].detach().clone().requires_grad_(True)
    t = pts["t_res"].detach().clone().requires_grad_(True)
    r = residual(model, x, t, training_graph=True)
    lres = torch.mean(r ** 2)
    lbc, lic1, lic2 = data_losses(model, pts)
    ldata = lbc + lic1 + lic2
    return {
        "optimization_total_loss": lres + ldata,
        "monitor_total_loss": lres + ldata,
        "optimization_residual_loss": lres,
        "monitor_residual_loss": lres,
        "data_loss": ldata,
        "bc_loss": lbc,
        "ic1_loss": lic1,
        "ic2_loss": lic2,
    }


def compute_rar_loss(model, pts, base_n=N_RES):
    x = pts["x_res"].detach().clone().requires_grad_(True)
    t = pts["t_res"].detach().clone().requires_grad_(True)
    r = residual(model, x, t, training_graph=True)
    rsq = r ** 2
    opt_res = torch.mean(rsq)
    mon_res = torch.mean(rsq[:base_n])
    lbc, lic1, lic2 = data_losses(model, pts)
    ldata = lbc + lic1 + lic2
    return {
        "optimization_total_loss": opt_res + ldata,
        "monitor_total_loss": mon_res + ldata,
        "optimization_residual_loss": opt_res,
        "monitor_residual_loss": mon_res,
        "data_loss": ldata,
        "bc_loss": lbc,
        "ic1_loss": lic1,
        "ic2_loss": lic2,
    }


def compute_loss(method, model, pts):
    return compute_rar_loss(model, pts) if method == "rar" else compute_standard_loss(model, pts)


def candidate_pool(seed: int, rar_round: int, n_candidates=RAR_CANDIDATES):


    rng = np.random.default_rng(
        int(seed) + RAR_RNG_OFFSET + 1000003 * int(rar_round)
    )
    return (
        rng.random((n_candidates, 1)).astype(np.float64),
        rng.random((n_candidates, 1)).astype(np.float64),
    )


def select_rar_points(
    model,
    seed: int,
    rar_round: int,
    n_candidates=RAR_CANDIDATES,
    n_add=RAR_ADD_PER_ROUND,
):

    x_all, t_all = candidate_pool(seed, rar_round, n_candidates)
    values = []
    was_training = model.training
    model.eval()
    sync()
    start = time.perf_counter()

    for s0 in range(0, n_candidates, RAR_CANDIDATE_BATCH):
        e = min(s0 + RAR_CANDIDATE_BATCH, n_candidates)
        x = torch.tensor(
            x_all[s0:e], dtype=DTYPE_ADAM, device=DEVICE, requires_grad=True
        )
        t = torch.tensor(
            t_all[s0:e], dtype=DTYPE_ADAM, device=DEVICE, requires_grad=True
        )
        with torch.enable_grad():
            r = residual(model, x, t, training_graph=False)
        values.append(np.abs(r.detach().cpu().numpy()).reshape(-1))

    if was_training:
        model.train()
    sync()
    selection_seconds = time.perf_counter() - start

    abs_r = np.concatenate(values)
    ids = np.argpartition(abs_r, -n_add)[-n_add:]
    ids = ids[np.argsort(abs_r[ids])[::-1]]

    return x_all[ids].copy(), t_all[ids].copy(), {
        "rar_round": int(rar_round),
        "candidate_count": int(n_candidates),
        "selected_count": int(n_add),
        "candidate_abs_residual_mean": float(abs_r.mean()),
        "candidate_abs_residual_max": float(abs_r.max()),
        "selected_abs_residual_mean": float(abs_r[ids].mean()),
        "selected_abs_residual_min": float(abs_r[ids].min()),
        "selected_abs_residual_max": float(abs_r[ids].max()),
        "selection_wall_seconds": float(selection_seconds),
    }


def _metrics_to_float(loss_dict):
    return {k: float(v.detach().cpu()) for k, v in loss_dict.items()}


def _build_time_controlled_lbfgs(model):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=TIME_CONTROLLED_LBFGS_MAX_ITER,
        max_eval=TIME_CONTROLLED_LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn=LBFGS_LINE_SEARCH_FN,
    )


def train_seedwise_matched(method: str, seed: int) -> dict:
    if method not in {"tanh", "rar", "stan"}:
        raise ValueError(
            f"This supplement trains only tanh/RAR/Stan. Got method={method}. "
            "NMAF-AAT is the immutable reference and is never retrained."
        )

    seed = int(seed)
    nmaf_ref = load_original_nmaf_seed_reference(seed)
    target_seconds = float(nmaf_ref["target_wall_time_seconds"])

    out_dir = NEW_RESULT_ROOT / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)


    set_seed(seed)
    model = build_model(method, DTYPE_ADAM)
    base_np = load_training_points_np(seed)
    pts = make_points(base_np, DTYPE_ADAM)
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    x_train = base_np["x_res"].copy()
    t_train = base_np["t_res"].copy()
    source_round = np.zeros(N_RES, dtype=np.int32)

    rar_round_history = []
    cumulative_rar_selection_seconds = 0.0
    rar_meta = None
    if method == "rar":
        rar_meta = {
            "protocol": "full integrated RAR truncated only by same-seed wall-clock budget",
            "base_n_res": int(N_RES),
            "target_rounds": int(RAR_ROUNDS),
            "rar_interval": int(RAR_INTERVAL),
            "selection_adam_steps_target": list(RAR_SELECTION_ADAM_STEPS),
            "add_per_round": int(RAR_ADD_PER_ROUND),
            "target_added_n_res": int(RAR_ADD_N_RES),
            "target_final_n_res": int(RAR_FINAL_N_RES),
            "candidates_per_round": int(RAR_CANDIDATES),
            "score": "abs(u_tt-u_xx)",
            "completed_rounds": 0,
            "completed_selection_adam_steps": [],
            "added_n_res": 0,
            "current_n_res": int(N_RES),
            "cumulative_selection_wall_seconds": 0.0,
            "rounds": rar_round_history,
        }

    adam_steps = 0
    lbfgs_outer_steps = 0
    lbfgs_closure_evals = 0
    stop_phase = None


    sync()
    t0 = time.perf_counter()

    print("=" * 122)
    print(
        f"SEED-WISE NMAF MATCHED TIME | {method_label(method)} | seed={seed} | "
        f"TARGET = original NMAF-AAT seed {seed} time = {target_seconds:.6f} s"
    )
    if method == "rar":
        print(
            f"FULL RAR | every {RAR_INTERVAL} Adam steps | "
            f"up to {RAR_ROUNDS} rounds x {RAR_ADD_PER_ROUND} points | "
            f"{RAR_CANDIDATES} candidates/round | acquisition cost INCLUDED"
        )
    print("=" * 122)

    for ep in range(1, ADAM_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        L = compute_loss(method, model, pts)
        L["optimization_total_loss"].backward()
        optimizer.step()
        sync()
        adam_steps = ep
        elapsed = time.perf_counter() - t0

        if ep == 1 or ep % PRINT_EVERY_ADAM == 0:
            print(
                f"{method_label(method):17s} | seed={seed:4d} | Adam={ep:7d} | "
                f"elapsed={elapsed:10.3f}s / {target_seconds:.3f}s | "
                f"N_RES={len(x_train):4d} | "
                f"monitor={float(L['monitor_total_loss'].detach().cpu()):.6e}"
            )


        if (
            method == "rar"
            and ep in RAR_SELECTION_ADAM_STEPS
            and elapsed < target_seconds
        ):
            rar_round = ep // RAR_INTERVAL
            expected_round = len(rar_round_history) + 1
            if rar_round != expected_round:
                raise RuntimeError(
                    f"RAR round-order mismatch: got round={rar_round}, "
                    f"expected={expected_round}."
                )

            model_id = id(model)
            optimizer_id = id(optimizer)
            state_count = len(optimizer.state)

            xa, ta, round_meta = select_rar_points(model, seed, rar_round)

            n_before = len(x_train)
            x_train = np.vstack([x_train, xa])
            t_train = np.vstack([t_train, ta])
            source_round = np.concatenate([
                source_round,
                np.full(len(xa), rar_round, dtype=np.int32),
            ])

            expected_n = N_RES + rar_round * RAR_ADD_PER_ROUND
            if len(x_train) != expected_n:
                raise RuntimeError(
                    f"RAR round {rar_round}: N_RES={len(x_train)}, expected={expected_n}."
                )
            if not (
                np.array_equal(x_train[:N_RES], base_np["x_res"])
                and np.array_equal(t_train[:N_RES], base_np["t_res"])
            ):
                raise RuntimeError("Original 6400-point RAR prefix changed")

            pts = make_points(base_np, DTYPE_ADAM, x_train, t_train)
            if (
                id(model) != model_id
                or id(optimizer) != optimizer_id
                or len(optimizer.state) != state_count
            ):
                raise RuntimeError("RAR unexpectedly reset model/Adam optimizer state")

            cumulative_rar_selection_seconds += round_meta["selection_wall_seconds"]
            round_record = {
                "rar_round": int(rar_round),
                "adam_step": int(ep),
                "n_res_before": int(n_before),
                "n_res_after": int(len(x_train)),
                **round_meta,
            }
            rar_round_history.append(round_record)

            rar_meta.update({
                "completed_rounds": int(len(rar_round_history)),
                "completed_selection_adam_steps": [
                    int(r["adam_step"]) for r in rar_round_history
                ],
                "added_n_res": int(len(x_train) - N_RES),
                "current_n_res": int(len(x_train)),
                "cumulative_selection_wall_seconds": float(
                    cumulative_rar_selection_seconds
                ),
                "rounds": rar_round_history,
            })

            elapsed = time.perf_counter() - t0
            rar_meta["elapsed_after_latest_selection_seconds"] = float(elapsed)

            save_json(out_dir / "rar_selection_summary.json", rar_meta)
            np.savez_compressed(
                out_dir / f"wave_rar_selected_points_seed{seed}.npz",
                x_res=x_train,
                t_res=t_train,
                source_round=source_round,
                original_shared_count=np.array([N_RES], dtype=np.int32),
                adaptive_added_count=np.array([len(x_train) - N_RES], dtype=np.int32),
                current_n_res=np.array([len(x_train)], dtype=np.int32),
                completed_rar_rounds=np.array([len(rar_round_history)], dtype=np.int32),
                completed_selection_adam_steps=np.asarray(
                    [r["adam_step"] for r in rar_round_history], dtype=np.int32
                ),
            )

            print(
                f"RAR round {rar_round:2d}/{RAR_ROUNDS} at Adam {ep:6d} | "
                f"+{RAR_ADD_PER_ROUND} -> N_RES={len(x_train)} | "
                f"selected |f| mean={round_meta['selected_abs_residual_mean']:.6e} | "
                f"selection={round_meta['selection_wall_seconds']:.3f}s | "
                f"elapsed={elapsed:.3f}s"
            )


        if elapsed >= target_seconds:
            stop_phase = "adam"
            break

    elapsed = time.perf_counter() - t0
    final_model = model
    final_pts = pts


    if method == "rar" and adam_steps >= ADAM_EPOCHS and elapsed < target_seconds:
        if len(rar_round_history) != RAR_ROUNDS:
            raise RuntimeError(
                f"Adam500000 completed but only {len(rar_round_history)} RAR rounds "
                f"were completed; expected {RAR_ROUNDS}."
            )
        if len(x_train) != RAR_FINAL_N_RES:
            raise RuntimeError(
                f"Adam500000 completed but final RAR N_RES={len(x_train)}, "
                f"expected={RAR_FINAL_N_RES}."
            )


    if elapsed < target_seconds and adam_steps >= ADAM_EPOCHS:
        state64 = state_dict_to_dtype(model.state_dict(), DTYPE_LBFGS)
        model64 = build_model(method, DTYPE_LBFGS)
        model64.load_state_dict(state64, strict=True)
        del model, optimizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        final_pts = (
            make_points(base_np, DTYPE_LBFGS, x_train, t_train)
            if method == "rar"
            else make_points(base_np, DTYPE_LBFGS)
        )
        optimizer64 = _build_time_controlled_lbfgs(model64)
        final_model = model64

        while True:
            sync()
            if time.perf_counter() - t0 >= target_seconds:
                stop_phase = "lbfgs"
                break

            def closure():
                nonlocal lbfgs_closure_evals
                optimizer64.zero_grad(set_to_none=True)
                Lb = compute_loss(method, model64, final_pts)
                Lb["optimization_total_loss"].backward()
                lbfgs_closure_evals += 1
                return Lb["optimization_total_loss"]

            optimizer64.step(closure)
            lbfgs_outer_steps += 1
            sync()
            elapsed = time.perf_counter() - t0

            if (
                lbfgs_outer_steps == 1
                or lbfgs_outer_steps % PRINT_EVERY_LBFGS_OUTER == 0
            ):
                print(
                    f"{method_label(method):17s} | seed={seed:4d} | "
                    f"LBFGS outer={lbfgs_outer_steps:7d} | "
                    f"closures={lbfgs_closure_evals:7d} | "
                    f"elapsed={elapsed:10.3f}s / {target_seconds:.3f}s"
                )

            if elapsed >= target_seconds:
                stop_phase = "lbfgs"
                break

    sync()
    actual_training_time = time.perf_counter() - t0


    final_model = final_model.to(dtype=DTYPE_LBFGS, device=DEVICE)
    if stop_phase == "adam":
        final_pts = (
            make_points(base_np, DTYPE_LBFGS, x_train, t_train)
            if method == "rar"
            else make_points(base_np, DTYPE_LBFGS)
        )

    with torch.enable_grad():
        final_losses = compute_loss(method, final_model, final_pts)
    metrics = _metrics_to_float(final_losses)
    x, t, X, T, ue, up = predict_full_field(final_model)
    ae = np.abs(ue - up)
    l2 = relative_l2(ue, up)

    completed_rar_rounds = len(rar_round_history) if method == "rar" else 0
    rar_selected = completed_rar_rounds > 0

    summary = {
        "package_version": PACKAGE_VERSION,
        "protocol": "seed-wise matched wall-clock budget referenced exclusively to original NMAF-AAT",
        "method": method,
        "method_label": method_label(method),
        "seed": seed,
        "reference_method": "nmaf_aat",
        "reference_seed": seed,
        "average_target_used": False,
        "target_wall_time_seconds": target_seconds,
        "target_source": nmaf_ref["time_source"],
        "nmaf_reference_total_loss_same_seed": nmaf_ref["nmaf_total_loss"],
        "nmaf_reference_relative_l2_same_seed": nmaf_ref["nmaf_relative_l2_error"],
        "actual_training_time_seconds": float(actual_training_time),
        "time_overshoot_seconds": float(actual_training_time - target_seconds),
        "time_budget_hit_phase": stop_phase,
        "adam_steps_completed": int(adam_steps),
        "lbfgs_outer_steps_completed": int(lbfgs_outer_steps),
        "lbfgs_closure_evals_completed": int(lbfgs_closure_evals),
        "actual_hidden_layers": int(ACTUAL_HIDDEN_LAYERS),
        "hidden_neurons": int(HIDDEN_NEURONS),
        "final_n_res_training": int(len(x_train) if method == "rar" else N_RES),

        "rar_selected": bool(rar_selected),
        "rar_selection_seconds": (
            float(cumulative_rar_selection_seconds) if method == "rar" else None
        ),

        "rar_protocol": (
            "10-round integrated full RAR during Adam500000, truncated only by wall-clock budget"
            if method == "rar" else None
        ),
        "rar_interval": int(RAR_INTERVAL) if method == "rar" else None,
        "rar_target_rounds": int(RAR_ROUNDS) if method == "rar" else None,
        "rar_completed_rounds": int(completed_rar_rounds) if method == "rar" else None,
        "rar_add_per_round": int(RAR_ADD_PER_ROUND) if method == "rar" else None,
        "rar_target_added_n_res": int(RAR_ADD_N_RES) if method == "rar" else None,
        "rar_completed_added_n_res": (
            int(len(x_train) - N_RES) if method == "rar" else None
        ),
        "rar_candidates_per_round": int(RAR_CANDIDATES) if method == "rar" else None,
        "rar_selection_adam_steps_target": (
            list(RAR_SELECTION_ADAM_STEPS) if method == "rar" else None
        ),
        "rar_selection_adam_steps_completed": (
            [int(r["adam_step"]) for r in rar_round_history]
            if method == "rar" else None
        ),
        "rar_final_7680_reached": (
            bool(len(x_train) == RAR_FINAL_N_RES and completed_rar_rounds == RAR_ROUNDS)
            if method == "rar" else None
        ),
        "monitor_total_loss": metrics["monitor_total_loss"],
        "optimization_total_loss": metrics["optimization_total_loss"],
        "monitor_residual_loss": metrics["monitor_residual_loss"],
        "optimization_residual_loss": metrics["optimization_residual_loss"],
        "data_loss": metrics["data_loss"],
        "bc_loss": metrics["bc_loss"],
        "ic1_loss": metrics["ic1_loss"],
        "ic2_loss": metrics["ic2_loss"],
        "relative_l2_error": float(l2),
        "maximum_absolute_error": float(ae.max()),
        "trainable_parameters": int(count_parameters(final_model)),
        "fixed_training_points_file": str(training_points_path(seed)),
        "metric_evaluation_outside_training_timer": True,
        "IMPORTANT": (
            "No averaged target time was used; target is the same-seed original "
            "NMAF-AAT final time. Corrected RAR uses the same 10-round full-RAR "
            "schedule as the formal Wave experiment, with all reached acquisition "
            "costs included inside the matched-time budget."
        ),
    }

    if method == "rar":

        rar_meta.update({
            "completed_rounds": int(completed_rar_rounds),
            "completed_selection_adam_steps": [
                int(r["adam_step"]) for r in rar_round_history
            ],
            "added_n_res": int(len(x_train) - N_RES),
            "current_n_res": int(len(x_train)),
            "cumulative_selection_wall_seconds": float(cumulative_rar_selection_seconds),
            "time_budget_hit_phase": stop_phase,
            "target_wall_time_seconds": float(target_seconds),
            "actual_training_time_seconds": float(actual_training_time),
            "final_7680_reached": bool(
                len(x_train) == RAR_FINAL_N_RES
                and completed_rar_rounds == RAR_ROUNDS
            ),
        })
        save_json(out_dir / "rar_selection_summary.json", rar_meta)

    save_json(out_dir / "seedwise_matched_summary.json", summary)
    torch.save(
        {"summary": summary, "model_state_dict": cpu_state_dict(final_model)},
        out_dir / f"wave_{method}_seed{seed}_seedwise_nmaf_time_final.pth",
    )
    np.savez_compressed(
        out_dir / f"wave_{method}_seed{seed}_seedwise_nmaf_time_prediction.npz",
        x=x, t=t, X=X, T=T, u_exact=ue, u_pred=up, abs_error=ae,
    )

    print("-" * 122)
    if method == "rar":
        print(
            f"RAR rounds completed = {completed_rar_rounds}/{RAR_ROUNDS} | "
            f"final N_RES={len(x_train)} | "
            f"RAR selection time={cumulative_rar_selection_seconds:.3f}s"
        )
    print(
        f"DONE | {method_label(method)} | seed={seed} | "
        f"NMAF target={target_seconds:.3f}s | actual={actual_training_time:.3f}s | "
        f"loss={summary['monitor_total_loss']:.6e} | L2={l2:.6e}"
    )
    print("-" * 122)
    return summary


def train_all_seeds_seedwise_matched(method: str):
    if method not in {"tanh", "rar", "stan"}:
        raise ValueError("Only tanh, rar, stan are trained in this package.")
    NEW_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = export_nmaf_seedwise_targets()
    print("\nNMAF-AAT seed-wise target map (NO average target is used):")
    for seed in SEEDS:
        print(f"  seed {seed:4d}: {targets['target_wall_time_seconds_by_seed'][str(seed)]:.6f} s")
    summaries = [train_seedwise_matched(method, seed) for seed in SEEDS]
    save_json(
        NEW_RESULT_ROOT / method / "all_seedwise_matched_summaries.json",
        {
            "package_version": PACKAGE_VERSION,
            "method": method,
            "method_label": method_label(method),
            "reference_method": "nmaf_aat",
            "seeds": SEEDS,
            "summaries": summaries,
        },
    )
    return summaries
