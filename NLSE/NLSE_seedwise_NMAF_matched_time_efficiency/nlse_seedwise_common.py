from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
NLSE_ROOT = EXPERIMENT_DIR.parent
OLD_RESULT_ROOT = NLSE_ROOT / "nlse_multiseed_results"
OLD_POINTS_DIR = OLD_RESULT_ROOT / "fixed_training_points"
OLD_NMAF_DIR = OLD_RESULT_ROOT / "nmaf_aat"
REFERENCE_FILE = NLSE_ROOT / "NLS.mat"

NEW_RESULT_ROOT = EXPERIMENT_DIR / "results_seedwise_matched_to_nmaf"
TARGET_EXPORT = EXPERIMENT_DIR / "nmaf_seedwise_targets.json"
RAR_MANUAL_POINT_DIR = EXPERIMENT_DIR / "rar_preselected_points"

SEEDS = [99, 1234, 2026]
REPRESENTATIVE_SEED = 99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 100
OUTPUT_DIM = 2

N_RES = 10000
N_BC = 200
N_IC = 200
RAR_ADD_N_RES = 2000
RAR_FINAL_N_RES = N_RES + RAR_ADD_N_RES

X_MIN = -5.0
X_MAX = 5.0
T_MIN = 0.0
T_MAX = np.pi / 2.0

LAMBDA_RES = 1.0
LAMBDA_BC = 1.0
LAMBDA_IC = 100.0

LBFGS_LR = 1.0


LBFGS_HISTORY_SIZE = 50
LBFGS_TOLERANCE_GRAD = 1.0e-7
LBFGS_TOLERANCE_CHANGE = np.finfo(float).eps
LBFGS_LINE_SEARCH_FN = "strong_wolfe"


TIME_CONTROLLED_LBFGS_MAX_ITER = 1
TIME_CONTROLLED_LBFGS_MAX_EVAL = 25
PRINT_EVERY_LBFGS_OUTER = 50

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0
FINAL_CHECKPOINT_EVAL = 10000


RAR_METHOD_DIR_OVERRIDE = None


RAR_SELECTION_OVERHEAD_SECONDS_OVERRIDE = {
    99: None,
    1234: None,
    2026: None,
}


REQUIRE_RAR_SELECTION_OVERHEAD = False

N_PRED_BATCH = 65536
PACKAGE_VERSION = "NLSE_SEEDWISE_NMAF_MATCHED_V1_20260901"


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
    a = torch.ones((100, 100), dtype=DTYPE, device=DEVICE)
    b = torch.ones((100, 100), dtype=DTYPE, device=DEVICE)
    _ = a @ b
    torch.cuda.synchronize()
    del a, b


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_dict_to_double(state_dict):
    return {
        k: (v.double() if torch.is_tensor(v) and v.dtype != torch.float64 else v)
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


class TanhPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
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
    def __init__(self, in_features, initial_alpha=INITIAL_ALPHA,
                 initial_beta=INITIAL_BETA):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * initial_alpha
        )
        self.beta = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * initial_beta
        )

    def forward(self, z):
        X = self.alpha * z + self.beta
        return -X * (torch.exp(-0.5 * X ** 2) - torch.tanh(X))


class NMAFAATPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)

        self.activation = AdaptiveActivation(hidden_neurons)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)


class StanActivation(nn.Module):
    def __init__(self, in_features, initial_beta=STAN_INITIAL_BETA):
        super().__init__()
        self.beta = nn.Parameter(
            torch.ones(in_features, dtype=DTYPE) * initial_beta
        )

    def forward(self, z):
        tanh_z = torch.tanh(z)
        return tanh_z + self.beta * z * tanh_z


class StanPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()


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


def build_model(method: str) -> nn.Module:
    if method in ("tanh", "rar"):
        model = TanhPINN()
    elif method == "stan":
        model = StanPINN()
    elif method == "nmaf_aat":
        model = NMAFAATPINN()
    else:
        raise ValueError(method)
    return model.double().to(DEVICE)


def nmaf_seed_dir(seed: int) -> Path:
    return OLD_NMAF_DIR / f"seed_{int(seed)}"


def nmaf_final_checkpoint_candidates(seed: int):
    seed = int(seed)
    seed_dir = nmaf_seed_dir(seed)
    preferred = seed_dir / f"nlse_nmaf_aat_seed{seed}_eval{FINAL_CHECKPOINT_EVAL:05d}.pth"
    candidates = [preferred] if preferred.exists() else []
    if seed_dir.exists():
        for pattern in (
            f"*nmaf*seed{seed}*eval{FINAL_CHECKPOINT_EVAL:05d}*.pth",
            f"*eval{FINAL_CHECKPOINT_EVAL:05d}*.pth",
            f"*{FINAL_CHECKPOINT_EVAL}*.pth",
        ):
            candidates.extend(seed_dir.glob(pattern))
    unique, seen = [], set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def nmaf_final_checkpoint_path(seed: int) -> Path:
    candidates = nmaf_final_checkpoint_candidates(seed)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot locate original NMAF-AAT final checkpoint for seed={seed} in\n"
            f"  {nmaf_seed_dir(seed)}"
        )
    preferred = [p for p in candidates if "nmaf" in p.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    raise RuntimeError(
        f"Multiple candidate NMAF-AAT final checkpoints for seed={seed}:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


def read_original_nmaf_time(seed: int) -> tuple[float, str]:

    seed = int(seed)
    summary_path = nmaf_seed_dir(seed) / "training_summary.json"
    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        candidate = obj.get("total_time_seconds")
        if candidate is None and isinstance(obj.get("final_metrics"), dict):
            candidate = obj["final_metrics"].get("elapsed_seconds")
        if candidate is not None:
            value = float(candidate)
            if np.isfinite(value) and value > 0:
                return value, str(summary_path)

    ckpt_path = nmaf_final_checkpoint_path(seed)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    candidate = ck.get("elapsed_seconds")
    if candidate is None and isinstance(ck.get("latest_metrics"), dict):
        candidate = ck["latest_metrics"].get("elapsed_seconds")
    if candidate is None:
        raise KeyError(f"No elapsed time stored in {ckpt_path}")
    value = float(candidate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid NMAF-AAT time for seed={seed}: {value}")
    return value, str(ckpt_path)


def target_seconds_for_seed(seed: int) -> float:

    value, _ = read_original_nmaf_time(int(seed))
    return float(value)


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Reference file not found: {REFERENCE_FILE}")
    data = scipy.io.loadmat(REFERENCE_FILE)
    if "x" not in data:
        raise KeyError("NLS.mat must contain 'x'.")
    if "tt" in data:
        t_key = "tt"
    elif "t" in data:
        t_key = "t"
    else:
        raise KeyError("NLS.mat must contain 'tt' or 't'.")
    if "uu" not in data:
        raise KeyError("NLS.mat must contain complex reference field 'uu'.")

    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(data[t_key], dtype=np.float64).reshape(-1)
    h_exact = np.asarray(data["uu"])
    target_shape = (x.size, t.size)
    if h_exact.shape != target_shape:
        if h_exact.T.shape == target_shape:
            h_exact = h_exact.T
        else:
            raise ValueError(
                f"Unexpected NLS.mat uu shape {h_exact.shape}; expected "
                f"{target_shape} or transpose."
            )
    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, h_exact


@torch.no_grad()
def predict_complex(model: nn.Module, X, T, output_shape):
    model.eval()
    xf = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    tf = np.asarray(T, dtype=np.float64).reshape(-1, 1)
    pieces = []
    for start in range(0, len(xf), N_PRED_BATCH):
        stop = min(start + N_PRED_BATCH, len(xf))
        xt = torch.as_tensor(xf[start:stop], dtype=DTYPE, device=DEVICE)
        tt = torch.as_tensor(tf[start:stop], dtype=DTYPE, device=DEVICE)
        pieces.append(model(xt, tt).detach().cpu().numpy())
    sol = np.vstack(pieces)
    u = sol[:, 0].reshape(output_shape)
    v = sol[:, 1].reshape(output_shape)
    h = u + 1j * v
    return h, u, v


def relative_l2(h_exact, h_pred) -> float:
    a = np.asarray(h_exact).reshape(-1)
    b = np.asarray(h_pred).reshape(-1)
    denom = np.linalg.norm(a, 2)
    if denom == 0:
        raise ZeroDivisionError("Reference complex field has zero L2 norm.")
    return float(np.linalg.norm(a - b, 2) / denom)


def _final_row_from_seed_summary(path: Path):
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty seed summary: {path}")
    for col in ("checkpoint_eval", "actual_lbfgs_eval", "lbfgs_eval", "total_iteration"):
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce")
            if vals.notna().any():
                return frame.loc[vals.idxmax()]
    return frame.iloc[-1]


def load_original_nmaf_seed_reference(seed: int) -> dict:

    seed = int(seed)
    target_time, time_source = read_original_nmaf_time(seed)
    seed_dir = nmaf_seed_dir(seed)
    summary_csv = seed_dir / "seed_summary.csv"

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

    if not (np.isfinite(loss) and np.isfinite(l2) and np.isfinite(max_abs)):
        ckpt_path = nmaf_final_checkpoint_path(seed)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        latest = ck.get("latest_metrics", {})
        if not np.isfinite(loss):
            if "total_loss" not in latest:
                raise KeyError(f"No final total_loss in {ckpt_path}")
            loss = float(latest["total_loss"])
        if not np.isfinite(l2) or not np.isfinite(max_abs):
            model = build_model("nmaf_aat")
            model.load_state_dict(state_dict_to_double(ck["model_state_dict"]), strict=True)
            _, _, X, T, h_exact = load_reference()
            h_pred, _, _ = predict_complex(model, X, T, h_exact.shape)
            err = np.abs(h_exact - h_pred)
            if not np.isfinite(l2):
                l2 = relative_l2(h_exact, h_pred)
            if not np.isfinite(max_abs):
                max_abs = float(err.max())
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
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
    return {
        "package_version": PACKAGE_VERSION,
        "protocol": "seed-wise matched wall-clock budget referenced exclusively to original NMAF-AAT",
        "reference_method": "nmaf_aat",
        "reference_method_label": "NMAF-AAT",
        "seeds": SEEDS,
        "target_wall_time_seconds_by_seed": {
            str(seed): refs[str(seed)]["target_wall_time_seconds"] for seed in SEEDS
        },
        "nmaf_reference_by_seed": refs,
        "IMPORTANT": (
            "There is NO averaged stopping target. Each competitor seed uses the "
            "actual final wall time of the original NMAF-AAT run with the SAME seed."
        ),
    }


def export_nmaf_seedwise_targets() -> dict:
    info = load_nmaf_seedwise_targets()
    save_json(TARGET_EXPORT, info)
    return info


def training_points_path(seed: int) -> Path:
    return OLD_POINTS_DIR / f"nlse_training_points_seed{int(seed)}.npz"


def load_training_points_np(seed: int) -> dict:
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ORIGINAL fixed NLSE training points: {path}\n"
            "Do not regenerate points for this efficiency supplement."
        )
    with np.load(path, allow_pickle=False) as data:
        keys = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
        missing = [k for k in keys if k not in data.files]
        if missing:
            raise KeyError(f"{path} missing arrays: {missing}")
        return {k: np.asarray(data[k], dtype=np.float64).copy() for k in keys}


def make_points(base_np: dict, x_res=None, t_res=None) -> dict:
    xr = base_np["x_res"] if x_res is None else np.asarray(x_res, dtype=np.float64)
    tr = base_np["t_res"] if t_res is None else np.asarray(t_res, dtype=np.float64)
    return {
        "x_res": torch.tensor(xr, dtype=DTYPE, device=DEVICE),
        "t_res": torch.tensor(tr, dtype=DTYPE, device=DEVICE),
        "x_bc_l": torch.tensor(base_np["x_bc_l"], dtype=DTYPE, device=DEVICE),
        "x_bc_r": torch.tensor(base_np["x_bc_r"], dtype=DTYPE, device=DEVICE),
        "t_bc": torch.tensor(base_np["t_bc"], dtype=DTYPE, device=DEVICE),
        "x_ic": torch.tensor(base_np["x_ic"], dtype=DTYPE, device=DEVICE),
        "t_ic": torch.tensor(base_np["t_ic"], dtype=DTYPE, device=DEVICE),
    }


def compute_loss_components(model: nn.Module, points: dict):
    x_res = points["x_res"].detach().clone().requires_grad_(True)
    t_res = points["t_res"].detach().clone().requires_grad_(True)

    sol = model(x_res, t_res)
    u = sol[:, 0:1]
    v = sol[:, 1:2]

    u_t = torch.autograd.grad(
        u, t_res, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u, x_res, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_res, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0]

    v_t = torch.autograd.grad(
        v, t_res, torch.ones_like(v), create_graph=True, retain_graph=True
    )[0]
    v_x = torch.autograd.grad(
        v, x_res, torch.ones_like(v), create_graph=True, retain_graph=True
    )[0]
    v_xx = torch.autograd.grad(
        v_x, x_res, torch.ones_like(v_x), create_graph=True, retain_graph=True
    )[0]

    residual_real = -v_t + 0.5 * u_xx + (u ** 2 + v ** 2) * u
    residual_imag = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
    loss_residual = (
        torch.mean(residual_real ** 2) + torch.mean(residual_imag ** 2)
    )

    x_bc_l = points["x_bc_l"].detach().clone().requires_grad_(True)
    x_bc_r = points["x_bc_r"].detach().clone().requires_grad_(True)
    t_bc = points["t_bc"].detach().clone()

    sol_l = model(x_bc_l, t_bc)
    sol_r = model(x_bc_r, t_bc)
    u_l, v_l = sol_l[:, 0:1], sol_l[:, 1:2]
    u_r, v_r = sol_r[:, 0:1], sol_r[:, 1:2]

    u_x_l = torch.autograd.grad(
        u_l, x_bc_l, torch.ones_like(u_l), create_graph=True, retain_graph=True
    )[0]
    u_x_r = torch.autograd.grad(
        u_r, x_bc_r, torch.ones_like(u_r), create_graph=True, retain_graph=True
    )[0]
    v_x_l = torch.autograd.grad(
        v_l, x_bc_l, torch.ones_like(v_l), create_graph=True, retain_graph=True
    )[0]
    v_x_r = torch.autograd.grad(
        v_r, x_bc_r, torch.ones_like(v_r), create_graph=True, retain_graph=True
    )[0]

    loss_bc = (
        torch.mean((u_l - u_r) ** 2)
        + torch.mean((v_l - v_r) ** 2)
        + torch.mean((u_x_l - u_x_r) ** 2)
        + torch.mean((v_x_l - v_x_r) ** 2)
    )

    x_ic = points["x_ic"]
    t_ic = points["t_ic"]
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

    return {
        "total_loss": total_loss,
        "residual_loss": loss_residual,
        "data_loss": data_loss,
        "bc_loss": loss_bc,
        "ic_loss_unweighted": loss_ic_unweighted,
        "ic_loss_weighted": loss_ic_weighted,
    }


def _float_losses(losses: dict) -> dict:
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def locate_formal_rar_dir() -> Path:
    if RAR_METHOD_DIR_OVERRIDE is not None:
        p = OLD_RESULT_ROOT / str(RAR_METHOD_DIR_OVERRIDE)
        if not p.is_dir():
            raise FileNotFoundError(f"RAR_METHOD_DIR_OVERRIDE not found: {p}")
        return p

    excluded = {
        "tanh", "nmaf_aat", "stan", "fixed_training_points",
        "four_method_unified_analysis",
    }
    candidates = []
    if OLD_RESULT_ROOT.exists():
        for p in OLD_RESULT_ROOT.iterdir():
            if not p.is_dir() or p.name in excluded:
                continue
            if all((p / f"seed_{s}" / "fixed_monitor_history.csv").exists() for s in SEEDS):
                candidates.append(p)
    rar_named = [p for p in candidates if "rar" in p.name.lower()]
    if len(rar_named) == 1:
        return rar_named[0]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            "Could not locate the ORIGINAL formal NLSE RAR result directory.\n"
            "Expected a child of nlse_multiseed_results containing seed_99, "
            "seed_1234, seed_2026 with fixed_monitor_history.csv."
        )
    raise RuntimeError(
        "Multiple formal NLSE RAR directories found. Set RAR_METHOD_DIR_OVERRIDE "
        "in nlse_seedwise_common.py to one of:\n" +
        "\n".join(f"  {p.name}" for p in candidates)
    )


def _contains_all_base_points(x_full, t_full, base_np: dict) -> bool:


    full = np.column_stack([x_full.reshape(-1), t_full.reshape(-1)])
    base = np.column_stack([
        base_np["x_res"].reshape(-1), base_np["t_res"].reshape(-1)
    ])
    if len(full) < len(base):
        return False
    if np.allclose(full[:len(base)], base, rtol=0.0, atol=1.0e-12):
        return True


    full_set = {tuple(row) for row in np.round(full, 7)}
    return all(tuple(row) in full_set for row in np.round(base, 7))


def _try_extract_rar_arrays(npz_path: Path, base_np: dict):
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            full_pairs = [
                ("x_res", "t_res"), ("x_train", "t_train"),
                ("x_all", "t_all"), ("x_pde", "t_pde"),
            ]
            for xk, tk in full_pairs:
                if xk in d.files and tk in d.files:
                    x = np.asarray(d[xk], dtype=np.float64).reshape(-1, 1)
                    t = np.asarray(d[tk], dtype=np.float64).reshape(-1, 1)
                    if len(x) == RAR_FINAL_N_RES and len(t) == RAR_FINAL_N_RES:
                        if not _contains_all_base_points(x, t, base_np):
                            continue
                        return x.copy(), t.copy(), "full_12000"

            added_pairs = [
                ("x_added", "t_added"), ("x_new", "t_new"),
                ("x_selected", "t_selected"), ("x_adaptive", "t_adaptive"),
            ]
            for xk, tk in added_pairs:
                if xk in d.files and tk in d.files:
                    xa = np.asarray(d[xk], dtype=np.float64).reshape(-1, 1)
                    ta = np.asarray(d[tk], dtype=np.float64).reshape(-1, 1)
                    if len(xa) == RAR_ADD_N_RES and len(ta) == RAR_ADD_N_RES:
                        return (
                            np.vstack([base_np["x_res"], xa]),
                            np.vstack([base_np["t_res"], ta]),
                            "base10000_plus_added2000",
                        )
    except Exception:
        return None
    return None


def find_formal_rar_points(seed: int, base_np: dict):

    seed = int(seed)
    manual = RAR_MANUAL_POINT_DIR / f"nlse_rar_selected_points_seed{seed}.npz"
    search_files = []
    if manual.exists():
        search_files.append(manual)

    rar_dir = locate_formal_rar_dir()
    seed_dir = rar_dir / f"seed_{seed}"
    if seed_dir.exists():
        search_files.extend(seed_dir.rglob("*.npz"))
    search_files.extend(rar_dir.glob(f"*seed{seed}*.npz"))
    search_files.extend(rar_dir.glob(f"*seed_{seed}*.npz"))

    unique, seen = [], set()
    for p in search_files:
        rp = str(p.resolve())
        if rp not in seen:
            unique.append(p)
            seen.add(rp)

    matches = []
    for p in unique:
        extracted = _try_extract_rar_arrays(p, base_np)
        if extracted is not None:
            x, t, mode = extracted
            matches.append((p, x, t, mode))

    if len(matches) == 1:
        p, x, t, mode = matches[0]
        return x, t, str(p), mode
    if len(matches) > 1:
        named = [
            m for m in matches
            if any(k in m[0].name.lower() for k in ("selected", "point", "rar"))
        ]
        if len(named) == 1:
            p, x, t, mode = named[0]
            return x, t, str(p), mode
        raise RuntimeError(
            f"Multiple 12000-point RAR files found for seed={seed}.\n"
            "Place the intended formal set explicitly at:\n"
            f"  {manual}\nCandidates:\n" +
            "\n".join(f"  {m[0]}" for m in matches)
        )

    raise FileNotFoundError(
        f"Could not find the ORIGINAL formal NLSE RAR 12000-point set for seed={seed}.\n"
        f"RAR directory detected: {rar_dir}\n"
        "This supplement must reuse the formal selected set rather than invent a new selection rule.\n"
        "If necessary, copy the formal set to:\n"
        f"  {manual}\n"
        "with arrays x_res/t_res of length 12000 (or x_added/t_added of length 2000)."
    )


_SELECTION_KEYS = {
    "selection_wall_seconds",
    "selection_seconds",
    "rar_selection_seconds",
    "point_selection_seconds",
    "selection_time_seconds",
    "rar_selection_time_seconds",
    "stage_a_selection_seconds",
    "stage_a_time_seconds",
}


def _search_numeric_key(obj, path=""):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k).lower() in _SELECTION_KEYS:
                try:
                    value = float(v)
                    if np.isfinite(value) and value >= 0:
                        found.append((value, here))
                except Exception:
                    pass
            found.extend(_search_numeric_key(v, here))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_search_numeric_key(v, f"{path}[{i}]"))
    return found


def rar_selection_overhead(seed: int) -> tuple[float, str, bool]:

    seed = int(seed)
    override = RAR_SELECTION_OVERHEAD_SECONDS_OVERRIDE.get(seed)
    if override is not None:
        value = float(override)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid RAR selection overhead override for seed={seed}: {override}"
            )
        return value, "RAR_SELECTION_OVERHEAD_SECONDS_OVERRIDE", True

    try:
        rar_dir = locate_formal_rar_dir()
    except Exception:
        if REQUIRE_RAR_SELECTION_OVERHEAD:
            raise
        return 0.0, "not_available", False

    candidates = []
    for root in (rar_dir / f"seed_{seed}", rar_dir):
        if root.exists():
            candidates.extend(root.glob("*.json"))
            candidates.extend(root.glob(f"*seed{seed}*.json"))
            candidates.extend(root.glob(f"*seed_{seed}*.json"))

    seen, hits = set(), []
    for p in candidates:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for value, keypath in _search_numeric_key(obj):
            hits.append((value, f"{p}:{keypath}"))

    positive = [h for h in hits if h[0] > 0]
    if len(positive) == 1:
        return float(positive[0][0]), positive[0][1], True
    if len(positive) > 1:
        vals = np.asarray([h[0] for h in positive], dtype=float)
        if np.allclose(vals, vals[0], rtol=1e-10, atol=1e-10):
            return float(vals[0]), "repeated identical metadata: " + positive[0][1], True
        if REQUIRE_RAR_SELECTION_OVERHEAD:
            raise RuntimeError(
                f"Multiple inconsistent RAR selection-time values for seed={seed}: {positive}"
            )

    if REQUIRE_RAR_SELECTION_OVERHEAD:
        raise FileNotFoundError(
            f"No recorded RAR selection/Stage-A wall time for seed={seed}. "
            "Set RAR_SELECTION_OVERHEAD_SECONDS_OVERRIDE in nlse_seedwise_common.py."
        )
    return 0.0, "not_available; optimistic Stage-B-only timing", False


def _build_time_controlled_lbfgs(model: nn.Module):
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
            f"This supplement trains only tanh/RAR/Stan; got {method}. "
            "NMAF-AAT is the immutable reference and is never retrained."
        )

    seed = int(seed)
    target_seconds, target_source = read_original_nmaf_time(seed)
    base_np = load_training_points_np(seed)

    rar_overhead = 0.0
    rar_overhead_source = None
    rar_overhead_known = None
    rar_points_source = None
    rar_points_mode = None

    if method == "rar":
        x_train, t_train, rar_points_source, rar_points_mode = find_formal_rar_points(seed, base_np)
        rar_overhead, rar_overhead_source, rar_overhead_known = rar_selection_overhead(seed)
        stage_training_budget = target_seconds - rar_overhead
        if stage_training_budget <= 0:
            raise RuntimeError(
                f"RAR selection overhead ({rar_overhead:.6f}s) is >= NMAF target "
                f"({target_seconds:.6f}s) for seed={seed}."
            )
        train_points = make_points(base_np, x_train, t_train)
        monitor_points = make_points(base_np)
    else:
        stage_training_budget = target_seconds
        train_points = make_points(base_np)
        monitor_points = train_points

    out_dir = NEW_RESULT_ROOT / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)


    warmup_cuda()
    set_seed(seed)
    model = build_model(method)
    optimizer = _build_time_controlled_lbfgs(model)

    lbfgs_outer_steps = 0
    lbfgs_closure_evals = 0

    sync()
    t0 = time.perf_counter()

    print("=" * 124)
    print(
        f"NLSE SEED-WISE NMAF MATCHED TIME | {method_label(method)} | seed={seed} | "
        f"NMAF TARGET={target_seconds:.6f}s | TRAINING BUDGET={stage_training_budget:.6f}s"
    )
    if method == "rar":
        print(
            f"RAR formal points: {rar_points_source} | selection overhead applied="
            f"{rar_overhead:.6f}s | known={rar_overhead_known}"
        )
    print("=" * 124)

    while True:
        sync()
        elapsed = time.perf_counter() - t0
        if elapsed >= stage_training_budget:
            break

        def closure():
            nonlocal lbfgs_closure_evals
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss_components(model, train_points)
            losses["total_loss"].backward()
            lbfgs_closure_evals += 1
            return losses["total_loss"]

        optimizer.step(closure)
        lbfgs_outer_steps += 1
        sync()
        elapsed = time.perf_counter() - t0

        if lbfgs_outer_steps == 1 or lbfgs_outer_steps % PRINT_EVERY_LBFGS_OUTER == 0:
            print(
                f"{method_label(method):18s} | seed={seed:4d} | "
                f"outer={lbfgs_outer_steps:6d} | closures={lbfgs_closure_evals:7d} | "
                f"stage elapsed={elapsed:10.3f}s / {stage_training_budget:.3f}s"
            )

        if elapsed >= stage_training_budget:
            break

    sync()
    actual_stage_training_time = time.perf_counter() - t0
    equivalent_end_to_end_time = actual_stage_training_time + (
        rar_overhead if method == "rar" else 0.0
    )


    with torch.enable_grad():
        opt_losses = _float_losses(compute_loss_components(model, train_points))
        if method == "rar":
            monitor_losses = _float_losses(compute_loss_components(model, monitor_points))
        else:
            monitor_losses = opt_losses.copy()

    x, t, X, T, h_exact = load_reference()
    h_pred, u_pred, v_pred = predict_complex(model, X, T, h_exact.shape)
    complex_abs_error = np.abs(h_exact - h_pred)
    magnitude_error = np.abs(np.abs(h_exact) - np.abs(h_pred))
    l2 = relative_l2(h_exact, h_pred)

    if method == "rar":
        timing_scope = (
            "equivalent end-to-end: recorded/overridden RAR selection overhead + Stage-B training"
            if rar_overhead_known else
            "optimistic Stage-B-only: formal RAR selection overhead unavailable and treated as 0"
        )
        final_n_res = RAR_FINAL_N_RES
    else:
        timing_scope = "training wall time"
        final_n_res = N_RES

    summary = {
        "package_version": PACKAGE_VERSION,
        "protocol": "seed-wise matched wall-clock budget referenced exclusively to original NMAF-AAT",
        "method": method,
        "method_label": method_label(method),
        "seed": seed,
        "reference_method": "nmaf_aat",
        "reference_seed": seed,
        "target_wall_time_seconds": float(target_seconds),
        "target_source": target_source,
        "average_target_used": False,
        "nmaf_retrained": False,
        "actual_stage_training_time_seconds": float(actual_stage_training_time),
        "equivalent_end_to_end_time_seconds": float(equivalent_end_to_end_time),
        "time_overshoot_seconds": float(equivalent_end_to_end_time - target_seconds),
        "lbfgs_outer_steps_completed": int(lbfgs_outer_steps),
        "lbfgs_closure_evals_completed": int(lbfgs_closure_evals),
        "lbfgs_history_size": int(LBFGS_HISTORY_SIZE),
        "trainable_parameters": int(count_parameters(model)),
        "final_n_res_training": int(final_n_res),
        "fixed_training_points_file": str(training_points_path(seed)),

        "monitor_total_loss": float(monitor_losses["total_loss"]),
        "monitor_residual_loss": float(monitor_losses["residual_loss"]),
        "monitor_data_loss": float(monitor_losses["data_loss"]),
        "monitor_bc_loss": float(monitor_losses["bc_loss"]),
        "monitor_ic_loss_unweighted": float(monitor_losses["ic_loss_unweighted"]),
        "monitor_ic_loss_weighted": float(monitor_losses["ic_loss_weighted"]),

        "optimization_total_loss": float(opt_losses["total_loss"]),
        "optimization_residual_loss": float(opt_losses["residual_loss"]),

        "relative_l2_error": float(l2),
        "maximum_absolute_error": float(complex_abs_error.max()),
        "metric_evaluation_outside_training_timer": True,
        "rar_points_source": rar_points_source,
        "rar_points_mode": rar_points_mode,
        "rar_selection_overhead_seconds_applied": (
            float(rar_overhead) if method == "rar" else None
        ),
        "rar_selection_overhead_source": rar_overhead_source,
        "rar_selection_overhead_known": rar_overhead_known,
        "rar_timing_scope": timing_scope if method == "rar" else None,
        "IMPORTANT": (
            "No averaged target time was used. The target is the original NMAF-AAT "
            "final wall time for this exact seed."
        ),
    }

    save_json(out_dir / "seedwise_matched_summary.json", summary)
    torch.save(
        {"summary": summary, "model_state_dict": cpu_state_dict(model)},
        out_dir / f"nlse_{method}_seed{seed}_seedwise_nmaf_time_final.pth",
    )
    np.savez_compressed(
        out_dir / f"nlse_{method}_seed{seed}_seedwise_nmaf_time_prediction.npz",
        x=x, t=t, X=X, T=T,
        h_exact=h_exact, h_pred=h_pred, u_pred=u_pred, v_pred=v_pred,
        complex_abs_error=complex_abs_error,
        magnitude_error=magnitude_error,
    )

    print("-" * 124)
    print(
        f"DONE | {method_label(method)} | seed={seed} | NMAF target={target_seconds:.3f}s | "
        f"equivalent time={equivalent_end_to_end_time:.3f}s | "
        f"common loss={summary['monitor_total_loss']:.6e} | L2={l2:.6e}"
    )
    print("-" * 124)
    return summary


def train_all_seeds_seedwise_matched(method: str):
    if method not in {"tanh", "rar", "stan"}:
        raise ValueError("Only tanh, rar and stan are trained by this supplement.")
    NEW_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = export_nmaf_seedwise_targets()
    print("\nNMAF-AAT seed-wise target map (NO average target is used):")
    for seed in SEEDS:
        print(
            f"  seed {seed:4d}: "
            f"{targets['target_wall_time_seconds_by_seed'][str(seed)]:.6f} s"
        )
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
