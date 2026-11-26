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


def _discover_project_root() -> Path:

    for candidate in (EXPERIMENT_DIR.parent, EXPERIMENT_DIR):
        if (candidate / "burgers_multiseed_results").is_dir():
            return candidate
    return EXPERIMENT_DIR.parent


BURGERS_ROOT = _discover_project_root()
OLD_RESULT_ROOT = BURGERS_ROOT / "burgers_multiseed_results"
OLD_POINTS_DIR = OLD_RESULT_ROOT / "fixed_training_points"
OLD_NMAF_DIR = OLD_RESULT_ROOT / "nmaf_aat"
REFERENCE_FILE = BURGERS_ROOT / "burgers_shock.mat"

NEW_RESULT_ROOT = EXPERIMENT_DIR / "results_seedwise_matched_to_nmaf"
TARGET_EXPORT = EXPERIMENT_DIR / "nmaf_seedwise_targets.json"
RAR_MANUAL_POINT_DIR = EXPERIMENT_DIR / "rar_preselected_points"

SEEDS = [99, 1234, 2026]
REPRESENTATIVE_SEED = 99

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


HIDDEN_LAYERS = 4
HIDDEN_NEURONS = 64
OUTPUT_DIM = 1

N_RES = 5000
N_BC = 200
N_IC = 300
RAR_ADD_N_RES = 1000
RAR_FINAL_N_RES = N_RES + RAR_ADD_N_RES

X_MIN = -1.0
X_MAX = 1.0
T_MIN = 0.0
T_MAX = 1.0

LBFGS_LR = 1.0
LBFGS_HISTORY_SIZE = 50
LBFGS_TOLERANCE_GRAD = 1.0e-9
LBFGS_TOLERANCE_CHANGE = np.finfo(float).eps
LBFGS_LINE_SEARCH_FN = "strong_wolfe"


TIME_CONTROLLED_LBFGS_MAX_ITER = 1
TIME_CONTROLLED_LBFGS_MAX_EVAL = 25
PRINT_EVERY_LBFGS_OUTER = 50

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0
FINAL_CHECKPOINT_EVAL = 5000


RAR_ADAM_ACQUISITION_STEPS = 2500
RAR_INTERVAL = 250
RAR_ROUNDS = 8
RAR_ADD_PER_ROUND = 125
RAR_LAST_REFINEMENT_STEP = RAR_INTERVAL * RAR_ROUNDS


RAR_METHOD_DIR_OVERRIDE = None


RAR_OVERHEAD_MODE = "through_last_refinement"


RAR_ACQUISITION_OVERHEAD_SECONDS_OVERRIDE = {
    99: None,
    1234: None,
    2026: None,
}


REQUIRE_RAR_ACQUISITION_OVERHEAD = True

N_PRED_BATCH = 65536
PACKAGE_VERSION = "BURGERS_SEEDWISE_NMAF_MATCHED_V1_20260904"


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
        "rar": "RAR-PINN",
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
        self.output_layer = nn.Linear(hidden_neurons, 1)

        self.activation = AdaptiveActivation(hidden_neurons)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
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
    preferred = seed_dir / f"burgers_nmaf_aat_seed{seed}_eval{FINAL_CHECKPOINT_EVAL:05d}.pth"
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
            f"Cannot locate original NMAF-AAT final Burgers checkpoint for seed={seed} in\n"
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
        raise ValueError(f"Invalid NMAF-AAT target time for seed={seed}: {value}")
    return value, str(ckpt_path)


def target_seconds_for_seed(seed: int) -> float:
    value, _ = read_original_nmaf_time(int(seed))
    return float(value)


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Reference file not found: {REFERENCE_FILE}")
    data = scipy.io.loadmat(REFERENCE_FILE)
    for key in ("x", "t", "usol"):
        if key not in data:
            raise KeyError(f"{REFERENCE_FILE.name} must contain '{key}'.")
    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    exact = np.asarray(data["usol"], dtype=np.float64)
    target_shape = (x.size, t.size)
    if exact.shape != target_shape:
        if exact.T.shape == target_shape:
            exact = exact.T
        else:
            raise ValueError(
                f"Unexpected usol shape {exact.shape}; expected {target_shape} or transpose."
            )
    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, exact


@torch.no_grad()
def predict(model: nn.Module, X, T, output_shape):
    model.eval()
    xf = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    tf = np.asarray(T, dtype=np.float64).reshape(-1, 1)
    pieces = []
    for start in range(0, len(xf), N_PRED_BATCH):
        stop = min(start + N_PRED_BATCH, len(xf))
        xt = torch.as_tensor(xf[start:stop], dtype=DTYPE, device=DEVICE)
        tt = torch.as_tensor(tf[start:stop], dtype=DTYPE, device=DEVICE)
        pieces.append(model(xt, tt).detach().cpu().numpy())
    return np.vstack(pieces).reshape(output_shape)


def relative_l2(exact, pred) -> float:
    a = np.asarray(exact).reshape(-1)
    b = np.asarray(pred).reshape(-1)
    denom = np.linalg.norm(a, 2)
    if denom == 0:
        raise ZeroDivisionError("Reference Burgers field has zero L2 norm.")
    return float(np.linalg.norm(a - b, 2) / denom)


def load_original_nmaf_seed_reference(seed: int) -> dict:

    seed = int(seed)
    target_time, time_source = read_original_nmaf_time(seed)
    summary_path = nmaf_seed_dir(seed) / "training_summary.json"
    checkpoint_path = nmaf_final_checkpoint_path(seed)

    stored_loss = np.nan
    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(obj.get("final_metrics"), dict):
            value = obj["final_metrics"].get("total_loss")
            if value is not None:
                stored_loss = float(value)

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    latest = ck.get("latest_metrics", {})
    if not np.isfinite(stored_loss) and "total_loss" in latest:
        stored_loss = float(latest["total_loss"])

    model = build_model("nmaf_aat")
    model.load_state_dict(state_dict_to_double(ck["model_state_dict"]), strict=True)


    base_np = load_training_points_np(seed)
    common_points = make_points(base_np)
    with torch.enable_grad():
        common_losses = _float_losses(compute_loss_components(model, common_points))

    x, t, X, T, exact = load_reference()
    pred = predict(model, X, T, exact.shape)
    abs_error = np.abs(exact - pred)
    l2 = relative_l2(exact, pred)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "reference_method": "nmaf_aat",
        "reference_method_label": "NMAF-AAT",
        "target_wall_time_seconds": float(target_time),
        "nmaf_total_loss": float(common_losses["total_loss"]),
        "nmaf_stored_final_loss": float(stored_loss) if np.isfinite(stored_loss) else None,
        "nmaf_relative_l2_error": float(l2),
        "nmaf_maximum_absolute_error": float(abs_error.max()),
        "time_source": time_source,
        "metric_source": str(checkpoint_path),
        "original_nmaf_seed_dir": str(nmaf_seed_dir(seed)),
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
    return OLD_POINTS_DIR / f"burgers_training_points_seed{int(seed)}.npz"


def load_training_points_np(seed: int) -> dict:
    path = training_points_path(seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ORIGINAL fixed Burgers training points: {path}\n"
            "Do not regenerate points for this efficiency supplement."
        )
    with np.load(path, allow_pickle=False) as data:
        keys = ["x_res", "t_res", "x_bc", "t_bc", "x_ic", "t_ic"]
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
        "x_bc": torch.tensor(base_np["x_bc"], dtype=DTYPE, device=DEVICE),
        "t_bc": torch.tensor(base_np["t_bc"], dtype=DTYPE, device=DEVICE),
        "x_ic": torch.tensor(base_np["x_ic"], dtype=DTYPE, device=DEVICE),
        "t_ic": torch.tensor(base_np["t_ic"], dtype=DTYPE, device=DEVICE),
    }


def compute_loss_components(model: nn.Module, points: dict):
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

    residual = u_t + u * u_x - (0.01 / np.pi) * u_xx
    loss_residual = torch.mean(residual ** 2)

    loss_bc = torch.mean(model(points["x_bc"], points["t_bc"]) ** 2)
    u_ic_pred = model(points["x_ic"], points["t_ic"])
    u_ic_exact = -torch.sin(np.pi * points["x_ic"])
    loss_ic = torch.mean((u_ic_pred - u_ic_exact) ** 2)

    data_loss = loss_bc + loss_ic
    total_loss = loss_residual + data_loss

    return {
        "total_loss": total_loss,
        "residual_loss": loss_residual,
        "data_loss": data_loss,
        "bc_loss": loss_bc,
        "ic_loss": loss_ic,
    }


def _float_losses(losses: dict) -> dict:
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def locate_formal_rar_dir() -> Path:
    if RAR_METHOD_DIR_OVERRIDE is not None:
        p = OLD_RESULT_ROOT / str(RAR_METHOD_DIR_OVERRIDE)
        if not p.is_dir():
            raise FileNotFoundError(f"RAR_METHOD_DIR_OVERRIDE not found: {p}")
        return p

    preferred_names = [
        "rar_sampling_only_base5000_plus1000",
        "rar_sampling_only_base5000_plus1000_fresh_lbfgs5000",
    ]
    for name in preferred_names:
        p = OLD_RESULT_ROOT / name
        if p.is_dir() and all((p / f"seed_{s}").is_dir() for s in SEEDS):
            return p

    candidates = []
    if OLD_RESULT_ROOT.exists():
        for p in OLD_RESULT_ROOT.iterdir():
            if not p.is_dir() or "rar" not in p.name.lower():
                continue
            if all((p / f"seed_{s}" / "sampling_stage_summary.json").exists() for s in SEEDS):
                candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            "Could not locate the ORIGINAL formal Burgers RAR directory.\n"
            "Expected a child of burgers_multiseed_results containing seed_99, "
            "seed_1234 and seed_2026 with sampling_stage_summary.json."
        )
    raise RuntimeError(
        "Multiple formal Burgers RAR directories found. Set RAR_METHOD_DIR_OVERRIDE "
        "in burgers_seedwise_common.py to one of:\n" +
        "\n".join(f"  {p.name}" for p in candidates)
    )


def _contains_all_base_points(x_full, t_full, base_np: dict) -> bool:
    full = np.column_stack([x_full.reshape(-1), t_full.reshape(-1)])
    base = np.column_stack([base_np["x_res"].reshape(-1), base_np["t_res"].reshape(-1)])
    if len(full) < len(base):
        return False
    if np.allclose(full[:len(base)], base, rtol=0.0, atol=1.0e-12):
        return True
    full_set = {tuple(row) for row in np.round(full, 7)}
    return all(tuple(row) in full_set for row in np.round(base, 7))


def _try_extract_rar_arrays(npz_path: Path, base_np: dict):
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            for xk, tk in (("x_res", "t_res"), ("x_train", "t_train"), ("x_all", "t_all")):
                if xk in d.files and tk in d.files:
                    x = np.asarray(d[xk], dtype=np.float64).reshape(-1, 1)
                    t = np.asarray(d[tk], dtype=np.float64).reshape(-1, 1)
                    if len(x) == RAR_FINAL_N_RES and len(t) == RAR_FINAL_N_RES:
                        if _contains_all_base_points(x, t, base_np):
                            return x.copy(), t.copy(), "full_6000"
            for xk, tk in (("x_added", "t_added"), ("x_new", "t_new"),
                           ("x_selected", "t_selected"), ("x_adaptive", "t_adaptive")):
                if xk in d.files and tk in d.files:
                    xa = np.asarray(d[xk], dtype=np.float64).reshape(-1, 1)
                    ta = np.asarray(d[tk], dtype=np.float64).reshape(-1, 1)
                    if len(xa) == RAR_ADD_N_RES and len(ta) == RAR_ADD_N_RES:
                        return (
                            np.vstack([base_np["x_res"], xa]),
                            np.vstack([base_np["t_res"], ta]),
                            "base5000_plus_added1000",
                        )
    except Exception:
        return None
    return None


def find_formal_rar_points(seed: int, base_np: dict):
    seed = int(seed)
    manual = RAR_MANUAL_POINT_DIR / f"burgers_rar_selected_points_seed{seed}.npz"
    search_files = []
    if manual.exists():
        search_files.append(manual)

    rar_dir = locate_formal_rar_dir()
    seed_dir = rar_dir / f"seed_{seed}"
    preferred = seed_dir / f"burgers_rar_base5000_plus1000_points_seed{seed}.npz"
    if preferred.exists():
        search_files.insert(0, preferred)
    if seed_dir.exists():
        search_files.extend(seed_dir.rglob("*.npz"))

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
    preferred_matches = [m for m in matches if m[0].name == preferred.name]
    if len(preferred_matches) == 1:
        p, x, t, mode = preferred_matches[0]
        return x, t, str(p), mode
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple 6000-point RAR files found for seed={seed}.\n"
            "Place the intended formal set explicitly at:\n"
            f"  {manual}\nCandidates:\n" +
            "\n".join(f"  {m[0]}" for m in matches)
        )

    raise FileNotFoundError(
        f"Could not find the ORIGINAL formal Burgers RAR 6000-point set for seed={seed}.\n"
        f"RAR directory detected: {rar_dir}\n"
        "This supplement must reuse the formal selected set rather than rerun RAR.\n"
        "If necessary, copy the intended set to:\n"
        f"  {manual}\n"
        "with arrays x_res/t_res of length 6000 (or x_added/t_added of length 1000)."
    )


def _rar_overhead_through_last_refinement(seed: int, rar_dir: Path) -> tuple[float, str]:
    seed_dir = rar_dir / f"seed_{seed}"
    adam_csv = seed_dir / "sampling_adam_history.csv"
    rar_csv = seed_dir / "rar_refinement_history.csv"
    if not adam_csv.exists() or not rar_csv.exists():
        raise FileNotFoundError(
            f"Need both {adam_csv.name} and {rar_csv.name} to recover acquisition time "
            "through the final refinement."
        )

    adam = pd.read_csv(adam_csv)
    rar = pd.read_csv(rar_csv)
    if "adam_step" not in adam.columns or "sampling_stage_wall_seconds" not in adam.columns:
        raise KeyError(f"Unexpected columns in {adam_csv}")
    if "rar_round" not in rar.columns or "selection_time_seconds" not in rar.columns:
        raise KeyError(f"Unexpected columns in {rar_csv}")

    row = adam.loc[pd.to_numeric(adam["adam_step"], errors="coerce") == RAR_LAST_REFINEMENT_STEP]
    final_round = rar.loc[pd.to_numeric(rar["rar_round"], errors="coerce") == RAR_ROUNDS]
    if len(row) != 1 or len(final_round) != 1:
        raise RuntimeError(
            f"Cannot uniquely recover RAR timing at Adam step {RAR_LAST_REFINEMENT_STEP}, "
            f"round {RAR_ROUNDS}, seed={seed}."
        )


    pre_final_selection = float(row.iloc[0]["sampling_stage_wall_seconds"])
    final_selection = float(final_round.iloc[0]["selection_time_seconds"])
    value = pre_final_selection + final_selection
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid recovered RAR acquisition time for seed={seed}: {value}")
    return value, f"{adam_csv}@step{RAR_LAST_REFINEMENT_STEP} + {rar_csv}@round{RAR_ROUNDS}"


def _rar_overhead_full_sampling_stage(seed: int, rar_dir: Path) -> tuple[float, str]:
    summary = rar_dir / f"seed_{seed}" / "sampling_stage_summary.json"
    if not summary.exists():
        raise FileNotFoundError(summary)
    obj = json.loads(summary.read_text(encoding="utf-8"))
    if "sampling_stage_wall_seconds" not in obj:
        raise KeyError(f"No sampling_stage_wall_seconds in {summary}")
    value = float(obj["sampling_stage_wall_seconds"])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid full RAR sampling-stage time for seed={seed}: {value}")
    return value, f"{summary}:sampling_stage_wall_seconds"


def rar_acquisition_overhead(seed: int) -> tuple[float, str, bool]:

    seed = int(seed)
    override = RAR_ACQUISITION_OVERHEAD_SECONDS_OVERRIDE.get(seed)
    if override is not None:
        value = float(override)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid RAR acquisition overhead override for seed={seed}: {override}")
        return value, "RAR_ACQUISITION_OVERHEAD_SECONDS_OVERRIDE", True

    try:
        rar_dir = locate_formal_rar_dir()
        if RAR_OVERHEAD_MODE == "through_last_refinement":
            value, source = _rar_overhead_through_last_refinement(seed, rar_dir)
        elif RAR_OVERHEAD_MODE == "full_sampling_stage":
            value, source = _rar_overhead_full_sampling_stage(seed, rar_dir)
        else:
            raise ValueError(
                "RAR_OVERHEAD_MODE must be 'through_last_refinement' or 'full_sampling_stage'."
            )
        return float(value), source, True
    except Exception as exc:
        if REQUIRE_RAR_ACQUISITION_OVERHEAD:
            raise RuntimeError(
                f"Cannot recover strict RAR acquisition overhead for seed={seed}. "
                "Either restore the original RAR timing CSV/JSON files or set a measured value in "
                "RAR_ACQUISITION_OVERHEAD_SECONDS_OVERRIDE."
            ) from exc
        return 0.0, f"not_available; optimistic Stage-B-only timing; cause={exc}", False


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
        rar_overhead, rar_overhead_source, rar_overhead_known = rar_acquisition_overhead(seed)
        stage_training_budget = max(0.0, target_seconds - rar_overhead)
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

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    lbfgs_outer_steps = 0
    lbfgs_closure_evals = 0

    sync()
    t0 = time.perf_counter()

    print("=" * 124)
    print(
        f"BURGERS SEED-WISE NMAF MATCHED TIME | {method_label(method)} | seed={seed} | "
        f"NMAF TARGET={target_seconds:.6f}s | TRAINING BUDGET={stage_training_budget:.6f}s"
    )
    if method == "rar":
        print(
            f"RAR formal points: {rar_points_source} | acquisition overhead applied="
            f"{rar_overhead:.6f}s | mode={RAR_OVERHEAD_MODE} | known={rar_overhead_known}"
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

    peak_gpu_memory_gb = None
    if DEVICE.type == "cuda":
        peak_gpu_memory_gb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 3))


    with torch.enable_grad():
        opt_losses = _float_losses(compute_loss_components(model, train_points))
        if method == "rar":
            monitor_losses = _float_losses(compute_loss_components(model, monitor_points))
        else:
            monitor_losses = opt_losses.copy()

    x, t, X, T, exact = load_reference()
    pred = predict(model, X, T, exact.shape)
    abs_error = np.abs(exact - pred)
    l2 = relative_l2(exact, pred)

    if method == "rar":
        timing_scope = (
            f"end-to-end equivalent: recorded RAR acquisition overhead ({RAR_OVERHEAD_MODE}) + Stage-B L-BFGS"
            if rar_overhead_known else
            "optimistic Stage-B-only: RAR acquisition overhead unavailable and treated as 0"
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
        "monitor_ic_loss": float(monitor_losses["ic_loss"]),
        "optimization_total_loss": float(opt_losses["total_loss"]),
        "optimization_residual_loss": float(opt_losses["residual_loss"]),
        "relative_l2_error": float(l2),
        "maximum_absolute_error": float(abs_error.max()),
        "peak_gpu_memory_allocated_gb_during_training": peak_gpu_memory_gb,
        "metric_evaluation_outside_training_timer": True,
        "rar_points_source": rar_points_source,
        "rar_points_mode": rar_points_mode,
        "rar_acquisition_overhead_seconds_applied": (
            float(rar_overhead) if method == "rar" else None
        ),
        "rar_acquisition_overhead_source": rar_overhead_source,
        "rar_acquisition_overhead_known": rar_overhead_known,
        "rar_overhead_mode": RAR_OVERHEAD_MODE if method == "rar" else None,
        "rar_timing_scope": timing_scope if method == "rar" else None,
        "IMPORTANT": (
            "No averaged target time was used. The target is the original NMAF-AAT "
            "final wall time for this exact seed."
        ),
    }

    save_json(out_dir / "seedwise_matched_summary.json", summary)
    torch.save(
        {"summary": summary, "model_state_dict": cpu_state_dict(model)},
        out_dir / f"burgers_{method}_seed{seed}_seedwise_nmaf_time_final.pth",
    )
    np.savez_compressed(
        out_dir / f"burgers_{method}_seed{seed}_seedwise_nmaf_time_prediction.npz",
        x=x, t=t, X=X, T=T,
        exact=exact, pred=pred, absolute_error=abs_error,
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
