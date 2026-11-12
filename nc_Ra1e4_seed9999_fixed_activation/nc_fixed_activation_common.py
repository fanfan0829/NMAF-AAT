from __future__ import annotations
import csv, gc, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import qmc

RA_EXPONENT = 4
RA = 10.0 ** RA_EXPONENT
PR = 0.71
SEEDS = [9999]
REPRESENTATIVE_SEED = 9999

PACKAGE_DIR = Path(__file__).resolve().parent
RESULT_ROOT = PACKAGE_DIR / "results_mlp7x64"
FIXED_ROOT = RESULT_ROOT / "fixed_budget"
MATCHED_ROOT = RESULT_ROOT / "matched_time"
POINTS_DIR = RESULT_ROOT / "fixed_training_points"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_TRAIN = torch.float32
DTYPE_EVAL = torch.float64
FAST_EXECUTION = True

N_BOUNDARY = 4000
N_INTERIOR = 10000
N_PER_EDGE = 1000
HIDDEN_NEURONS = 64
HIDDEN_LAYERS = 7
OUTPUT_DIM = 2
ACTIVATION_SITES = HIDDEN_LAYERS
ARCHITECTURE_ID = "MLP_7x64_v1"

ADAM_STEPS = 5000
ADAM_LR = 1.0e-3
LBFGS_STEPS = 5000
LBFGS_LR = 1.0
LBFGS_HISTORY_SIZE = 200

LBFGS_MAX_ITER = LBFGS_STEPS
LBFGS_MAX_EVAL = LBFGS_STEPS

TIME_CONTROLLED_LBFGS_MAX_ITER = 1
TIME_CONTROLLED_LBFGS_MAX_EVAL = 25
LBFGS_TOLERANCE_GRAD = 1.0e-10
LBFGS_TOLERANCE_CHANGE = np.finfo(float).eps
LBFGS_LINE_SEARCH_FN = "strong_wolfe"


CURVE_EVERY_ADAM = 1
CURVE_EVERY_LBFGS = 1
PRINT_EVERY_ADAM = 1000
PRINT_EVERY_LBFGS = 1000


RAR_ROUNDS = 10
RAR_SELECTION_STEPS = tuple(int(round(k * ADAM_STEPS / 12.0)) for k in range(1, 11))
if len(set(RAR_SELECTION_STEPS)) != RAR_ROUNDS:
    raise RuntimeError("RAR selection steps are not unique for this Adam budget")
RAR_CANDIDATES_PER_ROUND = 100000
RAR_CANDIDATE_BATCH = 5000
RAR_ADD_PER_ROUND = 200
RAR_TOTAL_ADD = RAR_ROUNDS * RAR_ADD_PER_ROUND
RAR_FINAL_N = N_INTERIOR + RAR_TOTAL_ADD
RAR_SEED_OFFSET = 731021
RAR_PROTOCOL_ID = "MULTI_ROUND_10x200_RELATIVE_TO_ADAM_OLD_5k_to_50k_of_60k"

PREDICTION_GRID_SIZE = 200
PREDICTION_BATCH_SIZE = 10000
IMAGE_DPI = 800
INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0

BENCHMARK = {
    "psi_mid_abs": 5.071,
    "psi_max_abs": None,
    "u_max_vertical_midplane": 16.178,
    "w_max_horizontal_midplane": 19.617,
    "Nu_max_hot_wall": 3.528,
    "Nu_min_hot_wall": 0.586,
    "Nu_mean_hot_wall": 2.243,
}
METHODS = [
    "tanh", "relu", "sigmoid", "swish", "sine", "gaussian",
    "tanswish", "sigswish", "sigtanh", "gausstanh",
    "gausssigmoid", "gaussswish", "gelu", "mish", "xtanh", "nmaf",
]
METHOD_LABELS = {
    "tanh": "tanh",
    "relu": "ReLU",
    "sigmoid": "sigmoid",
    "swish": "Swish",
    "sine": "sine",
    "gaussian": "Gaussian",
    "tanswish": "TanSwish",
    "sigswish": "SigSwish",
    "sigtanh": "SigTanh",
    "gausstanh": "GaussTanh",
    "gausssigmoid": "GaussSigmoid",
    "gaussswish": "GaussSwish",
    "gelu": "GELU",
    "mish": "Mish",
    "xtanh": "xTanh",
    "nmaf": "NMAF",
}
METHOD_FORMULAS = {
    "tanh": "tanh(x)",
    "relu": "ReLU(x)",
    "sigmoid": "sigmoid(x)",
    "swish": "x*sigmoid(x)",
    "sine": "sin(x)",
    "gaussian": "-x*exp(-x^2/2)",
    "tanswish": "tanh(x) + x*sigmoid(x)",
    "sigswish": "sigmoid(x) + x*sigmoid(x)",
    "sigtanh": "tanh(x) + sigmoid(x)",
    "gausstanh": "-x*exp(-x^2/2) + tanh(x)",
    "gausssigmoid": "-x*exp(-x^2/2) + sigmoid(x)",
    "gaussswish": "-x*exp(-x^2/2) + x*sigmoid(x)",
    "gelu": "GELU(x)=x*Phi(x), exact erf form (PyTorch approximate='none')",
    "mish": "x*tanh(softplus(x))",
    "xtanh": "x*tanh(x)",
    "nmaf": "-x*(exp(-x^2/2)-tanh(x)) = x*tanh(x)-x*exp(-x^2/2)",
}
QUANTITY_LABELS = {
    "psi_mid_abs": r"$|\psi_{mid}|$",
    "psi_max_abs": r"$|\psi|_{max}$",
    "u_max_vertical_midplane": r"$u_{max}$",
    "w_max_horizontal_midplane": r"$w_{max}$",
    "Nu_max_hot_wall": r"$Nu_{max}$",
    "Nu_min_hot_wall": r"$Nu_{min}$",
    "Nu_mean_hot_wall": r"$\overline{Nu}$",
}


def set_seed(seed):
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


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def release_method_memory(tag=""):


    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        before_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        before_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    else:
        before_alloc = before_reserved = 0.0

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        after_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        label = f" [{tag}]" if tag else ""
        print(
            f"CUDA cleanup{label} | "
            f"allocated {before_alloc:.1f} -> {after_alloc:.1f} MiB | "
            f"reserved {before_reserved:.1f} -> {after_reserved:.1f} MiB"
        )


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def cast_state_dict(state, dtype):
    return {k: (v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in state.items()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FixedActivation(nn.Module):

    def __init__(self, kind: str):
        super().__init__()
        if kind not in METHODS:
            raise ValueError(kind)
        self.kind = kind

    def forward(self, x):
        k = self.kind
        if k == "tanh":
            return torch.tanh(x)
        if k == "relu":
            return torch.relu(x)
        if k == "sigmoid":
            return torch.sigmoid(x)
        if k == "swish":
            return x * torch.sigmoid(x)
        if k == "sine":
            return torch.sin(x)
        if k == "gaussian":
            return -x * torch.exp(-0.5 * x ** 2)
        if k == "tanswish":
            return torch.tanh(x) + x * torch.sigmoid(x)
        if k == "sigswish":
            sig = torch.sigmoid(x)
            return sig + x * sig
        if k == "sigtanh":
            return torch.tanh(x) + torch.sigmoid(x)
        if k == "gausstanh":
            return (-x * torch.exp(-0.5 * x ** 2)) + torch.tanh(x)
        if k == "gausssigmoid":
            return (-x * torch.exp(-0.5 * x ** 2)) + torch.sigmoid(x)
        if k == "gaussswish":
            return (-x * torch.exp(-0.5 * x ** 2)) + x * torch.sigmoid(x)
        if k == "gelu":
            return torch.nn.functional.gelu(x, approximate="none")
        if k == "mish":
            return x * torch.tanh(torch.nn.functional.softplus(x))
        if k == "xtanh":
            return x * torch.tanh(x)
        if k == "nmaf":
            return -x * (torch.exp(-0.5 * x ** 2) - torch.tanh(x))
        raise ValueError(k)


def _act_factory(kind, dtype):
    return FixedActivation(kind)


class MLPPINN(nn.Module):

    def __init__(self, kind="tanh", dtype=DTYPE_TRAIN):
        super().__init__()
        layers = [nn.Linear(2, HIDDEN_NEURONS, dtype=dtype), _act_factory(kind, dtype)]
        for _ in range(HIDDEN_LAYERS - 1):
            layers.extend([
                nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS, dtype=dtype),
                _act_factory(kind, dtype),
            ])
        layers.append(nn.Linear(HIDDEN_NEURONS, OUTPUT_DIM, dtype=dtype))
        self.layers = nn.Sequential(*layers)
        self.kind = kind

    def forward(self, xz):
        return self.layers(xz)


def architecture_metadata():
    return {
        "architecture": "MLP: 2 -> 64 -> 64 -> 64 -> 64 -> 64 -> 64 -> 64 -> 2",
        "architecture_id": ARCHITECTURE_ID,
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_neurons": HIDDEN_NEURONS,
        "activation_sites": ACTIVATION_SITES,
    }


def build_model(method, dtype=DTYPE_TRAIN):
    if method not in METHODS:
        raise ValueError(method)
    return MLPPINN(kind=method, dtype=dtype).to(DEVICE)

def linear_state_hash(model):
    h = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h.update(name.encode())
            h.update(np.ascontiguousarray(module.weight.detach().cpu().numpy()).tobytes())
            h.update(np.ascontiguousarray(module.bias.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def verify_same_seed_linear_initialization(seed):
    hashes = {}
    for method in METHODS:
        set_seed(seed)
        hashes[method] = linear_state_hash(build_model(method, DTYPE_TRAIN))
    ok = len(set(hashes.values())) == 1
    return ok, hashes


def points_path(seed):
    return POINTS_DIR / f"nc_mlp_Ra1e{RA_EXPONENT}_seed{int(seed)}.npz"


def generate_points(seed):
    seed = int(seed)
    x = np.linspace(0, 1, N_PER_EDGE)
    z = np.linspace(0, 1, N_PER_EDGE)
    bottom = np.stack([x, np.zeros_like(x)], axis=1)
    right = np.stack([np.ones_like(z), z], axis=1)
    top = np.stack([x, np.ones_like(x)], axis=1)
    left = np.stack([np.zeros_like(z), z], axis=1)
    boundary = np.concatenate([bottom, top, right, left], axis=0)
    eps = 1e-8
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    interior = qmc.scale(sampler.random(N_INTERIOR), [eps, eps], [1-eps, 1-eps])
    return {
        "boundary": boundary.astype(np.float64),
        "interior": interior.astype(np.float64),
        "top": top.astype(np.float64),
        "bottom": bottom.astype(np.float64),
        "left": left.astype(np.float64),
        "right": right.astype(np.float64),
    }


def generate_and_save_all_points(overwrite=False):
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        p = points_path(seed)
        if p.exists() and not overwrite:
            print("KEEP:", p)
            continue
        np.savez_compressed(p, **generate_points(seed))
        print("SAVE:", p)


def load_points_np(seed):
    p = points_path(seed)
    if not p.exists():
        raise FileNotFoundError(f"{p}\nRun 00_generate_fixed_points.py first.")
    with np.load(p) as d:
        return {k: np.asarray(d[k], dtype=np.float64).copy() for k in d.files}


def make_tensors(base, interior=None, dtype=DTYPE_TRAIN):
    pts = base["interior"] if interior is None else np.asarray(interior, dtype=np.float64)
    top = torch.tensor(base["top"], dtype=dtype, device=DEVICE)
    bottom = torch.tensor(base["bottom"], dtype=dtype, device=DEVICE)
    return {
        "interior": torch.tensor(pts, dtype=dtype, device=DEVICE),
        "td": torch.vstack((top, bottom)),
    }


def raw_to_fields(model, x, z):
    res = model(torch.hstack((x, z)))
    bcv = 16 * x * (1 - x) * z * (1 - z)
    fai = -bcv * bcv * res[:, 0:1]
    T = 1 - x + torch.sin(torch.pi * x) * res[:, 1:2]
    return fai, T


def residuals(model, xz):
    x = xz[:, 0:1].detach().clone().requires_grad_(True)
    z = xz[:, 1:2].detach().clone().requires_grad_(True)
    fai, T = raw_to_fields(model, x, z)

    grads = torch.autograd.grad(fai, [x, z], torch.ones_like(fai), create_graph=True)
    w = -grads[0]
    u = grads[1]
    u_z = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    w_x = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
    vort = w_x - u_z
    vort_grads = torch.autograd.grad(vort, [x, z], torch.ones_like(vort), create_graph=True)
    vort_x, vort_z = vort_grads[0], vort_grads[1]
    vort_xx = torch.autograd.grad(vort_x, x, torch.ones_like(vort_x), create_graph=True)[0]
    vort_zz = torch.autograd.grad(vort_z, z, torch.ones_like(vort_z), create_graph=True)[0]

    T_grads = torch.autograd.grad(T, [x, z], torch.ones_like(T), create_graph=True)
    T_x, T_z = T_grads[0], T_grads[1]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_zz = torch.autograd.grad(T_z, z, torch.ones_like(T_z), create_graph=True)[0]

    f1 = u * vort_x + w * vort_z - (vort_xx + vort_zz) * PR - T_x * PR * RA
    f1 = f1 / RA
    f2 = u * T_x + w * T_z - (T_xx + T_zz)
    return f1, f2


def bc_temperature_gradient(model, edge_points):
    x = edge_points[:, 0:1].detach().clone().requires_grad_(True)
    z = edge_points[:, 1:2].detach().clone().requires_grad_(True)
    _, T = raw_to_fields(model, x, z)
    return torch.autograd.grad(T, z, torch.ones_like(T), create_graph=True)[0]


def loss_components(model, tensors):
    f1, f2 = residuals(model, tensors["interior"])
    L1 = torch.mean(f1**2)
    L2 = torch.mean(f2**2)
    Lpde = L1 + L2
    Tz = bc_temperature_gradient(model, tensors["td"])
    Lbc = torch.mean(Tz**2)
    total = Lpde + Lbc
    return {
        "total_loss": total,
        "pde_loss": Lpde,
        "bc_loss": Lbc,
        "f1_loss": L1,
        "f2_loss": L2,
    }

def detached_losses(d):
    return {k: float(v.detach().cpu()) for k, v in d.items()}


def rar_candidate_seed(seed, round_index):
    return int(seed) + RAR_SEED_OFFSET + 1000 * RA_EXPONENT + 1000003 * int(round_index)


def rar_candidates(seed, round_index):
    eps = 1e-8
    sampler = qmc.LatinHypercube(d=2, seed=rar_candidate_seed(seed, round_index))
    return qmc.scale(
        sampler.random(RAR_CANDIDATES_PER_ROUND), [eps, eps], [1-eps, 1-eps]
    ).astype(np.float64)


def select_rar_points(model, seed, round_index, adam_step):


    cand = rar_candidates(seed, round_index)
    score_parts = []
    t0 = time.perf_counter()
    for start in range(0, len(cand), RAR_CANDIDATE_BATCH):
        stop = min(start + RAR_CANDIDATE_BATCH, len(cand))
        batch = torch.tensor(cand[start:stop], dtype=DTYPE_TRAIN, device=DEVICE)
        f1, f2 = residuals(model, batch)
        score = torch.sqrt(f1.detach()**2 + f2.detach()**2)
        score_parts.append(score.cpu().numpy().reshape(-1))
        del batch, f1, f2, score
    elapsed = time.perf_counter() - t0
    scores = np.concatenate(score_parts)
    idx = np.argpartition(scores, -RAR_ADD_PER_ROUND)[-RAR_ADD_PER_ROUND:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return cand[idx].copy(), {
        "protocol_id": RAR_PROTOCOL_ID,
        "round_index": int(round_index),
        "adam_step": int(adam_step),
        "candidate_seed": int(rar_candidate_seed(seed, round_index)),
        "candidate_count": int(RAR_CANDIDATES_PER_ROUND),
        "added_count": int(RAR_ADD_PER_ROUND),
        "indicator": "sqrt(f1^2+f2^2)",
        "selection_wall_seconds": float(elapsed),
    }


def eval_model_from_train_model(method, model):
    state = cast_state_dict(cpu_state_dict(model), DTYPE_EVAL)
    out = build_model(method, DTYPE_EVAL)
    out.load_state_dict(state, strict=True)
    out.eval()
    return out


def generate_prediction_grid(grid_size=PREDICTION_GRID_SIZE):
    x = np.linspace(0, 1, grid_size)
    z = np.linspace(0, 1, grid_size)
    X, Z = np.meshgrid(x, z)
    points = np.stack([X.flatten(), Z.flatten()], axis=1)
    return X, Z, points


def batch_predict(model, points, batch_size=PREDICTION_BATCH_SIZE):
    A, B, C, D = [], [], [], []
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        p = torch.tensor(points[start:stop], dtype=DTYPE_EVAL, device=DEVICE)
        x = p[:, 0:1].clone().detach().requires_grad_(True)
        z = p[:, 1:2].clone().detach().requires_grad_(True)
        psi, T = raw_to_fields(model, x, z)
        u = torch.autograd.grad(psi.sum(), z, create_graph=False, retain_graph=True)[0]
        w = -torch.autograd.grad(psi.sum(), x, create_graph=False, retain_graph=False)[0]
        A.append(psi.detach().cpu().numpy())
        B.append(u.detach().cpu().numpy())
        C.append(w.detach().cpu().numpy())
        D.append(T.detach().cpu().numpy())
    return tuple(np.vstack(v).reshape(-1) for v in (A, B, C, D))


def psi_point(model, x0, z0):
    x = torch.tensor([[x0]], dtype=DTYPE_EVAL, device=DEVICE)
    z = torch.tensor([[z0]], dtype=DTYPE_EVAL, device=DEVICE)
    psi, _ = raw_to_fields(model, x, z)
    return float(psi.detach().cpu())


def velocity_profiles(model, n=PREDICTION_GRID_SIZE):
    s = np.linspace(0, 1, n)
    vertical = np.column_stack([np.full(n, 0.5), s])
    horizontal = np.column_stack([s, np.full(n, 0.5)])
    _, u_v, _, _ = batch_predict(model, vertical, batch_size=n)
    _, _, w_h, _ = batch_predict(model, horizontal, batch_size=n)
    return s, u_v, s, w_h


def hot_wall_nusselt(model, n=PREDICTION_GRID_SIZE):
    z_np = np.linspace(0, 1, n)
    x = torch.zeros((n, 1), dtype=DTYPE_EVAL, device=DEVICE, requires_grad=True)
    z = torch.tensor(z_np[:, None], dtype=DTYPE_EVAL, device=DEVICE, requires_grad=True)
    _, T = raw_to_fields(model, x, z)
    Tx = torch.autograd.grad(T.sum(), x, create_graph=False)[0]
    Nu = -Tx.detach().cpu().numpy().reshape(-1)
    if hasattr(np, "trapezoid"):
        Nu_mean = float(np.trapezoid(Nu, z_np))
    else:
        Nu_mean = float(np.trapz(Nu, z_np))
    return z_np, Nu, Nu_mean


def physical_quantities(model):
    _, _, points = generate_prediction_grid(PREDICTION_GRID_SIZE)
    psi, _, _, _ = batch_predict(model, points)
    _, u_v, _, w_h = velocity_profiles(model)
    _, Nu, Nu_mean = hot_wall_nusselt(model)
    return {
        "psi_mid_abs": abs(psi_point(model, 0.5, 0.5)),
        "psi_max_abs": float(np.max(np.abs(psi))),
        "u_max_vertical_midplane": float(np.max(u_v)),
        "w_max_horizontal_midplane": float(np.max(w_h)),
        "Nu_max_hot_wall": float(np.max(Nu)),
        "Nu_min_hot_wall": float(np.min(Nu)),
        "Nu_mean_hot_wall": float(Nu_mean),
    }


def relerr_percent(pred, bench):
    if bench is None:
        return np.nan
    return abs(float(pred) - float(bench)) / abs(float(bench)) * 100.0


def final_metrics(method, model, base, current_interior):
    eval_model = eval_model_from_train_model(method, model)
    monitor = make_tensors(base, base["interior"], DTYPE_EVAL)
    opt = make_tensors(base, current_interior, DTYPE_EVAL)
    with torch.enable_grad():
        common = detached_losses(loss_components(eval_model, monitor))
        optimization = detached_losses(loss_components(eval_model, opt)) if method == "rar" else dict(common)
    return eval_model, common, optimization, physical_quantities(eval_model)


def fixed_out_dir(method, seed):
    return FIXED_ROOT / method / f"seed_{int(seed)}"


def formal_summary_path(method, seed):
    return fixed_out_dir(method, seed) / "training_summary.json"


def _append_monitor(history, model, monitor_pts, current_n, stage, stage_step, combined_step, formal_elapsed):

    with torch.enable_grad():
        d = loss_components(model, monitor_pts)
        total = float(d["total_loss"].detach().cpu())
        del d
    history.append({
        "stage": stage,
        "stage_step": int(stage_step),
        "combined_step": int(combined_step),
        "formal_elapsed_seconds": float(formal_elapsed),
        "total_loss": total,
        "training_n_interior": int(current_n),
    })

def make_fixed_budget_lbfgs_optimizer(model):


    return torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        max_eval=LBFGS_MAX_EVAL,
        history_size=LBFGS_HISTORY_SIZE,
        tolerance_grad=LBFGS_TOLERANCE_GRAD,
        tolerance_change=LBFGS_TOLERANCE_CHANGE,
        line_search_fn=LBFGS_LINE_SEARCH_FN,
    )


def make_time_controlled_lbfgs_optimizer(model):


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


def _append_known_loss(history, total_loss, current_n, stage, stage_step, combined_step, formal_elapsed):

    history.append({
        "stage": stage,
        "stage_step": int(stage_step),
        "combined_step": int(combined_step),
        "formal_elapsed_seconds": float(formal_elapsed),
        "total_loss": float(total_loss),
        "training_n_interior": int(current_n),
    })


def train_fixed_seed(method, seed):
    if method not in METHOD_LABELS:
        raise ValueError(method)
    seed = int(seed)
    base = load_points_np(seed)
    set_seed(seed)
    model = build_model(method, DTYPE_TRAIN)
    init_hash = linear_state_hash(model)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)

    original = base["interior"].copy()
    current = original.copy()
    train_pts = make_tensors(base, current, DTYPE_TRAIN)
    monitor_pts = make_tensors(base, original, DTYPE_TRAIN)
    out = fixed_out_dir(method, seed)
    out.mkdir(parents=True, exist_ok=True)

    history = []
    rar_round_history = []
    selected_all = []
    formal_elapsed = 0.0
    completed_rar_rounds = 0

    model.train()
    print("=" * 112)
    print(
        f"Ra=1e{RA_EXPONENT} | {METHOD_LABELS[method]} | seed={seed} | "
        f"Adam {ADAM_STEPS} + continuous L-BFGS max_iter=max_eval={LBFGS_STEPS}"
    )
    print(
        f"FAST execution | L-BFGS history_size={LBFGS_HISTORY_SIZE} | "
        f"curve sample every {CURVE_EVERY_ADAM} Adam steps / "
        f"{CURVE_EVERY_LBFGS} L-BFGS function evaluations"
    )
    print("=" * 112)


    sync()
    segment_start = time.perf_counter()
    for step in range(1, ADAM_STEPS + 1):
        optimizer_adam.zero_grad(set_to_none=True)
        d = loss_components(model, train_pts)
        d["total_loss"].backward()
        optimizer_adam.step()
        del d

        if method == "rar" and step in RAR_SELECTION_STEPS:
            round_index = RAR_SELECTION_STEPS.index(step) + 1
            selected, meta = select_rar_points(model, seed, round_index, step)
            current = np.vstack([current, selected])
            expected_n = N_INTERIOR + round_index * RAR_ADD_PER_ROUND
            if len(current) != expected_n:
                raise RuntimeError("RAR point-count mismatch")
            if not np.array_equal(current[:N_INTERIOR], original):
                raise RuntimeError("Original 10000-point prefix changed")
            train_pts = make_tensors(base, current, DTYPE_TRAIN)
            completed_rar_rounds += 1
            meta = dict(meta)
            meta["training_n_interior_after_round"] = int(len(current))
            rar_round_history.append(meta)
            selected_all.append(selected.copy())

        need_curve = (step % CURVE_EVERY_ADAM == 0) or (step == ADAM_STEPS)
        if need_curve:
            sync()
            formal_elapsed += time.perf_counter() - segment_start
            _append_monitor(
                history, model, monitor_pts, len(current), "adam", step, step,
                formal_elapsed,
            )
            if step % PRINT_EVERY_ADAM == 0 or step == ADAM_STEPS:
                print(
                    f"Adam {step:6d}/{ADAM_STEPS} | "
                    f"common loss={history[-1]['total_loss']:.6e} | N_RES={len(current)}"
                )
            if step != ADAM_STEPS:
                segment_start = time.perf_counter()

    adam_formal_time = formal_elapsed


    optimizer_lbfgs = make_fixed_budget_lbfgs_optimizer(model)
    closure_counter = 0
    lbfgs_diagnostic_seconds = 0.0

    sync()
    lbfgs_wall_start = time.perf_counter()

    def closure():
        nonlocal closure_counter, lbfgs_diagnostic_seconds

        optimizer_lbfgs.zero_grad(set_to_none=True)
        d = loss_components(model, train_pts)
        loss = d["total_loss"]
        loss.backward()
        closure_counter += 1

        need_curve = (closure_counter % CURVE_EVERY_LBFGS == 0)
        if need_curve:


            sync()
            formal_lbfgs_so_far = (
                time.perf_counter() - lbfgs_wall_start - lbfgs_diagnostic_seconds
            )
            combined_elapsed = adam_formal_time + formal_lbfgs_so_far

            diag_start = time.perf_counter()
            if method == "rar":
                _append_monitor(
                    history, model, monitor_pts, len(current), "lbfgs",
                    closure_counter, ADAM_STEPS + closure_counter,
                    combined_elapsed,
                )
            else:

                total_value = float(loss.detach().cpu())
                _append_known_loss(
                    history, total_value, len(current), "lbfgs",
                    closure_counter, ADAM_STEPS + closure_counter,
                    combined_elapsed,
                )

            if closure_counter % PRINT_EVERY_LBFGS == 0:
                print(
                    f"L-BFGS eval {closure_counter:6d}/{LBFGS_MAX_EVAL} | "
                    f"common loss={history[-1]['total_loss']:.6e}"
                )

            sync()
            lbfgs_diagnostic_seconds += time.perf_counter() - diag_start

        return loss

    optimizer_lbfgs.step(closure)
    sync()
    lbfgs_wall_elapsed = time.perf_counter() - lbfgs_wall_start
    lbfgs_formal_time = max(0.0, lbfgs_wall_elapsed - lbfgs_diagnostic_seconds)
    formal_elapsed = adam_formal_time + lbfgs_formal_time

    if closure_counter <= 0:
        raise RuntimeError("No L-BFGS closure evaluation was performed")


    if not history or not (
        history[-1]["stage"] == "lbfgs"
        and int(history[-1]["stage_step"]) == int(closure_counter)
    ):
        _append_monitor(
            history, model, monitor_pts, len(current), "lbfgs",
            closure_counter, ADAM_STEPS + closure_counter, formal_elapsed,
        )

    print(
        f"L-BFGS completed | configured max_iter=max_eval={LBFGS_STEPS} | "
        f"actual closure evals={closure_counter} | "
        f"final common loss={history[-1]['total_loss']:.6e}"
    )


    eval_model, common, optimization, quantities = final_metrics(method, model, base, current)


    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    if method == "rar":
        selected_stack = np.vstack(selected_all) if selected_all else np.empty((0, 2), dtype=np.float64)
        np.savez_compressed(
            out / f"rar_selected_points_seed{seed}.npz",
            original_interior=original,
            selected_interior=selected_stack,
            interior=current,
        )
        save_json(out / "rar_round_history.json", rar_round_history)

    summary = {
        "protocol": "FAST fixed Adam + continuous L-BFGS function-evaluation budget",
        "Ra_exponent": int(RA_EXPONENT), "Ra": float(RA), "Pr": float(PR),
        "method": method, "method_label": METHOD_LABELS[method], "seed": seed,
        "adam_steps": int(ADAM_STEPS),
        "lbfgs_configured_max_iter": int(LBFGS_MAX_ITER),
        "lbfgs_configured_max_eval": int(LBFGS_MAX_EVAL),
        "lbfgs_closure_evaluations": int(closure_counter),
        "lbfgs_budget_definition": "continuous torch.optim.LBFGS; one optimizer.step(closure); max_iter=max_eval=configured budget",
        "lbfgs_optimizer_step_calls": 1,
        "adam_lr": float(ADAM_LR), "lbfgs_lr": float(LBFGS_LR),
        "lbfgs_history_size": int(LBFGS_HISTORY_SIZE),
        "formal_adam_time_seconds": float(adam_formal_time),
        "formal_lbfgs_time_seconds": float(lbfgs_formal_time),
        "formal_training_time_seconds": float(formal_elapsed),
        "lbfgs_diagnostic_seconds_excluded": float(lbfgs_diagnostic_seconds),
        "curve_sampling_every_adam_steps": int(CURVE_EVERY_ADAM),
        "curve_sampling_every_lbfgs_function_evaluations": int(CURVE_EVERY_LBFGS),
        "diagnostic_curve_time_included": False,
        "checkpoint_io_time_included": False,
        "per_step_cuda_synchronize": False,
        "initial_linear_state_sha256": init_hash,
        **architecture_metadata(),
        "activation_formula": METHOD_FORMULAS[method],
        "activation_parameters_trainable": False,
        "trainable_parameters": int(count_parameters(model)),
        "common_monitor_n_interior": int(N_INTERIOR),
        "final_training_n_interior": int(len(current)),
        "final_common_total_loss": float(common["total_loss"]),
        "final_common_pde_loss": float(common["pde_loss"]),
        "final_common_bc_loss": float(common["bc_loss"]),
        "final_optimization_total_loss": float(optimization["total_loss"]),
        "physical_quantities": quantities,
        "rar_protocol_id": RAR_PROTOCOL_ID if method == "rar" else None,
        "rar_selection_steps": list(RAR_SELECTION_STEPS) if method == "rar" else None,
        "rar_rounds_completed": int(completed_rar_rounds) if method == "rar" else None,
        "rar_candidates_per_round": int(RAR_CANDIDATES_PER_ROUND) if method == "rar" else None,
        "rar_add_per_round": int(RAR_ADD_PER_ROUND) if method == "rar" else None,
        "rar_selection_seconds_total": float(sum(m.get("selection_wall_seconds", 0.0) for m in rar_round_history)) if method == "rar" else None,
    }
    save_json(out / "training_summary.json", summary)
    torch.save(
        {"model_state_dict": cpu_state_dict(eval_model), "summary": summary},
        out / f"Ra1e{RA_EXPONENT}_{method}_seed{seed}_final.pth",
    )
    print(
        f"DONE | {METHOD_LABELS[method]} | seed={seed} | "
        f"formal training={formal_elapsed:.3f}s | common loss={common['total_loss']:.6e}"
    )
    return summary

def train_all_fixed(method):
    for seed in SEEDS:
        train_fixed_seed(method, seed)


def training_points_sha256(seed=9999):
    base = load_points_np(seed)
    h = hashlib.sha256()
    for key in sorted(base):
        h.update(key.encode())
        h.update(np.ascontiguousarray(base[key]).tobytes())
    return h.hexdigest()


def write_fairness_audit(seed=9999):
    ok, hashes = verify_same_seed_linear_initialization(seed)
    if not ok:
        raise RuntimeError(f"Initial Linear parameters are NOT identical: {hashes}")
    params = {}
    for method in METHODS:
        set_seed(seed)
        model = build_model(method, DTYPE_TRAIN)
        params[method] = count_parameters(model)
    if len(set(params.values())) != 1:
        raise RuntimeError(f"Trainable parameter counts differ: {params}")
    audit = {
        "seed": int(seed),
        "all_initial_linear_hashes_identical": bool(ok),
        **architecture_metadata(),
        "initial_linear_state_sha256": hashes,
        "trainable_parameter_counts": params,
        "all_parameter_counts_identical": True,
        "training_points_sha256": training_points_sha256(seed),
        "methods": METHODS,
        "activation_formulas": METHOD_FORMULAS,
        "note": "All compared activations are fixed and parameter-free; only the activation mapping changes.",
    }
    save_json(RESULT_ROOT / "fairness_audit_seed9999.json", audit)
    return audit


def matched_out_dir(method, seed):
    return MATCHED_ROOT / method / f"seed_{int(seed)}"


def _load_nmaf_target(seed=9999):
    p = formal_summary_path("nmaf", seed)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing formal fixed-budget NMAF summary: {p}\n"
            "Run fixed-budget NMAF first; its formal training time defines the matched-time target."
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    target = float(d["formal_training_time_seconds"])
    if not np.isfinite(target) or target <= 0:
        raise RuntimeError(f"Invalid NMAF target time: {target}")
    return target, d


def train_matched_time_seed(method, seed=9999):
    if method not in METHODS:
        raise ValueError(method)
    seed = int(seed)
    if seed != 9999:
        raise ValueError("This revision experiment is intentionally restricted to seed=9999")

    target_seconds, target_summary = _load_nmaf_target(seed)
    if method == "nmaf":
        return {
            "method": "nmaf",
            "seed": seed,
            "reused_fixed_budget_reference": True,
            "target_seconds": target_seconds,
            "selected_formal_training_time_seconds": target_seconds,
            "checkpoint": str(fixed_out_dir("nmaf", seed) / f"Ra1e{RA_EXPONENT}_nmaf_seed{seed}_final.pth"),
        }

    base = load_points_np(seed)
    set_seed(seed)
    model = build_model(method, DTYPE_TRAIN)
    init_hash = linear_state_hash(model)
    train_pts = make_tensors(base, base["interior"], DTYPE_TRAIN)
    out = matched_out_dir(method, seed)
    out.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    ref_hash = linear_state_hash(build_model("nmaf", DTYPE_TRAIN))
    if init_hash != ref_hash:
        raise RuntimeError(f"Initialization mismatch for {method}: {init_hash} != {ref_hash}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    formal_elapsed = 0.0
    adam_completed = 0
    adam_step_times = []
    discarded_adam_overshoot = False
    history = []

    model.train()
    print("=" * 112)
    print(f"MATCHED TIME | Ra=1e{RA_EXPONENT} | {METHOD_LABELS[method]} | seed={seed} | target={target_seconds:.6f} s")
    print("=" * 112)

    for step in range(1, ADAM_STEPS + 1):
        remaining = target_seconds - formal_elapsed
        if remaining <= 0:
            break
        avg = float(np.mean(adam_step_times[-20:])) if adam_step_times else 0.0
        keep_snapshot = avg > 0 and remaining <= max(0.5, 3.0 * avg)
        pre_state = cpu_state_dict(model) if keep_snapshot else None

        sync()
        t0 = time.perf_counter()
        optimizer_adam.zero_grad(set_to_none=True)
        d = loss_components(model, train_pts)
        loss = d["total_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss for {method} during Adam step {step}: {loss}")
        loss.backward()
        optimizer_adam.step()
        sync()
        dt = time.perf_counter() - t0
        loss_value = float(loss.detach().cpu())
        del d, loss

        if formal_elapsed + dt > target_seconds and pre_state is not None:
            model.load_state_dict(pre_state, strict=True)
            discarded_adam_overshoot = True
            break

        formal_elapsed += dt
        adam_completed = step
        adam_step_times.append(dt)
        if step == 1 or step % 100 == 0 or step == ADAM_STEPS:
            history.append({
                "stage": "adam", "stage_step": step,
                "formal_elapsed_seconds": formal_elapsed,
                "training_loss_before_update": loss_value,
            })
        if formal_elapsed >= target_seconds:
            break

    lbfgs_outer_completed = 0
    lbfgs_closure_evals = 0
    discarded_lbfgs_overshoot = False
    discarded_lbfgs_step_seconds = 0.0
    last_closure_loss = np.nan
    last_closure_loss_tensor = None

    if adam_completed == ADAM_STEPS and formal_elapsed < target_seconds:
        optimizer_lbfgs = make_time_controlled_lbfgs_optimizer(model)
        while formal_elapsed < target_seconds:
            pre_state = cpu_state_dict(model)
            before = formal_elapsed
            closures_this_outer = 0

            def closure():
                nonlocal lbfgs_closure_evals, closures_this_outer, last_closure_loss_tensor
                optimizer_lbfgs.zero_grad(set_to_none=True)
                d = loss_components(model, train_pts)
                loss = d["total_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss for {method} during matched-time L-BFGS "
                        f"outer={lbfgs_outer_completed+1}, closure={lbfgs_closure_evals+1}: {loss}"
                    )
                loss.backward()
                lbfgs_closure_evals += 1
                closures_this_outer += 1


                last_closure_loss_tensor = loss.detach()
                return loss

            sync()
            t0 = time.perf_counter()
            optimizer_lbfgs.step(closure)
            sync()
            dt = time.perf_counter() - t0
            last_closure_loss = (
                float(last_closure_loss_tensor.cpu())
                if last_closure_loss_tensor is not None else np.nan
            )

            if before + dt > target_seconds:
                model.load_state_dict(pre_state, strict=True)
                discarded_lbfgs_overshoot = True
                discarded_lbfgs_step_seconds = float(dt)
                formal_elapsed = before
                break

            formal_elapsed = before + dt
            lbfgs_outer_completed += 1
            history.append({
                "stage": "lbfgs", "stage_step": lbfgs_outer_completed,
                "formal_elapsed_seconds": formal_elapsed,
                "training_loss_last_closure": last_closure_loss,
                "closures_this_outer": closures_this_outer,
                "closure_evals_total": lbfgs_closure_evals,
            })
            if lbfgs_outer_completed % 100 == 0:
                print(
                    f"L-BFGS outer {lbfgs_outer_completed:6d} | closures={lbfgs_closure_evals:6d} | "
                    f"formal={formal_elapsed:.3f}/{target_seconds:.3f} s | loss={last_closure_loss:.6e}"
                )
            if lbfgs_outer_completed >= 100000:
                raise RuntimeError("Matched-time L-BFGS safety cap reached before target time")

    eval_model, common, optimization, quantities = final_metrics(method, model, base, base["interior"])
    pd.DataFrame(history).to_csv(out / "matched_time_history.csv", index=False)

    summary = {
        "protocol": "matched formal wall-clock to fixed-budget NMAF seed9999",
        **architecture_metadata(),
        "Ra_exponent": int(RA_EXPONENT), "Ra": float(RA), "Pr": float(PR),
        "method": method, "method_label": METHOD_LABELS[method], "seed": seed,
        "activation_formula": METHOD_FORMULAS[method],
        "target_method": "nmaf",
        "target_formal_training_time_seconds": float(target_seconds),
        "selected_formal_training_time_seconds": float(formal_elapsed),
        "unused_target_time_seconds": float(max(0.0, target_seconds - formal_elapsed)),
        "adam_steps_completed": int(adam_completed),
        "lbfgs_outer_updates_completed": int(lbfgs_outer_completed),
        "lbfgs_closure_evaluations_attempted": int(lbfgs_closure_evals),
        "discarded_adam_overshoot_step": bool(discarded_adam_overshoot),
        "discarded_lbfgs_overshoot_outer_update": bool(discarded_lbfgs_overshoot),
        "discarded_lbfgs_outer_update_seconds": float(discarded_lbfgs_step_seconds),
        "initial_linear_state_sha256": init_hash,
        "training_points_sha256": training_points_sha256(seed),
        "trainable_parameters": int(count_parameters(model)),
        "final_common_total_loss": float(common["total_loss"]),
        "final_common_pde_loss": float(common["pde_loss"]),
        "final_common_bc_loss": float(common["bc_loss"]),
        "physical_quantities": quantities,
        "nmaf_target_summary_source": str(formal_summary_path("nmaf", seed)),
        "timing_note": "Only training compute is counted; post-hoc evaluation and I/O are outside the formal timer.",
    }
    save_json(out / "training_summary.json", summary)
    torch.save(
        {"model_state_dict": cpu_state_dict(eval_model), "summary": summary},
        out / f"Ra1e{RA_EXPONENT}_{method}_seed{seed}_matched_time.pth",
    )
    print(f"DONE MATCHED | {METHOD_LABELS[method]} | used={formal_elapsed:.6f}s / target={target_seconds:.6f}s")
    return summary


def train_all_matched_time(seed=9999, skip_existing=True):
    target_seconds, _ = _load_nmaf_target(seed)
    print(f"NMAF target formal time: {target_seconds:.6f} s")
    rows = []
    for method in METHODS:
        try:
            if method == "nmaf":
                rows.append(train_matched_time_seed(method, seed))
                continue
            out = matched_out_dir(method, seed)
            summary_path = out / "training_summary.json"
            if skip_existing and summary_path.exists():
                print(f"KEEP existing matched-time result: {summary_path}")
                rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            else:
                rows.append(train_matched_time_seed(method, seed))
        finally:
            release_method_memory(f"matched-time {method}")
    save_json(MATCHED_ROOT / "all_matched_time_summaries_seed9999.json", rows)
    return rows
