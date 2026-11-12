from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results"
DATASET_ROOT = RESULT_ROOT / "generated_datasets"

SEEDS = [1234, 99, 2026]
METHODS = ["tanh", "nmaf_aat"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


BASE_DEPTH = 7
BASE_WIDTH = 128
BASE_N_RES = 3000
BASE_N_BC = 200
BASE_N_IC = 500
BASE_NOMINAL_POINTS = BASE_N_RES + BASE_N_BC + BASE_N_IC
BASE_COORDINATE_POINTS = BASE_N_RES + 2 * BASE_N_BC + BASE_N_IC

X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

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

CHECKPOINT_EVAL = 10000
PRINT_EVERY = 100

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    study: str
    level: float
    depth: int = BASE_DEPTH
    width: int = BASE_WIDTH
    data_factor: float = 1.0
    noise_pct: float = 0.0

    @property
    def is_baseline(self) -> bool:
        return (
            self.depth == BASE_DEPTH
            and self.width == BASE_WIDTH
            and self.data_factor == 1.0
            and self.noise_pct == 0.0
        )


TRAIN_EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("depth_5", "depth", 5, depth=5),
    ExperimentSpec("depth_9", "depth", 9, depth=9),

    ExperimentSpec("width_64", "width", 64, width=64),
    ExperimentSpec("width_192", "width", 192, width=192),

    ExperimentSpec("data_0p5x", "data", 0.5, data_factor=0.5),
    ExperimentSpec("data_2x", "data", 2.0, data_factor=2.0),
    ExperimentSpec("data_4x", "data", 4.0, data_factor=4.0),

    ExperimentSpec("noise_1pct", "noise", 1.0, noise_pct=1.0),
    ExperimentSpec("noise_5pct", "noise", 5.0, noise_pct=5.0),
    ExperimentSpec("noise_10pct", "noise", 10.0, noise_pct=10.0),
]

BASELINE_SPECS = {
    "depth": ExperimentSpec("baseline", "depth", 7),
    "width": ExperimentSpec("baseline", "width", 128),
    "data": ExperimentSpec("baseline", "data", 1.0),
    "noise": ExperimentSpec("baseline", "noise", 0.0),
}


def find_project_root() -> Path:


    candidates = [HERE, HERE.parent, HERE.parent.parent]
    for root in candidates:
        if (root / "allen_cahn_multiseed_results" / "fixed_training_points").is_dir():
            return root
    raise FileNotFoundError(
        "Could not find 'allen_cahn_multiseed_results/fixed_training_points'.\n"
        "Place this package inside the same project directory as the existing "
        "baseline result folder, or one level below it."
    )


def baseline_result_root() -> Path:
    return find_project_root() / "allen_cahn_multiseed_results"


def reference_file() -> Path:
    root = find_project_root()
    candidates = [
        root / "AC.mat",
        HERE / "AC.mat",
        HERE.parent / "AC.mat",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "AC.mat was not found. Put AC.mat in the project root or package directory."
    )


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


class TanhPINN(nn.Module):


    def __init__(self, depth: int, width: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = int(depth)
        self.width = int(width)

        self.input_layer = nn.Linear(2, width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(width, 1)
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
        in_features: int,
        initial_alpha: float = INITIAL_ALPHA,
        initial_beta: float = INITIAL_BETA,
    ):
        super().__init__()
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


    def __init__(self, depth: int, width: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = int(depth)
        self.width = int(width)


        self.input_layer = nn.Linear(2, width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(width, 1)
        self.activation = AdaptiveActivation(width)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = self.activation(self.input_layer(inputs))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        return self.output_layer(out)


def build_model(method: str, depth: int, width: int) -> nn.Module:
    if method == "tanh":
        model = TanhPINN(depth, width)
    elif method == "nmaf_aat":
        model = NMAFAATPINN(depth, width)
    else:
        raise ValueError(f"Unknown method: {method}")
    return model.double().to(DEVICE)


def method_label(method: str) -> str:
    return {"tanh": "Standard PINN", "nmaf_aat": "NMAF-AAT"}[method]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_adaptive_parameters(model: nn.Module) -> int:
    if hasattr(model, "activation") and hasattr(model.activation, "alpha"):
        return int(model.activation.alpha.numel() + model.activation.beta.numel())
    return 0


def baseline_points_path(seed: int) -> Path:
    return (
        baseline_result_root()
        / "fixed_training_points"
        / f"allen_cahn_training_points_seed{seed}.npz"
    )


def _load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    d = np.load(path)
    keys = ["x_res", "t_res", "x_bc_l", "x_bc_r", "t_bc", "x_ic", "t_ic"]
    return {k: np.asarray(d[k], dtype=np.float64) for k in keys}


def _generate_lhs_batch(seed: int, batch_index: int) -> Dict[str, np.ndarray]:


    offset = int(seed) * 100 + int(batch_index)

    res_sampler = qmc.LatinHypercube(d=2, seed=100000 + offset)
    res = res_sampler.random(n=BASE_N_RES)
    res = qmc.scale(
        res,
        np.array([X_MIN, T_MIN]),
        np.array([X_MAX, T_MAX]),
    )

    bc_sampler = qmc.LatinHypercube(d=1, seed=200000 + offset)
    t_bc = bc_sampler.random(n=BASE_N_BC) * (T_MAX - T_MIN) + T_MIN

    ic_sampler = qmc.LatinHypercube(d=1, seed=300000 + offset)
    x_ic = ic_sampler.random(n=BASE_N_IC) * (X_MAX - X_MIN) + X_MIN

    return {
        "x_res": res[:, 0:1].astype(np.float64),
        "t_res": res[:, 1:2].astype(np.float64),
        "x_bc_l": np.full((BASE_N_BC, 1), X_MIN, dtype=np.float64),
        "x_bc_r": np.full((BASE_N_BC, 1), X_MAX, dtype=np.float64),
        "t_bc": t_bc.astype(np.float64),
        "x_ic": x_ic.astype(np.float64),
        "t_ic": np.zeros((BASE_N_IC, 1), dtype=np.float64),
    }


def _subset_half(base: Dict[str, np.ndarray], seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(400000 + int(seed))

    idx_res = np.sort(rng.choice(BASE_N_RES, BASE_N_RES // 2, replace=False))
    idx_bc = np.sort(rng.choice(BASE_N_BC, BASE_N_BC // 2, replace=False))
    idx_ic = np.sort(rng.choice(BASE_N_IC, BASE_N_IC // 2, replace=False))

    return {
        "x_res": base["x_res"][idx_res],
        "t_res": base["t_res"][idx_res],
        "x_bc_l": base["x_bc_l"][idx_bc],
        "x_bc_r": base["x_bc_r"][idx_bc],
        "t_bc": base["t_bc"][idx_bc],
        "x_ic": base["x_ic"][idx_ic],
        "t_ic": base["t_ic"][idx_ic],
    }


def scaled_dataset_path(seed: int, factor: float) -> Path:
    tag = {0.5: "0p5x", 2.0: "2x", 4.0: "4x"}[float(factor)]
    return DATASET_ROOT / f"data_{tag}" / f"seed_{seed}.npz"


def prepare_scaled_dataset(seed: int, factor: float, overwrite: bool = False) -> Path:


    factor = float(factor)
    if factor not in (0.5, 2.0, 4.0):
        raise ValueError("factor must be 0.5, 2.0, or 4.0")

    out = scaled_dataset_path(seed, factor)
    if out.is_file() and not overwrite:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    base = _load_npz_arrays(baseline_points_path(seed))

    if factor == 0.5:
        arrays = _subset_half(base, seed)
    else:
        n_extra_batches = {2.0: 1, 4.0: 3}[factor]
        batches = [base] + [
            _generate_lhs_batch(seed, batch_index=i + 1)
            for i in range(n_extra_batches)
        ]
        arrays = {
            key: np.concatenate([b[key] for b in batches], axis=0)
            for key in base.keys()
        }

    np.savez_compressed(out, **arrays)

    manifest = {
        "seed": int(seed),
        "factor": factor,
        "baseline_source": str(baseline_points_path(seed)),
        "construction": (
            "deterministic subset of existing baseline points"
            if factor == 0.5
            else "existing baseline points plus deterministic additional LHS batches"
        ),
        "n_res": int(arrays["x_res"].shape[0]),
        "n_bc": int(arrays["t_bc"].shape[0]),
        "n_ic": int(arrays["x_ic"].shape[0]),
        "nominal_training_count": int(
            arrays["x_res"].shape[0]
            + arrays["t_bc"].shape[0]
            + arrays["x_ic"].shape[0]
        ),
        "physical_coordinate_count": int(
            arrays["x_res"].shape[0]
            + 2 * arrays["t_bc"].shape[0]
            + arrays["x_ic"].shape[0]
        ),
    }
    out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def prepare_all_generated_datasets(overwrite: bool = False) -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for factor in (0.5, 2.0, 4.0):
            prepare_scaled_dataset(seed, factor, overwrite=overwrite)


def data_path_for_spec(seed: int, spec: ExperimentSpec) -> Path:
    if spec.data_factor == 1.0:
        return baseline_points_path(seed)
    return prepare_scaled_dataset(seed, spec.data_factor, overwrite=False)


def noise_target_path(seed: int, noise_pct: float) -> Path:
    pct_tag = str(float(noise_pct)).replace(".", "p")
    return DATASET_ROOT / "noise_targets" / f"seed_{seed}_noise_{pct_tag}pct.npz"


def _build_noisy_ic_target(
    x_ic: np.ndarray, seed: int, noise_pct: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    exact = (x_ic ** 2) * np.cos(np.pi * x_ic)
    sigma = float(np.std(exact, ddof=0))
    rng = np.random.default_rng(500000 + int(seed))
    epsilon = rng.standard_normal(size=exact.shape)
    noisy = exact + (float(noise_pct) / 100.0) * sigma * epsilon
    return exact.astype(np.float64), noisy.astype(np.float64), sigma


def load_points_for_spec(seed: int, spec: ExperimentSpec) -> Dict[str, torch.Tensor]:
    arrays = _load_npz_arrays(data_path_for_spec(seed, spec))

    if spec.noise_pct > 0.0:
        exact, noisy, sigma = _build_noisy_ic_target(
            arrays["x_ic"], seed, spec.noise_pct
        )
        target = noisy

        out = noise_target_path(seed, spec.noise_pct)
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out,
                x_ic=arrays["x_ic"],
                u_ic_exact=exact,
                u_ic_noisy=noisy,
                sigma_ic=np.array([sigma], dtype=np.float64),
                noise_pct=np.array([spec.noise_pct], dtype=np.float64),
            )
    else:
        target = (arrays["x_ic"] ** 2) * np.cos(np.pi * arrays["x_ic"])

    tensors = {
        key: torch.tensor(value, dtype=DTYPE, device=DEVICE)
        for key, value in arrays.items()
    }
    tensors["u_ic_target"] = torch.tensor(
        target, dtype=DTYPE, device=DEVICE
    )
    return tensors


def sample_counts(spec: ExperimentSpec) -> Tuple[int, int, int]:
    if spec.data_factor == 1.0:
        return BASE_N_RES, BASE_N_BC, BASE_N_IC
    return (
        int(round(BASE_N_RES * spec.data_factor)),
        int(round(BASE_N_BC * spec.data_factor)),
        int(round(BASE_N_IC * spec.data_factor)),
    )


def nominal_training_count(spec: ExperimentSpec) -> int:
    n_res, n_bc, n_ic = sample_counts(spec)
    return n_res + n_bc + n_ic


def physical_coordinate_count(spec: ExperimentSpec) -> int:
    n_res, n_bc, n_ic = sample_counts(spec)
    return n_res + 2 * n_bc + n_ic


def spec_to_dict(spec: ExperimentSpec) -> Dict:
    d = asdict(spec)
    d["actual_hidden_layers"] = int(spec.depth)
    d["hidden_to_hidden_layers"] = int(spec.depth - 1)
    d["nominal_training_count"] = nominal_training_count(spec)
    d["physical_coordinate_count"] = physical_coordinate_count(spec)
    return d


def baseline_checkpoint_path(method: str, seed: int) -> Path:
    return (
        baseline_result_root()
        / method
        / f"seed_{seed}"
        / f"allen_cahn_{method}_seed{seed}_eval20000.pth"
    )


def verify_required_baseline_files(require_reference: bool = False) -> None:
    missing = []
    for seed in SEEDS:
        p = baseline_points_path(seed)
        if not p.is_file():
            missing.append(str(p))
        for method in METHODS:
            ckpt = baseline_checkpoint_path(method, seed)
            if not ckpt.is_file():
                missing.append(str(ckpt))
    if require_reference:
        try:
            reference_file()
        except FileNotFoundError as exc:
            missing.append(str(exc))

    if missing:
        raise FileNotFoundError(
            "Required baseline files are missing:\n  - " + "\n  - ".join(missing)
        )


def write_experiment_plan() -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = {
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "seeds": SEEDS,
        "methods": METHODS,
        "baseline_not_retrained": {
            "depth": BASE_DEPTH,
            "width": BASE_WIDTH,
            "n_res": BASE_N_RES,
            "n_bc": BASE_N_BC,
            "n_ic": BASE_N_IC,
            "noise_pct": 0.0,
        },
        "new_experiments": [spec_to_dict(s) for s in TRAIN_EXPERIMENTS],
        "noise_definition": (
            "Only initial-condition target values are perturbed: "
            "u_noisy = u_exact + (noise_pct/100) * std(u_exact) * epsilon, "
            "epsilon ~ N(0,1). PDE residual and periodic BC constraints remain exact."
        ),
    }
    path = RESULT_ROOT / "experiment_plan.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
