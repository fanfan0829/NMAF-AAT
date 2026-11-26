from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_PATH = SCRIPT_DIR / "wave_common_fixed_points_adam_lbfgs.py"
if not COMMON_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find local wave_common_fixed_points_adam_lbfgs.py:\n{COMMON_PATH}\n"
        "Put this analysis script in the Wave project root."
    )

_spec = importlib.util.spec_from_file_location(
    "wave_common_cpu_postprocess_local",
    COMMON_PATH,
)
W = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = W
_spec.loader.exec_module(W)

CPU = torch.device("cpu")
W.DEVICE = CPU

RESULT_ROOT = SCRIPT_DIR / "wave_multiseed_results"
OUTPUT_ROOT = RESULT_ROOT / "wave_cpu_postprocessing"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [int(s) for s in W.SEEDS]
METHODS = ["tanh", "nmaf_aat", "rar", "stan"]
METHOD_LABELS = {
    "tanh": "Standard PINN",
    "nmaf_aat": "NMAF-AAT",
    "rar": "RAR-guided PINN",
    "stan": "Stan-PINN",
}
CHECKPOINTS = [spec["name"] for spec in W.CHECKPOINT_SPECS]
CHECKPOINT_LABELS = {
    "adam0500000": "Adam 500k",
    "total0700000": "Adam 500k + L-BFGS",
}

RAR_METHOD_DIR_NAME = "rar_adam_integrated_base6400_plus1280"
IMAGE_DPI = 800
SHOW_FIGURES = True

def configure_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.7,
    })

def mean_std(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, 0
    if len(s) == 1:
        return float(s.iloc[0]), np.nan, 1
    return float(s.mean()), float(s.std(ddof=1)), int(len(s))

def build_model(method):
    if method in ("tanh", "rar"):
        model = W.TanhPINN(W.HIDDEN_LAYERS, W.HIDDEN_NEURONS)
    elif method == "nmaf_aat":
        model = W.NMAFAATPINN(
            W.HIDDEN_LAYERS,
            W.HIDDEN_NEURONS,
            dtype=W.DTYPE_LBFGS,
        )
    elif method == "stan":
        if not hasattr(W, "StanPINN"):
            raise AttributeError(
                "The local wave_common_fixed_points_adam_lbfgs.py has no StanPINN. "
                "Use the current four-method Wave common file."
            )
        model = W.StanPINN(
            W.HIDDEN_LAYERS,
            W.HIDDEN_NEURONS,
            dtype=W.DTYPE_LBFGS,
        )
    else:
        raise ValueError(method)
    return model.to(device=CPU, dtype=W.DTYPE_LBFGS)

def rar_dir():
    preferred = RESULT_ROOT / RAR_METHOD_DIR_NAME
    if preferred.is_dir():
        return preferred

    candidates = []
    if RESULT_ROOT.exists():
        for p in RESULT_ROOT.iterdir():
            if not p.is_dir() or "rar" not in p.name.lower():
                continue
            good = True
            for seed in SEEDS:
                sd = p / f"seed_{seed}"
                if not sd.is_dir():
                    good = False
                    break
            if good:
                candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        "Cannot uniquely locate the Wave RAR result directory. "
        f"Preferred: {preferred}; candidates={candidates}"
    )

def checkpoint_path(method, seed, checkpoint_name):
    if method == "rar":
        sd = rar_dir() / f"seed_{seed}"
        exact = sd / f"wave_{RAR_METHOD_DIR_NAME}_seed{seed}_{checkpoint_name}.pth"
        if exact.exists():
            return exact
        matches = sorted(sd.glob(f"wave_*_seed{seed}_{checkpoint_name}.pth"))
        rar_matches = [p for p in matches if "rar" in p.name.lower()]
        if len(rar_matches) == 1:
            return rar_matches[0]
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Cannot uniquely locate RAR checkpoint for seed={seed}, "
            f"checkpoint={checkpoint_name}: {matches}"
        )

    return (
        RESULT_ROOT
        / method
        / f"seed_{seed}"
        / f"wave_{method}_seed{seed}_{checkpoint_name}.pth"
    )

def load_model(method, seed, checkpoint_name):
    path = checkpoint_path(method, seed, checkpoint_name)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        ck = torch.load(path, map_location=CPU, weights_only=False)
    except TypeError:
        ck = torch.load(path, map_location=CPU)

    state = ck["model_state_dict"]
    state = {
        k: (v.to(device=CPU, dtype=W.DTYPE_LBFGS) if torch.is_tensor(v) else v)
        for k, v in state.items()
    }

    model = build_model(method)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ck, path

@torch.no_grad()
def predict_field(model, X, T, batch_size=20000):
    xf = torch.as_tensor(
        X.reshape(-1, 1),
        dtype=W.DTYPE_LBFGS,
        device=CPU,
    )
    tf = torch.as_tensor(
        T.reshape(-1, 1),
        dtype=W.DTYPE_LBFGS,
        device=CPU,
    )

    out = []
    for start in range(0, len(xf), batch_size):
        end = min(start + batch_size, len(xf))
        out.append(
            model(xf[start:end], tf[start:end])
            .detach()
            .cpu()
            .numpy()
        )

    return np.vstack(out).reshape(X.shape)

def exact_field(X, T):
    return (
        np.sin(2.0 * np.pi * X) * np.sin(2.0 * np.pi * T)
        + np.sin(10.0 * np.pi * X) * np.sin(10.0 * np.pi * T)
    )

def trapz(y, x, axis):
    fn = getattr(np, "trapezoid", np.trapz)
    return fn(y, x=x, axis=axis)

def cleanup_model(model):
    del model
    gc.collect()


OUT = OUTPUT_ROOT / "06_physics_consistency"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "total0700000"
N_INTERIOR_X = 101
N_INTERIOR_T = 101
N_BOUNDARY_T = 401
N_IC_X = 801
AUTOGRAD_BATCH = 2048

def pde_residual_metrics(model):
    xg = np.linspace(0.0, 1.0, N_INTERIOR_X)
    tg = np.linspace(0.0, 1.0, N_INTERIOR_T)
    X, T = np.meshgrid(xg, tg, indexing="ij")
    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    pieces = []
    for start in range(0, len(xf), AUTOGRAD_BATCH):
        end = min(start + AUTOGRAD_BATCH, len(xf))

        x = torch.tensor(
            xf[start:end],
            dtype=W.DTYPE_LBFGS,
            device=CPU,
            requires_grad=True,
        )
        t = torch.tensor(
            tf[start:end],
            dtype=W.DTYPE_LBFGS,
            device=CPU,
            requires_grad=True,
        )

        u = model(x, t)
        one = torch.ones_like(u)

        ux = torch.autograd.grad(
            u, x, grad_outputs=one,
            create_graph=True, retain_graph=True
        )[0]
        ut = torch.autograd.grad(
            u, t, grad_outputs=one,
            create_graph=True, retain_graph=True
        )[0]

        uxx = torch.autograd.grad(
            ux, x, grad_outputs=torch.ones_like(ux),
            create_graph=False, retain_graph=True
        )[0]
        utt = torch.autograd.grad(
            ut, t, grad_outputs=torch.ones_like(ut),
            create_graph=False, retain_graph=False
        )[0]

        r = (utt - uxx).detach().cpu().numpy().reshape(-1)
        pieces.append(r)

        del x, t, u, ux, ut, uxx, utt, r
        gc.collect()

    residual = np.concatenate(pieces)
    return {
        "pde_residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "pde_residual_max_abs": float(np.max(np.abs(residual))),
    }

@torch.no_grad()
def boundary_metrics(model):
    t = torch.linspace(
        0.0, 1.0, N_BOUNDARY_T,
        dtype=W.DTYPE_LBFGS, device=CPU
    ).reshape(-1, 1)
    x0 = torch.zeros_like(t)
    x1 = torch.ones_like(t)
    u0 = model(x0, t).detach().cpu().numpy().reshape(-1)
    u1 = model(x1, t).detach().cpu().numpy().reshape(-1)
    both = np.concatenate([u0, u1])
    return {
        "boundary_value_rms": float(np.sqrt(np.mean(both ** 2))),
        "boundary_value_max_abs": float(np.max(np.abs(both))),
    }

def initial_condition_metrics(model):
    x = torch.linspace(
        0.0, 1.0, N_IC_X,
        dtype=W.DTYPE_LBFGS, device=CPU
    ).reshape(-1, 1)
    t0 = torch.zeros_like(x, requires_grad=True)

    u0 = model(x, t0)
    ut0 = torch.autograd.grad(
        u0,
        t0,
        grad_outputs=torch.ones_like(u0),
        create_graph=False,
        retain_graph=False,
    )[0]

    x_np = x.detach().cpu().numpy().reshape(-1)
    u0_np = u0.detach().cpu().numpy().reshape(-1)
    ut_np = ut0.detach().cpu().numpy().reshape(-1)

    ut_exact = (
        2.0 * np.pi * np.sin(2.0 * np.pi * x_np)
        + 10.0 * np.pi * np.sin(10.0 * np.pi * x_np)
    )

    low_basis = np.sin(2.0 * np.pi * x_np)
    high_basis = np.sin(10.0 * np.pi * x_np)

    low_pred = 2.0 * trapz(ut_np * low_basis, x_np, axis=0)
    high_pred = 2.0 * trapz(ut_np * high_basis, x_np, axis=0)

    return {
        "initial_displacement_rms": float(np.sqrt(np.mean(u0_np ** 2))),
        "initial_velocity_rms_error": float(
            np.sqrt(np.mean((ut_np - ut_exact) ** 2))
        ),
        "initial_velocity_low_mode_relative_amplitude_error": float(
            abs(low_pred - 2.0 * np.pi) / (2.0 * np.pi)
        ),
        "initial_velocity_high_mode_relative_amplitude_error": float(
            abs(high_pred - 10.0 * np.pi) / (10.0 * np.pi)
        ),
    }

def main():
    configure_style()

    rows = []
    for method in METHODS:
        for seed in SEEDS:
            model, ck, path = load_model(method, seed, CHECKPOINT)

            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "checkpoint_file": str(path),
            }
            row.update(pde_residual_metrics(model))
            row.update(boundary_metrics(model))
            row.update(initial_condition_metrics(model))
            rows.append(row)

            cleanup_model(model)

    seedwise = pd.DataFrame(rows)
    seedwise.to_csv(OUT / "physics_consistency_seedwise.csv", index=False)

    metrics = [
        "pde_residual_rms",
        "pde_residual_max_abs",
        "boundary_value_rms",
        "boundary_value_max_abs",
        "initial_displacement_rms",
        "initial_velocity_rms_error",
        "initial_velocity_low_mode_relative_amplitude_error",
        "initial_velocity_high_mode_relative_amplitude_error",
    ]

    summary_rows = []
    for method, g in seedwise.groupby("method"):
        r = {"method": method, "method_label": METHOD_LABELS[method]}
        for metric in metrics:
            mu, sd, n = mean_std(g[metric])
            r[metric + "_mean"] = mu
            r[metric + "_sample_std"] = sd
            r[metric + "_n"] = n
        summary_rows.append(r)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "physics_consistency_mean_std.csv", index=False)

    summary["order"] = summary["method"].map({m: i for i, m in enumerate(METHODS)})
    summary = summary.sort_values("order")

    xloc = np.arange(len(summary), dtype=float)
    fig, ax = plt.subplots(figsize=(8.3, 5.1))
    ax.bar(
        xloc,
        summary["pde_residual_rms_mean"],
        yerr=summary["pde_residual_rms_sample_std"],
        capsize=4,
    )
    ax.set_yscale("log")
    ax.set_xticks(xloc)
    ax.set_xticklabels(summary["method_label"])
    ax.set_ylabel("PDE residual RMS")
    ax.grid(True, axis="y", which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig(
        OUT / "wave_pde_residual_rms_final.png",
        dpi=IMAGE_DPI,
        bbox_inches="tight",
    )
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

    print("CPU-only physics-consistency analysis completed:", OUT.resolve())

if __name__ == "__main__":
    main()
