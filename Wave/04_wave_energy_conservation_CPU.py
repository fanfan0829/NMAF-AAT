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


OUT = OUTPUT_ROOT / "04_energy_conservation"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "total0700000"
N_X = 801
N_T = 401
EXACT_ENERGY = 26.0 * np.pi ** 2

def energy_curve(x, t, u):
    ux = np.gradient(u, x, axis=0, edge_order=2)
    ut = np.gradient(u, t, axis=1, edge_order=2)
    density = 0.5 * (ut ** 2 + ux ** 2)
    return trapz(density, x, axis=0)

def main():
    configure_style()

    x = np.linspace(0.0, 1.0, N_X)
    t = np.linspace(0.0, 1.0, N_T)
    X, T = np.meshgrid(x, t, indexing="ij")

    rows = []
    curves = {}

    for method in METHODS:
        method_curves = []
        for seed in SEEDS:
            model, ck, path = load_model(method, seed, CHECKPOINT)
            pred = predict_field(model, X, T)
            E = energy_curve(x, t, pred)
            rel = np.abs(E - EXACT_ENERGY) / EXACT_ENERGY

            rows.append({
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed": seed,
                "mean_relative_energy_error": float(np.mean(rel)),
                "max_relative_energy_error": float(np.max(rel)),
                "final_relative_energy_error": float(rel[-1]),
                "relative_energy_range": float(
                    (np.max(E) - np.min(E)) / EXACT_ENERGY
                ),
                "checkpoint_file": str(path),
            })

            method_curves.append(rel)
            np.savez_compressed(
                OUT / f"{method}_seed{seed}_energy_curve.npz",
                t=t,
                predicted_energy=E,
                exact_energy=np.full_like(t, EXACT_ENERGY),
                relative_energy_error=rel,
            )
            cleanup_model(model)

        curves[method] = np.vstack(method_curves)

    seedwise = pd.DataFrame(rows)
    seedwise.to_csv(OUT / "energy_conservation_seedwise.csv", index=False)

    summary_rows = []
    for method, g in seedwise.groupby("method"):
        r = {"method": method, "method_label": METHOD_LABELS[method]}
        for metric in [
            "mean_relative_energy_error",
            "max_relative_energy_error",
            "final_relative_energy_error",
            "relative_energy_range",
        ]:
            mu, sd, n = mean_std(g[metric])
            r[metric + "_mean"] = mu
            r[metric + "_sample_std"] = sd
            r[metric + "_n"] = n
        summary_rows.append(r)

    pd.DataFrame(summary_rows).to_csv(
        OUT / "energy_conservation_mean_std.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8.3, 5.1))
    for method in METHODS:
        arr = curves[method]
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        line, = ax.plot(t, mu, label=METHOD_LABELS[method])
        ax.fill_between(
            t,
            np.maximum(mu - sd, 0.0),
            mu + sd,
            alpha=0.14,
            color=line.get_color(),
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Relative energy error")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "wave_energy_error_vs_time_final.png",
                dpi=IMAGE_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

    print("Exact conserved energy =", EXACT_ENERGY)
    print("CPU-only energy-conservation analysis completed:", OUT.resolve())

if __name__ == "__main__":
    main()
