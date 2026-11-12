from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from wave_common_fixed_points_adam_lbfgs import (
    DEVICE,
    DTYPE_LBFGS,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    INITIAL_ALPHA,
    INITIAL_BETA,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    SEEDS,
    build_model,
    state_dict_to_dtype,
)


METHODS = ("tanh", "nmaf_aat")
METHOD_LABELS = {"tanh": "Standard PINN", "nmaf_aat": "NMAF-AAT"}

CHECKPOINTS = {
    "Adam 500k": "adam0500000",
    "Final 700k": "total0700000",
}

ACTUAL_HIDDEN_LAYERS = HIDDEN_LAYERS + 1
assert HIDDEN_LAYERS == 4
assert ACTUAL_HIDDEN_LAYERS == 5
assert HIDDEN_NEURONS == 128

ANALYSIS_DIR = RESULT_ROOT / "reviewer_wave_frequency_mechanism_500k_700k"
IMAGE_DPI = 800


MODAL_GRID_N = 301
PREDICTION_BATCH = 32768


SLICE_T = 0.25
SLICE_N = 4096
FFT_MAX_FREQUENCY = 12.0


DERIVATIVE_GRID_N = 64
DERIVATIVE_SMALL_THRESHOLD = 1.0e-3


RUN_RESIDUAL_JACOBIAN = False
JACOBIAN_GRID_N = 6
JACOBIAN_POSITIVE_REL_TOL = 1.0e-12


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.7,
    })


def checkpoint_path(method, seed, checkpoint_name):
    seed_dir = RESULT_ROOT / method / f"seed_{seed}"
    return seed_dir / f"wave_{method}_seed{seed}_{checkpoint_name}.pth"


def load_model(method, seed, checkpoint_name):
    path = checkpoint_path(method, seed, checkpoint_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            "This script never retrains models."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(method, DTYPE_LBFGS)
    state = state_dict_to_dtype(ckpt["model_state_dict"], DTYPE_LBFGS)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt, path


def exact_np(x, t):
    return (
        np.sin(2*np.pi*x) * np.sin(2*np.pi*t)
        + np.sin(10*np.pi*x) * np.sin(10*np.pi*t)
    )


@torch.no_grad()
def predict(model, x, t):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    out = np.empty_like(x)
    for s in range(0, len(x), PREDICTION_BATCH):
        e = min(s + PREDICTION_BATCH, len(x))
        xt = torch.tensor(x[s:e, None], dtype=DTYPE_LBFGS, device=DEVICE)
        tt = torch.tensor(t[s:e, None], dtype=DTYPE_LBFGS, device=DEVICE)
        out[s:e] = model(xt, tt).detach().cpu().numpy().reshape(-1)
    return out


def trapz(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def inner2(a, b, x, t):
    return float(trapz(trapz(a*b, x=x, axis=1), x=t, axis=0))


def norm2(a, x, t):
    return math.sqrt(max(inner2(a, a, x, t), 0.0))


def modal_metrics(method, seed, stage_label, checkpoint_name):
    model, ckpt, path = load_model(method, seed, checkpoint_name)

    x = np.linspace(0.0, 1.0, MODAL_GRID_N)
    t = np.linspace(0.0, 1.0, MODAL_GRID_N)
    X, T = np.meshgrid(x, t)

    pred = predict(model, X.ravel(), T.ravel()).reshape(X.shape)
    exact = exact_np(X, T)

    phi_low = np.sin(2*np.pi*X) * np.sin(2*np.pi*T)
    phi_high = np.sin(10*np.pi*X) * np.sin(10*np.pi*T)

    a_low = inner2(pred, phi_low, x, t) / inner2(phi_low, phi_low, x, t)
    a_high = inner2(pred, phi_high, x, t) / inner2(phi_high, phi_high, x, t)

    e_low = abs(a_low - 1.0)
    e_high = abs(a_high - 1.0)

    projection = a_low*phi_low + a_high*phi_high
    leakage = norm2(pred - projection, x, t) / max(norm2(pred, x, t), 1e-300)
    rel_l2 = norm2(pred - exact, x, t) / norm2(exact, x, t)

    return {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": int(seed),
        "stage": stage_label,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(path),
        "low_mode_exact_coefficient": 1.0,
        "low_mode_predicted_coefficient": float(a_low),
        "low_mode_relative_error": float(e_low),
        "high_mode_exact_coefficient": 1.0,
        "high_mode_predicted_coefficient": float(a_high),
        "high_mode_relative_error": float(e_high),
        "high_to_low_mode_error_ratio": float(e_high / e_low) if e_low > 0 else np.inf,
        "non_target_spectral_leakage_ratio": float(leakage),
        "overall_relative_l2_error": float(rel_l2),
        "maximum_absolute_error": float(np.max(np.abs(pred - exact))),
        "stored_total_iteration": ckpt.get("total_iteration"),
        "stored_lbfgs_eval": ckpt.get("lbfgs_eval"),
    }


def run_frequency_analysis():
    rows = []
    for stage_label, checkpoint_name in CHECKPOINTS.items():
        for method in METHODS:
            for seed in SEEDS:
                print(f"Modal analysis | {stage_label} | {METHOD_LABELS[method]} | seed={seed}")
                rows.append(modal_metrics(method, seed, stage_label, checkpoint_name))

    individual = pd.DataFrame(rows)
    individual.to_csv(
        ANALYSIS_DIR / "frequency_resolved_individual_500k_700k.csv",
        index=False
    )

    metric_names = [
        "low_mode_predicted_coefficient",
        "high_mode_predicted_coefficient",
        "low_mode_relative_error",
        "high_mode_relative_error",
        "high_to_low_mode_error_ratio",
        "non_target_spectral_leakage_ratio",
        "overall_relative_l2_error",
        "maximum_absolute_error",
    ]
    stats_rows = []
    for stage_label in CHECKPOINTS:
        for method in METHODS:
            d = individual[
                (individual["stage"] == stage_label) &
                (individual["method"] == method)
            ]
            row = {
                "stage": stage_label,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_seeds": len(d),
            }
            for metric in metric_names:
                a = d[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.mean(a))
                row[f"{metric}_sample_std"] = float(np.std(a, ddof=1))
            stats_rows.append(row)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(
        ANALYSIS_DIR / "frequency_resolved_statistics_500k_700k.csv",
        index=False
    )


    for stage_label in CHECKPOINTS:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        xpos = np.arange(len(METHODS))
        width = 0.34

        low_mean, low_std, high_mean, high_std = [], [], [], []
        for method in METHODS:
            d = individual[
                (individual["stage"] == stage_label) &
                (individual["method"] == method)
            ]
            low = d["low_mode_relative_error"].to_numpy(float)
            high = d["high_mode_relative_error"].to_numpy(float)
            low_mean.append(low.mean()); low_std.append(low.std(ddof=1))
            high_mean.append(high.mean()); high_std.append(high.std(ddof=1))

        ax.bar(
            xpos-width/2, low_mean, width, yerr=low_std, capsize=4,
            label=r"Low-frequency mode ($2\pi$)"
        )
        ax.bar(
            xpos+width/2, high_mean, width, yerr=high_std, capsize=4,
            label=r"High-frequency mode ($10\pi$)"
        )


        jit = np.linspace(-0.045, 0.045, len(SEEDS))
        for i, method in enumerate(METHODS):
            d = individual[
                (individual["stage"] == stage_label) &
                (individual["method"] == method)
            ].set_index("seed")
            for j, seed in enumerate(SEEDS):
                ax.scatter(
                    xpos[i]-width/2+jit[j],
                    d.loc[seed, "low_mode_relative_error"],
                    s=24, zorder=5
                )
                ax.scatter(
                    xpos[i]+width/2+jit[j],
                    d.loc[seed, "high_mode_relative_error"],
                    s=24, zorder=5
                )

        ax.set_xticks(xpos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
        ax.set_ylabel("Mode-wise relative error")
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.35)
        ax.legend(frameon=True)
        fig.tight_layout()
        safe = stage_label.lower().replace(" ", "_")
        p = ANALYSIS_DIR / f"figure_frequency_resolved_{safe}.jpg"
        fig.savefig(p, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
        plt.close(fig)

    return individual, stats


def run_slice_spectra():
    x = np.linspace(0.0, 1.0, SLICE_N, endpoint=False)
    t = np.full_like(x, SLICE_T)
    exact = exact_np(x, t)
    freq = np.fft.rfftfreq(SLICE_N, d=1.0/SLICE_N)
    exact_amp = 2*np.abs(np.fft.rfft(exact))/SLICE_N

    rows = []
    for stage_label, ck_name in CHECKPOINTS.items():
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        mask = freq <= FFT_MAX_FREQUENCY
        ax.plot(freq[mask], exact_amp[mask], label="Exact", linewidth=2.0)

        for method in METHODS:
            model, _, _ = load_model(method, REPRESENTATIVE_SEED, ck_name)
            pred = predict(model, x, t)
            amp = 2*np.abs(np.fft.rfft(pred))/SLICE_N
            ax.plot(freq[mask], amp[mask], label=METHOD_LABELS[method])

            for f, p, e in zip(freq[mask], amp[mask], exact_amp[mask]):
                rows.append({
                    "stage": stage_label,
                    "method": method,
                    "seed": REPRESENTATIVE_SEED,
                    "t": SLICE_T,
                    "frequency_cycles_per_unit": float(f),
                    "predicted_amplitude": float(p),
                    "exact_amplitude": float(e),
                })

        ax.axvline(1.0, linestyle=":", linewidth=1.2)
        ax.axvline(5.0, linestyle=":", linewidth=1.2)
        ax.set_xlim(0, FFT_MAX_FREQUENCY)
        ax.set_xlabel("Spatial frequency (cycles/unit length)")
        ax.set_ylabel("Fourier amplitude")
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=True)
        fig.tight_layout()
        safe = stage_label.lower().replace(" ", "_")
        p = ANALYSIS_DIR / f"figure_spatial_spectrum_{safe}_seed{REPRESENTATIVE_SEED}.jpg"
        fig.savefig(p, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(rows).to_csv(
        ANALYSIS_DIR / f"spatial_spectrum_seed{REPRESENTATIVE_SEED}_500k_700k.csv",
        index=False
    )


def summarize(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    return {
        "mean": float(a.mean()),
        "sample_std": float(a.std(ddof=1)),
        "min": float(a.min()),
        "q05": float(np.quantile(a, 0.05)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
        "q95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def aat_arrays(model):
    return (
        model.activation.alpha.detach().cpu().numpy().astype(float).copy(),
        model.activation.beta.detach().cpu().numpy().astype(float).copy(),
    )


def run_aat_evolution():
    rows = []
    raw = {}

    for seed in SEEDS:
        stages = [
            ("Initialization", None),
            ("Adam 500k", CHECKPOINTS["Adam 500k"]),
            ("Final 700k", CHECKPOINTS["Final 700k"]),
        ]
        for stage, ck_name in stages:
            if ck_name is None:
                alpha = np.full(HIDDEN_NEURONS, INITIAL_ALPHA, dtype=float)
                beta = np.full(HIDDEN_NEURONS, INITIAL_BETA, dtype=float)
            else:
                model, _, _ = load_model("nmaf_aat", seed, ck_name)
                alpha, beta = aat_arrays(model)

            raw[f"seed{seed}_{stage.replace(' ','_')}_alpha"] = alpha
            raw[f"seed{seed}_{stage.replace(' ','_')}_beta"] = beta

            for name, arr, init in (
                ("alpha", alpha, INITIAL_ALPHA),
                ("beta", beta, INITIAL_BETA),
            ):
                rows.append({
                    "seed": int(seed),
                    "stage": stage,
                    "parameter": name,
                    "initial_value": float(init),
                    "l2_change_from_initialization": float(np.linalg.norm(arr-init)),
                    "mean_abs_change_from_initialization": float(np.mean(np.abs(arr-init))),
                    **summarize(arr),
                })

    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS_DIR / "aat_parameter_evolution_500k_700k.csv", index=False)
    np.savez_compressed(ANALYSIS_DIR / "aat_parameter_raw_arrays.npz", **raw)


    seed = REPRESENTATIVE_SEED
    for parameter, init in (("alpha", INITIAL_ALPHA), ("beta", INITIAL_BETA)):
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.axvline(init, linestyle="--", linewidth=1.7, label="Initialization")

        for stage, ck_name in (
            ("Adam 500k", CHECKPOINTS["Adam 500k"]),
            ("Final 700k", CHECKPOINTS["Final 700k"]),
        ):
            arr = raw[f"seed{seed}_{stage.replace(' ','_')}_{parameter}"]
            ax.hist(arr, bins=24, alpha=0.45, density=True, label=stage)

        ax.set_xlabel(r"$\alpha$" if parameter=="alpha" else r"$\beta$")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.30)
        ax.legend(frameon=True)
        fig.tight_layout()
        p = ANALYSIS_DIR / f"figure_aat_{parameter}_evolution_seed{seed}.jpg"
        fig.savefig(p, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
        plt.close(fig)

    return df


def nmaf_prime(X):
    tanhX = torch.tanh(X)
    sech2 = 1.0 - tanhX**2
    return tanhX + X*sech2 + (X**2 - 1.0)*torch.exp(-0.5*X**2)


@torch.no_grad()
def preactivations(model, x, t):
    z0 = torch.cat([x, t], dim=1)
    zs = []
    z = model.input_layer(z0)
    zs.append(z)
    h = model.activation(z)
    for layer in model.hidden_layers:
        z = layer(h)
        zs.append(z)
        h = model.activation(z)
    assert len(zs) == ACTUAL_HIDDEN_LAYERS
    return zs


def activation_derivative(method, model, z):
    if method == "tanh":
        return 1.0 - torch.tanh(z)**2
    alpha = model.activation.alpha.reshape(1, -1)
    beta = model.activation.beta.reshape(1, -1)
    X = alpha*z + beta

    return alpha*nmaf_prime(X)


def run_derivative_analysis():
    q = np.linspace(
        1/(DERIVATIVE_GRID_N+1),
        DERIVATIVE_GRID_N/(DERIVATIVE_GRID_N+1),
        DERIVATIVE_GRID_N
    )
    X, T = np.meshgrid(q, q)
    x = torch.tensor(X.reshape(-1,1), dtype=DTYPE_LBFGS, device=DEVICE)
    t = torch.tensor(T.reshape(-1,1), dtype=DTYPE_LBFGS, device=DEVICE)

    rows = []
    for stage_label, ck_name in CHECKPOINTS.items():
        for method in METHODS:
            for seed in SEEDS:
                model, _, _ = load_model(method, seed, ck_name)
                zs = preactivations(model, x, t)
                for loc, z in enumerate(zs, start=1):
                    a = activation_derivative(method, model, z).abs().detach().cpu().numpy().reshape(-1)
                    rows.append({
                        "stage": stage_label,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "seed": int(seed),
                        "activation_location": loc,
                        "mean_abs_derivative": float(a.mean()),
                        "median_abs_derivative": float(np.median(a)),
                        "q10_abs_derivative": float(np.quantile(a, 0.10)),
                        "q90_abs_derivative": float(np.quantile(a, 0.90)),
                        "fraction_below_1e-3": float(np.mean(a < DERIVATIVE_SMALL_THRESHOLD)),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(
        ANALYSIS_DIR / "activation_derivative_diagnostic_500k_700k.csv",
        index=False
    )


    for stage_label in CHECKPOINTS:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        for method in METHODS:
            d = df[
                (df["stage"] == stage_label) &
                (df["method"] == method) &
                (df["seed"] == REPRESENTATIVE_SEED)
            ].sort_values("activation_location")
            ax.plot(
                d["activation_location"],
                d["median_abs_derivative"],
                marker="o",
                label=METHOD_LABELS[method]
            )
        ax.set_xticks(range(1, ACTUAL_HIDDEN_LAYERS+1))
        ax.set_xlabel("Hidden activation location")
        ax.set_ylabel(r"Median $|\partial \phi/\partial z|$")
        ax.grid(True, alpha=0.35)
        ax.legend(frameon=True)
        fig.tight_layout()
        safe = stage_label.lower().replace(" ", "_")
        p = ANALYSIS_DIR / f"figure_activation_derivative_{safe}_seed{REPRESENTATIVE_SEED}.jpg"
        fig.savefig(p, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
        plt.close(fig)

    return df


def residual_scalar(model, xv, tv):
    x = torch.tensor([[xv]], dtype=DTYPE_LBFGS, device=DEVICE, requires_grad=True)
    t = torch.tensor([[tv]], dtype=DTYPE_LBFGS, device=DEVICE, requires_grad=True)
    u = model(x,t)
    ut = torch.autograd.grad(u,t,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    ux = torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True,retain_graph=True)[0]
    utt = torch.autograd.grad(ut,t,torch.ones_like(ut),create_graph=True,retain_graph=True)[0]
    uxx = torch.autograd.grad(ux,x,torch.ones_like(ux),create_graph=True,retain_graph=True)[0]
    return (utt-uxx).reshape(())


def residual_jacobian(model):
    q = np.linspace(1/(JACOBIAN_GRID_N+1), JACOBIAN_GRID_N/(JACOBIAN_GRID_N+1), JACOBIAN_GRID_N)
    points = [(x,t) for t in q for x in q]
    params = [p for p in model.parameters() if p.requires_grad]
    npar = sum(p.numel() for p in params)
    J = np.zeros((len(points), npar), dtype=np.float64)

    for i,(xv,tv) in enumerate(points):
        r = residual_scalar(model, float(xv), float(tv))
        grads = torch.autograd.grad(r, params, allow_unused=True)
        pieces = []
        for p,g in zip(params,grads):
            pieces.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
        J[i] = torch.cat(pieces).detach().cpu().numpy()
        print(f"Jacobian row {i+1}/{len(points)}")
    return J


def jacobian_metrics(J):
    G = J @ J.T
    G = (G+G.T)/2
    eig = np.maximum(np.linalg.eigvalsh(G),0)[::-1]
    lam_max = eig[0]
    pos = eig[eig > lam_max*JACOBIAN_POSITIVE_REL_TOL]
    cond = np.sqrt(pos[0]/pos[-1]) if len(pos)>=2 else np.nan
    if pos.sum()>0:
        p = pos/pos.sum()
        erank = float(np.exp(-np.sum(p*np.log(p))))
    else:
        erank = 0.0
    return eig, cond, erank, len(pos)


def run_jacobian_analysis():
    rows = []
    for stage_label, ck_name in CHECKPOINTS.items():
        for method in METHODS:
            for seed in SEEDS:
                print(f"Residual-Jacobian | {stage_label} | {METHOD_LABELS[method]} | seed={seed}")
                model,_,_ = load_model(method,seed,ck_name)
                J = residual_jacobian(model)
                eig,cond,erank,npos = jacobian_metrics(J)
                rows.append({
                    "stage": stage_label,
                    "method": method,
                    "seed": int(seed),
                    "jacobian_condition_number_positive": float(cond),
                    "effective_rank": float(erank),
                    "positive_eigenvalue_count": int(npos),
                    "lambda_max": float(eig[0]),
                    "lambda_min_positive": float(eig[npos-1]) if npos else 0.0,
                })
                pd.DataFrame({
                    "rank": np.arange(1,len(eig)+1),
                    "eigenvalue_JJt": eig,
                    "normalized_eigenvalue": eig/eig[0] if eig[0]>0 else eig
                }).to_csv(
                    ANALYSIS_DIR / f"residual_jacobian_spectrum_{method}_seed{seed}_{ck_name}.csv",
                    index=False
                )
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(
        ANALYSIS_DIR / "residual_jacobian_conditioning_500k_700k.csv",
        index=False
    )


def main():
    configure_plot_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


    missing = []
    for method in METHODS:
        for seed in SEEDS:
            for stage, ck_name in CHECKPOINTS.items():
                p = checkpoint_path(method, seed, ck_name)
                if not p.exists():
                    missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing formal checkpoints. This script will not retrain anything:\n"
            + "\n".join(missing)
        )

    frequency_individual, frequency_stats = run_frequency_analysis()
    run_slice_spectra()
    aat_df = run_aat_evolution()
    derivative_df = run_derivative_analysis()

    if RUN_RESIDUAL_JACOBIAN:
        run_jacobian_analysis()

    manifest = {
        "analysis": "Wave reviewer response using existing 500k and 700k checkpoints",
        "no_retraining": True,
        "methods": list(METHODS),
        "seeds": [int(s) for s in SEEDS],
        "representative_seed": int(REPRESENTATIVE_SEED),
        "available_training_stages": ["Initialization", "Adam 500k", "Final 700k"],
        "frequency_analysis_stages": list(CHECKPOINTS.keys()),
        "actual_architecture": "2 -> 128 x 5 -> 1",
        "NMAF_AAT_parameterization": "one shared neuron-wise alpha/beta vector reused at all 5 hidden activation locations",
        "frequency_modes": {
            "low": "sin(2*pi*x)sin(2*pi*t)",
            "high": "sin(10*pi*x)sin(10*pi*t)",
        },
        "interpretation_limit": (
            "With only 500k and 700k checkpoints, the script provides stage-wise evidence "
            "rather than a dense iteration-by-iteration spectral or AAT trajectory."
        ),
        "residual_jacobian_enabled": bool(RUN_RESIDUAL_JACOBIAN),
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n" + "="*100)
    print("Completed. No model was retrained.")
    print("Output:", ANALYSIS_DIR.resolve())
    print("="*100)


if __name__ == "__main__":
    main()
