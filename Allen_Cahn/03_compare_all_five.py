from __future__ import annotations
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "allen_cahn_multiseed_results"
REFERENCE_FILE = SCRIPT_DIR / "AC.mat"

SEED = 99
CHECKPOINT_EVALS = (10000, 20000)


EXISTING_NMAF_DIR = None


EXISTING_TANH_DIR = RESULT_ROOT / "tanh" / f"seed_{SEED}"
EXISTING_NMAF_AAT_DIR = RESULT_ROOT / "nmaf_aat" / f"seed_{SEED}"

NEW_RATIONAL_DIR = RESULT_ROOT / "rational" / f"seed_{SEED}"
NEW_SIREN_DIR = RESULT_ROOT / "siren" / f"seed_{SEED}"

ANALYSIS_DIR = RESULT_ROOT / "broader_activation_comparison"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 128
OMEGA_0 = 30.0

METHODS = [
    ("tanh", "Standard PINN"),
    ("nmaf", "NMAF"),
    ("nmaf_aat", "NMAF-AAT"),
    ("rational", "Rational NN"),
    ("siren", "SIREN"),
]

CHECKPOINT_PREFIX = {
    "tanh": "allen_cahn_tanh",
    "nmaf": "allen_cahn_nmaf",
    "nmaf_aat": "allen_cahn_nmaf_aat",
    "rational": "allen_cahn_rational",
    "siren": "allen_cahn_siren",
}

METHOD_DIR = {
    "tanh": EXISTING_TANH_DIR,
    "nmaf_aat": EXISTING_NMAF_AAT_DIR,
    "rational": NEW_RATIONAL_DIR,
    "siren": NEW_SIREN_DIR,
}

METHOD_LABEL = dict(METHODS)


PLOT_DPI = 800
PLOT_FIGSIZE = (7.0, 5.0)
PLOT_LINEWIDTH = 1.5


METHOD_COLOR = {
    "tanh": "#1f77b4",
    "nmaf": "#9467bd",
    "nmaf_aat": "tab:red",
    "rational": "#2ca02c",
    "siren": "#ff7f0e",
}


def configure_paper_plot_style():

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "axes.unicode_minus": False,
    })


def locate_nmaf_dir():

    if EXISTING_NMAF_DIR is not None:
        p = Path(EXISTING_NMAF_DIR)
        if p.exists():
            return p
        print(f"[SKIP NMAF] EXISTING_NMAF_DIR does not exist: {p}")
        return None

    candidates = [
        RESULT_ROOT / "nmaf" / f"seed_{SEED}",
        RESULT_ROOT / "nmaf_fixed" / f"seed_{SEED}",
        RESULT_ROOT / "NMAF" / f"seed_{SEED}",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = list(
        RESULT_ROOT.rglob(
            f"allen_cahn_nmaf_seed{SEED}_eval20000.pth"
        )
    )
    if matches:
        return matches[0].parent

    print(
        "[SKIP NMAF] No existing fixed-NMAF result was found. "
        "NMAF will be omitted from this comparison."
    )
    return None


def nmaf_available():

    d = locate_nmaf_dir()
    if d is None:
        return False

    required = [
        d / "lbfgs_history.csv",
        *[
            d / f"{CHECKPOINT_PREFIX['nmaf']}_seed{SEED}_eval{target:05d}.pth"
            for target in CHECKPOINT_EVALS
        ],
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[SKIP NMAF] NMAF results are incomplete:")
        for p in missing:
            print("   missing:", p)
        return False

    return True


def get_active_methods():
    use_nmaf = nmaf_available()
    return [
        (method, label)
        for method, label in METHODS
        if method != "nmaf" or use_nmaf
    ]


def checkpoint_path(method, target):
    if method == "nmaf":
        d = locate_nmaf_dir()
        if d is None:
            raise FileNotFoundError("NMAF is unavailable and has been skipped.")
    else:
        d = METHOD_DIR[method]
    return d / f"{CHECKPOINT_PREFIX[method]}_seed{SEED}_eval{target:05d}.pth"


def history_path(method):
    if method == "nmaf":
        d = locate_nmaf_dir()
        if d is None:
            raise FileNotFoundError("NMAF is unavailable and has been skipped.")
    else:
        d = METHOD_DIR[method]
    return d / "lbfgs_history.csv"


class TanhPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_NEURONS)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            for _ in range(HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.activation = nn.Tanh()

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class NMAFActivation(nn.Module):
    def forward(self, x):
        return -x * (torch.exp(-0.5 * x**2) - torch.tanh(x))


class NMAFPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_NEURONS)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            for _ in range(HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.activation = NMAFActivation()

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class AdaptiveActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(HIDDEN_NEURONS, dtype=DTYPE))
        self.beta = nn.Parameter(torch.zeros(HIDDEN_NEURONS, dtype=DTYPE))

    def forward(self, x):
        X = self.alpha * x + self.beta
        return -X * (torch.exp(-0.5 * X**2) - torch.tanh(X))


class NMAFAATPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_NEURONS)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            for _ in range(HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.activation = AdaptiveActivation()

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.activation(self.input_layer(h))
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class RationalActivation(nn.Module):
    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.tensor(
            [0.0218, 0.5, 1.5957, 1.1915], dtype=DTYPE
        ))
        self.b = nn.Parameter(torch.tensor(
            [1.0, 0.0, 2.383], dtype=DTYPE
        ))

    def forward(self, x):
        p = ((self.a[0] * x + self.a[1]) * x + self.a[2]) * x + self.a[3]
        q = (self.b[0] * x + self.b[1]) * x + self.b[2]
        return p / q


class RationalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, HIDDEN_NEURONS)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(HIDDEN_NEURONS, HIDDEN_NEURONS)
            for _ in range(HIDDEN_LAYERS)
        ])
        self.output_layer = nn.Linear(HIDDEN_NEURONS, 1)
        self.rational_activations = nn.ModuleList([
            RationalActivation() for _ in range(HIDDEN_LAYERS + 1)
        ])

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.rational_activations[0](self.input_layer(h))
        for i, layer in enumerate(self.hidden_layers, start=1):
            h = self.rational_activations[i](layer(h))
        return self.output_layer(h)


class Sine(nn.Module):
    def __init__(self, omega_0=OMEGA_0):
        super().__init__()
        self.omega_0 = float(omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


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

    def forward(self, x, t):
        h = torch.cat([x, t], dim=1)
        h = self.sine_activations[0](self.input_layer(h))
        for i, layer in enumerate(self.hidden_layers, start=1):
            h = self.sine_activations[i](layer(h))
        return self.output_layer(h)


def build_model(method):
    if method == "tanh":
        return TanhPINN().double()
    if method == "nmaf":
        return NMAFPINN().double()
    if method == "nmaf_aat":
        return NMAFAATPINN().double()
    if method == "rational":
        return RationalPINN().double()
    if method == "siren":
        return SirenPINN().double()
    raise ValueError(method)


def load_state(method, target):
    p = checkpoint_path(method, target)
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    return ckpt, ckpt["model_state_dict"]


def load_reference():
    data = scipy.io.loadmat(REFERENCE_FILE)
    x = np.asarray(data["x"]).reshape(-1)
    t = np.asarray(data["tt"]).reshape(-1)
    exact = np.asarray(data["uu"], dtype=np.float64)

    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, exact


def predict_model(model, X, T):
    x = torch.tensor(X.reshape(-1, 1), dtype=DTYPE)
    t = torch.tensor(T.reshape(-1, 1), dtype=DTYPE)
    model.eval()
    with torch.no_grad():
        pred = model(x, t).reshape(X.shape).numpy()
    return pred


def relative_l2(exact, pred):
    e = np.asarray(exact, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(e - p, ord=2) / np.linalg.norm(e, ord=2))


def maximum_absolute_error(exact, pred):
    e = np.asarray(exact, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(e - p)))


def load_loss_history(method):
    p = history_path(method)
    if not p.exists():
        raise FileNotFoundError(f"Missing loss history: {p}")

    df = pd.read_csv(p)
    required = {"lbfgs_eval", "total_loss"}
    if not required.issubset(df.columns):
        raise ValueError(f"{p} lacks {required}. Columns={list(df.columns)}")
    df = df[["lbfgs_eval", "total_loss"]].copy()
    df["lbfgs_eval"] = pd.to_numeric(df["lbfgs_eval"], errors="coerce")
    df["total_loss"] = pd.to_numeric(df["total_loss"], errors="coerce")
    df = df.dropna().sort_values("lbfgs_eval")
    return df


def evaluate_checkpoint(method, target, X, T, exact):
    ckpt, state = load_state(method, target)
    model = build_model(method)


    model.load_state_dict(state, strict=True)
    pred = predict_model(model, X, T)
    l2 = relative_l2(exact, pred)
    max_abs = maximum_absolute_error(exact, pred)
    return {
        "method": method,
        "method_label": METHOD_LABEL[method],
        "seed": SEED,
        "checkpoint_eval": target,
        "loss": float(
            ckpt.get("latest_metrics", {}).get(
                "total_loss",
                np.nan
            )
        ),
        "relative_l2_error": l2,
        "maximum_absolute_error": max_abs,
    }


def make_loss_curve():
    active_methods = get_active_methods()
    rows = []
    for method, label in active_methods:
        d = load_loss_history(method)
        d = d[(d["lbfgs_eval"] >= 1) & (d["lbfgs_eval"] <= 20000)].copy()
        d["method"] = method
        d["method_label"] = label
        rows.append(d)

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(ANALYSIS_DIR / "loss_curves_all_methods_seed99.csv", index=False)

    configure_paper_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    for method, label in active_methods:
        d = all_df[all_df["method"] == method]
        ax.semilogy(
            d["lbfgs_eval"],
            d["total_loss"],
            linewidth=PLOT_LINEWIDTH,
            color=METHOD_COLOR[method],
            label=label,
        )

    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, numticks=20)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100,
        )
    )
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Total loss", fontsize=14)

    ax.legend(
        loc="upper right",
        fontsize=10,
        frameon=True,
    )

    ax.grid(
        True,
        which="major",
        linestyle="-",
        alpha=0.6,
    )
    ax.grid(
        True,
        which="minor",
        linestyle="--",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        ANALYSIS_DIR / "allen_cahn_broader_activation_total_loss_seed99.png",
        dpi=PLOT_DPI,
        format="png",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_l2_tables():
    active_methods = get_active_methods()
    _, _, X, T, exact = load_reference()

    rows = []
    for method, _ in active_methods:
        for target in CHECKPOINT_EVALS:
            row = evaluate_checkpoint(method, target, X, T, exact)

            d = load_loss_history(method)
            hit = d[d["lbfgs_eval"] == target]
            if not hit.empty:
                row["loss"] = float(hit.iloc[-1]["total_loss"])

            print(
                f"{row['method_label']:14s} | "
                f"eval={target:5d} | "
                f"relative L2={row['relative_l2_error']:.6e} | "
                f"max abs error={row['maximum_absolute_error']:.6e}"
            )

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS_DIR / "allen_cahn_broader_activation_individual_seed99.csv", index=False)

    summary = (
        df.groupby(["method", "method_label"], sort=False)
          .agg(
              loss_at_10000=("loss", lambda s: float(s.iloc[0])),
              loss_at_20000=("loss", lambda s: float(s.iloc[-1])),
              relative_l2_at_10000=("relative_l2_error", lambda s: float(s.iloc[0])),
              relative_l2_at_20000=("relative_l2_error", lambda s: float(s.iloc[-1])),
              max_absolute_error_at_10000=("maximum_absolute_error", lambda s: float(s.iloc[0])),
              max_absolute_error_at_20000=("maximum_absolute_error", lambda s: float(s.iloc[-1])),
          )
          .reset_index()
    )
    summary.to_csv(
        ANALYSIS_DIR / "allen_cahn_broader_activation_paper_ready_seed99.csv",
        index=False
    )

    return df, summary


def write_manifest():
    active_methods = get_active_methods()
    active_names = [m for m, _ in active_methods]
    skipped_names = [m for m, _ in METHODS if m not in active_names]

    manifest = {
        "experiment": "Broader activation-based method comparison — Allen-Cahn",
        "seed": SEED,
        "methods": active_names,
        "skipped_methods": skipped_names,
        "nmaf_optional": True,
        "new_training_methods": [
            m for m in ["rational", "siren"] if m in active_names
        ],
        "reused_methods": [
            m for m in ["tanh", "nmaf", "nmaf_aat"] if m in active_names
        ],
        "hidden_layers": HIDDEN_LAYERS,
        "hidden_neurons": HIDDEN_NEURONS,
        "n_res": 3000,
        "n_bc": 200,
        "n_ic": 500,
        "dtype": "float64",
        "optimizer": "single continuous LBFGS",
        "lbfgs_lr": 1.0,
        "lbfgs_max_iter": 20000,
        "lbfgs_max_eval": 20000,
        "lbfgs_history_size": 200,
        "checkpoints": list(CHECKPOINT_EVALS),
        "relative_l2_definition": "||reshape(u_exact)-reshape(u_pred)||_2 / ||reshape(u_exact)||_2",
        "rational": {
            "type": "(3,2)",
            "initial_P_descending_power": [0.0218, 0.5, 1.5957, 1.1915],
            "initial_Q_descending_power": [1.0, 0.0, 2.383],
            "parameter_sharing": "one coefficient set per activation location, shared across neurons",
        },
        "siren": {
            "omega0": 30.0,
            "first_layer_init": "U(-1/fan_in,+1/fan_in)",
            "subsequent_init": "U(-sqrt(6/fan_in)/omega0,+sqrt(6/fan_in)/omega0)",
            "output": "linear",
        },
        "fairness_note": (
            "Method-specific initialization is retained for Rational NN and SIREN "
            "because it is part of their original formulations; all other training "
            "conditions are held fixed."
        ),
    }
    (ANALYSIS_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    configure_paper_plot_style()

    active_now = get_active_methods()
    active_names = [m for m, _ in active_now]
    skipped_names = [m for m, _ in METHODS if m not in active_names]
    print("Active methods:", active_names)
    if skipped_names:
        print("Skipped methods:", skipped_names)

    write_manifest()
    make_loss_curve()
    df, summary = make_l2_tables()
    print("=" * 120)
    print("Allen-Cahn broader activation-based comparison completed.")
    print(summary.to_string(index=False))
    print("Output:", ANALYSIS_DIR)
