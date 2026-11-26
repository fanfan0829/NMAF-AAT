from __future__ import annotations

import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn


PLOT_CONFIG = {
    "label_size": 16,
    "tick_size": 14,
    "legend_size": 14,
    "font_family": "Times New Roman",
}

IMAGE_DPI = 800

REPRESENTATIVE_SEED = 2026
FINAL_EVAL = 5000
REQUESTED_TIMES = (0.25, 0.50, 0.75)

HIDDEN_LAYERS = 4
HIDDEN_NEURONS = 64

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0


RAR_METHOD_DIR_OVERRIDE = None

METHOD_ORDER = ["tanh", "rar", "stan", "nmaf_aat"]

METHOD_LABELS = {
    "tanh": "Standard PINN",
    "rar": "RAR-PINN",
    "stan": "Stan-PINN",
    "nmaf_aat": "NMAF-AAT",
}


COMBINED_LINESTYLES = {
    "tanh": "-",
    "rar": "--",
    "stan": "-.",
    "nmaf_aat": ":",
}


SCRIPT_DIR = Path(__file__).resolve().parent

RESULT_ROOT = SCRIPT_DIR / "burgers_multiseed_results"
REFERENCE_FILE = SCRIPT_DIR / "burgers_shock.mat"
ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"
OUTPUT_DIR = ANALYSIS_DIR / "final_profile_lines"

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def configure_plot_style():
    plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": PLOT_CONFIG["label_size"],
        "axes.titlesize": PLOT_CONFIG["label_size"],
        "xtick.labelsize": PLOT_CONFIG["tick_size"],
        "ytick.labelsize": PLOT_CONFIG["tick_size"],
        "legend.fontsize": PLOT_CONFIG["legend_size"],
        "path.simplify": False,
    })


def save_png(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / filename
    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )
    print(f"Saved: {path.resolve()}")
    return path


class TanhPINN(nn.Module):
    def __init__(
        self,
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
    ):
        super().__init__()

        self.input_layer = nn.Linear(
            2,
            hidden_neurons,
        )

        self.hidden_layers = nn.ModuleList([
            nn.Linear(
                hidden_neurons,
                hidden_neurons,
            )
            for _ in range(hidden_layers)
        ])

        self.output_layer = nn.Linear(
            hidden_neurons,
            1,
        )

        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat(
            [x, t],
            dim=1,
        )

        out = self.activation(
            self.input_layer(inputs)
        )

        for layer in self.hidden_layers:
            out = self.activation(
                layer(out)
            )

        return self.output_layer(out)


class AdaptiveActivation(nn.Module):


    def __init__(
        self,
        in_features,
        initial_alpha=INITIAL_ALPHA,
        initial_beta=INITIAL_BETA,
    ):
        super().__init__()

        self.alpha = nn.Parameter(
            torch.ones(
                in_features,
                dtype=DTYPE,
            ) * initial_alpha
        )

        self.beta = nn.Parameter(
            torch.ones(
                in_features,
                dtype=DTYPE,
            ) * initial_beta
        )

    def forward(self, z):
        X = (
            self.alpha * z
            + self.beta
        )

        return -X * (
            torch.exp(
                -0.5 * X ** 2
            )
            - torch.tanh(X)
        )


class NMAFAATPINN(nn.Module):
    def __init__(
        self,
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
    ):
        super().__init__()

        self.input_layer = nn.Linear(
            2,
            hidden_neurons,
        )

        self.hidden_layers = nn.ModuleList([
            nn.Linear(
                hidden_neurons,
                hidden_neurons,
            )
            for _ in range(hidden_layers)
        ])

        self.output_layer = nn.Linear(
            hidden_neurons,
            1,
        )


        self.activation = AdaptiveActivation(
            hidden_neurons
        )

    def forward(self, x, t):
        inputs = torch.cat(
            [x, t],
            dim=1,
        )

        out = self.activation(
            self.input_layer(inputs)
        )

        for layer in self.hidden_layers:
            out = self.activation(
                layer(out)
            )

        return self.output_layer(out)


class StanActivation(nn.Module):


    def __init__(
        self,
        in_features,
        initial_beta=STAN_INITIAL_BETA,
    ):
        super().__init__()

        self.beta = nn.Parameter(
            torch.ones(
                in_features,
                dtype=DTYPE,
            ) * initial_beta
        )

    def forward(self, z):
        tanh_z = torch.tanh(z)

        return (
            tanh_z
            + self.beta * z * tanh_z
        )


class StanPINN(nn.Module):
    def __init__(
        self,
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
    ):
        super().__init__()

        self.input_layer = nn.Linear(
            2,
            hidden_neurons,
        )

        self.hidden_layers = nn.ModuleList([
            nn.Linear(
                hidden_neurons,
                hidden_neurons,
            )
            for _ in range(hidden_layers)
        ])

        self.output_layer = nn.Linear(
            hidden_neurons,
            1,
        )


        self.stan_activations = nn.ModuleList([
            StanActivation(
                hidden_neurons
            )
            for _ in range(hidden_layers + 1)
        ])

    def forward(self, x, t):
        inputs = torch.cat(
            [x, t],
            dim=1,
        )

        out = self.stan_activations[0](
            self.input_layer(inputs)
        )

        for activation_index, layer in enumerate(
            self.hidden_layers,
            start=1,
        ):
            out = self.stan_activations[
                activation_index
            ](
                layer(out)
            )

        return self.output_layer(out)


def build_model(method):
    if method == "tanh":
        model = TanhPINN()

    elif method == "rar":

        model = TanhPINN()

    elif method == "stan":
        model = StanPINN()

    elif method == "nmaf_aat":
        model = NMAFAATPINN()

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    return model.double().to(DEVICE)


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE}"
        )

    try:
        data = scipy.io.loadmat(
            REFERENCE_FILE
        )

        x = np.asarray(
            data["x"],
            dtype=np.float64,
        ).reshape(-1)

        t = np.asarray(
            data["t"],
            dtype=np.float64,
        ).reshape(-1)

        u_exact = np.asarray(
            data["usol"],
            dtype=np.float64,
        )

    except NotImplementedError:
        import h5py

        with h5py.File(
            REFERENCE_FILE,
            "r",
        ) as f:
            x = np.asarray(
                f["x"],
                dtype=np.float64,
            ).reshape(-1)

            t = np.asarray(
                f["t"],
                dtype=np.float64,
            ).reshape(-1)

            u_exact = np.asarray(
                f["usol"],
                dtype=np.float64,
            ).T

    target_shape = (
        x.size,
        t.size,
    )

    if u_exact.shape != target_shape:
        if u_exact.T.shape == target_shape:
            u_exact = u_exact.T
        else:
            raise ValueError(
                f"Unexpected usol shape {u_exact.shape}; "
                f"expected {target_shape} or its transpose."
            )

    X, T = np.meshgrid(
        x,
        t,
        indexing="ij",
    )

    return (
        x,
        t,
        X,
        T,
        u_exact,
    )


def fixed_method_dir(method):
    path = RESULT_ROOT / method

    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing result directory: {path}"
        )

    return path


def _is_valid_rar_dir(path):
    if not path.is_dir():
        return False


    for seed in (99, 1234, 2026):
        if not (
            path
            / f"seed_{seed}"
            / "fixed_monitor_history.csv"
        ).exists():
            return False

    return True


def locate_rar_method_dir():
    if RAR_METHOD_DIR_OVERRIDE is not None:
        path = (
            RESULT_ROOT
            / RAR_METHOD_DIR_OVERRIDE
        )

        if not _is_valid_rar_dir(path):
            raise FileNotFoundError(
                "RAR_METHOD_DIR_OVERRIDE is not a valid "
                f"formal RAR directory: {path}"
            )

        return path

    excluded = {
        "tanh",
        "nmaf_aat",
        "stan",
        "fixed_training_points",
        "four_method_unified_analysis",
    }

    candidates = [
        path
        for path in RESULT_ROOT.iterdir()
        if (
            path.name not in excluded
            and _is_valid_rar_dir(path)
        )
    ]

    rar_named = [
        path
        for path in candidates
        if "rar" in path.name.lower()
    ]

    if len(rar_named) == 1:
        return rar_named[0]

    if len(candidates) == 1:
        return candidates[0]

    candidate_text = "\n".join(
        f"  - {p}"
        for p in candidates
    )

    raise RuntimeError(
        "RAR result directory could not be identified uniquely.\n"
        "Set RAR_METHOD_DIR_OVERRIDE near the top of this script.\n"
        f"Candidates:\n"
        f"{candidate_text if candidate_text else '  (none)'}"
    )


def exact_eval_from_filename(path):


    match = re.search(
        r"eval_?0*(\d+)(?!\d)",
        path.name,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def find_final_checkpoint(
    method,
    rar_dir,
):
    if method == "rar":
        seed_dir = (
            rar_dir
            / f"seed_{REPRESENTATIVE_SEED}"
        )
    else:
        seed_dir = (
            fixed_method_dir(method)
            / f"seed_{REPRESENTATIVE_SEED}"
        )

    if not seed_dir.is_dir():
        raise FileNotFoundError(
            f"Missing seed directory: {seed_dir}"
        )


    preferred = []

    if method == "rar":
        preferred.extend([
            seed_dir
            / (
                f"burgers_{rar_dir.name}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pth"
            ),
            seed_dir
            / (
                f"burgers_{rar_dir.name}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pt"
            ),
        ])
    else:
        preferred.extend([
            seed_dir
            / (
                f"burgers_{method}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pth"
            ),
            seed_dir
            / (
                f"burgers_{method}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pt"
            ),
        ])

    for path in preferred:
        if path.exists():
            return path


    candidates = sorted(
        list(seed_dir.glob("*.pth"))
        + list(seed_dir.glob("*.pt"))
    )

    exact_matches = [
        path
        for path in candidates
        if exact_eval_from_filename(path)
        == FINAL_EVAL
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        method_token = (
            "rar"
            if method == "rar"
            else method
        )

        named = [
            path
            for path in exact_matches
            if method_token
            in path.name.lower()
        ]

        if len(named) == 1:
            return named[0]

    text = "\n".join(
        f"  - {p.name}"
        for p in exact_matches
    )

    raise FileNotFoundError(
        f"Could not uniquely locate final checkpoint for "
        f"{METHOD_LABELS[method]}, seed={REPRESENTATIVE_SEED}, "
        f"eval={FINAL_EVAL}.\n"
        f"Seed directory: {seed_dir}\n"
        f"Exact eval-token matches:\n"
        f"{text if text else '  (none)'}"
    )


def prediction_npz_path(method):
    return (
        ANALYSIS_DIR
        / (
            f"prediction_{method}"
            f"_seed{REPRESENTATIVE_SEED}"
            f"_eval{FINAL_EVAL:05d}.npz"
        )
    )


def load_existing_prediction(
    method,
    x_ref,
    t_ref,
    u_exact_ref,
):
    path = prediction_npz_path(method)

    if not path.exists():
        return None

    data = np.load(
        path
    )

    required = {
        "x",
        "t",
        "u_exact",
        "u_pred",
    }

    if not required.issubset(
        data.files
    ):
        return None

    x = np.asarray(
        data["x"]
    ).reshape(-1)

    t = np.asarray(
        data["t"]
    ).reshape(-1)

    u_exact = np.asarray(
        data["u_exact"]
    )

    u_pred = np.asarray(
        data["u_pred"]
    )

    if not np.array_equal(
        x,
        x_ref,
    ):
        return None

    if not np.array_equal(
        t,
        t_ref,
    ):
        return None

    if not np.array_equal(
        u_exact,
        u_exact_ref,
    ):
        return None

    if u_pred.shape != u_exact_ref.shape:
        return None

    print(
        f"Using existing prediction NPZ for "
        f"{METHOD_LABELS[method]}: {path.name}"
    )

    return u_pred.astype(
        np.float64,
        copy=False,
    )


@torch.no_grad()
def predict_from_checkpoint(
    method,
    rar_dir,
    X,
    T,
):
    checkpoint_path = find_final_checkpoint(
        method,
        rar_dir,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"'model_state_dict' missing in {checkpoint_path}"
        )

    model = build_model(
        method
    )

    state_dict = {}

    for key, value in checkpoint[
        "model_state_dict"
    ].items():
        if torch.is_tensor(value):
            state_dict[key] = value.to(
                dtype=DTYPE,
                device="cpu",
            )
        else:
            state_dict[key] = value

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    x_flat = X.reshape(
        -1,
        1,
    )

    t_flat = T.reshape(
        -1,
        1,
    )

    outputs = []
    batch_size = 65536

    for start in range(
        0,
        x_flat.shape[0],
        batch_size,
    ):
        stop = min(
            start + batch_size,
            x_flat.shape[0],
        )

        x_tensor = torch.as_tensor(
            x_flat[start:stop],
            dtype=DTYPE,
            device=DEVICE,
        )

        t_tensor = torch.as_tensor(
            t_flat[start:stop],
            dtype=DTYPE,
            device=DEVICE,
        )

        pred = model(
            x_tensor,
            t_tensor,
        )

        outputs.append(
            pred.cpu().numpy()
        )

    u_pred = np.vstack(
        outputs
    ).reshape(
        X.shape
    )

    print(
        f"Rebuilt prediction for {METHOD_LABELS[method]} "
        f"from checkpoint: {checkpoint_path.name}"
    )

    return u_pred


def load_all_final_predictions():
    (
        x,
        t,
        X,
        T,
        u_exact,
    ) = load_reference()

    rar_dir = locate_rar_method_dir()

    predictions = {}

    for method in METHOD_ORDER:
        u_pred = load_existing_prediction(
            method,
            x,
            t,
            u_exact,
        )

        if u_pred is None:
            u_pred = predict_from_checkpoint(
                method,
                rar_dir,
                X,
                T,
            )

        predictions[
            method
        ] = u_pred

    return (
        x,
        t,
        u_exact,
        predictions,
        rar_dir,
    )


def nearest_time_indices(
    t,
    requested_times=REQUESTED_TIMES,
):
    return [
        int(
            np.argmin(
                np.abs(
                    t - requested_time
                )
            )
        )
        for requested_time in requested_times
    ]


def style_profile_axis(
    ax,
    title,
):
    ax.set_title(
        title,
        fontsize=PLOT_CONFIG[
            "label_size"
        ],
    )

    ax.set_xlabel(
        "$x$",
        fontsize=PLOT_CONFIG[
            "label_size"
        ],
    )

    ax.set_ylabel(
        "$u(t,x)$",
        fontsize=PLOT_CONFIG[
            "label_size"
        ],
    )

    ax.set_xlim(
        [-1.0, 1.0]
    )

    ax.set_ylim(
        [-1.2, 1.2]
    )

    ax.tick_params(
        labelsize=PLOT_CONFIG[
            "tick_size"
        ]
    )


def plot_single_method_final(
    method,
    x,
    t,
    u_exact,
    u_pred,
):
    t_indices = nearest_time_indices(
        t
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4.4),
        sharey=True,
    )

    for i, t_idx in enumerate(
        t_indices
    ):
        actual_t = float(
            t[t_idx]
        )

        ax = axes[i]


        ax.plot(
            x,
            u_exact[:, t_idx],
            "b-",
            linewidth=2.0,
            label="Reference",
        )

        ax.plot(
            x,
            u_pred[:, t_idx],
            "r--",
            linewidth=2.0,
            label=METHOD_LABELS[
                method
            ],
        )

        style_profile_axis(
            ax,
            f"$t = {actual_t:.2f}$",
        )

        if i == 1:
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(
                    0.5,
                    -0.30,
                ),
                ncol=2,
                frameon=False,
                fontsize=PLOT_CONFIG[
                    "legend_size"
                ],
            )

    fig.tight_layout(
        h_pad=3.0
    )

    filename = (
        f"burgers_{method}"
        f"_final_profiles"
        f"_seed{REPRESENTATIVE_SEED}"
        f"_eval{FINAL_EVAL:05d}.png"
    )

    path = save_png(
        fig,
        filename,
    )

    plt.show()
    plt.close(
        fig
    )

    return path


def plot_combined_final(
    x,
    t,
    u_exact,
    predictions,
):
    t_indices = nearest_time_indices(
        t
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        sharey=True,
    )


    method_colors = {
        "tanh": "tab:blue",
        "rar": "tab:orange",
        "stan": "tab:green",
        "nmaf_aat": "tab:red",
    }

    for i, t_idx in enumerate(
        t_indices
    ):
        actual_t = float(
            t[t_idx]
        )

        ax = axes[i]

        ax.plot(
            x,
            u_exact[:, t_idx],
            color="black",
            linestyle="-",
            linewidth=2.2,
            label="Reference",
        )

        for method in METHOD_ORDER:
            ax.plot(
                x,
                predictions[
                    method
                ][:, t_idx],
                color=method_colors[
                    method
                ],
                linestyle=COMBINED_LINESTYLES[
                    method
                ],
                linewidth=1.8,
                label=METHOD_LABELS[
                    method
                ],
            )

        style_profile_axis(
            ax,
            f"$t = {actual_t:.2f}$",
        )

    handles, labels = axes[
        1
    ].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.02,
        ),
        ncol=5,
        frameon=False,
        fontsize=PLOT_CONFIG[
            "legend_size"
        ],
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.10,
            1.0,
            1.0,
        ]
    )

    filename = (
        "burgers_four_method_final_profiles"
        f"_seed{REPRESENTATIVE_SEED}"
        f"_eval{FINAL_EVAL:05d}.png"
    )

    path = save_png(
        fig,
        filename,
    )

    plt.show()
    plt.close(
        fig
    )

    return path


def main():
    configure_plot_style()

    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            f"Missing Burgers result root: {RESULT_ROOT}"
        )

    (
        x,
        t,
        u_exact,
        predictions,
        rar_dir,
    ) = load_all_final_predictions()

    print("=" * 100)
    print("Burgers final four-method profile plotting")
    print("=" * 100)
    print(f"Device             : {DEVICE}")
    print(f"Seed               : {REPRESENTATIVE_SEED}")
    print(f"Final checkpoint   : {FINAL_EVAL}")
    print(f"Profile times      : {REQUESTED_TIMES}")
    print(f"Reference file     : {REFERENCE_FILE}")
    print(f"RAR directory      : {rar_dir}")
    print(f"PNG dpi            : {IMAGE_DPI}")
    print("=" * 100)

    output_paths = []


    for method in METHOD_ORDER:
        output_paths.append(
            plot_single_method_final(
                method,
                x,
                t,
                u_exact,
                predictions[
                    method
                ],
            )
        )


    output_paths.append(
        plot_combined_final(
            x,
            t,
            u_exact,
            predictions,
        )
    )

    print("\nGenerated lossless PNG files:")
    for path in output_paths:
        print(f"  {path.resolve()}")

    print("\nCompleted.")
    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
