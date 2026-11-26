from __future__ import annotations

import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


PLOT_CONFIG = {
    "label_size": 16,
    "tick_size": 14,
    "legend_size": 14,
    "colorbar_size": 14,
    "font_family": "Times New Roman",
}

plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams.update({"font.size": 14})


IMAGE_DPI = 800

REPRESENTATIVE_SEED = 2026
FINAL_CHECKPOINT_NAME = "total0700000"


N_TEST = 200

HIDDEN_LAYERS = 4
HIDDEN_NEURONS = 128
OUTPUT_DIM = 1

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0

RAR_METHOD_DIR_NAME = "rar_adam_integrated_base6400_plus1280"
RAR_METHOD_DIR_OVERRIDE = None

METHOD_ORDER = [
    "tanh",
    "rar",
    "stan",
    "nmaf_aat",
]

METHOD_LABELS = {
    "tanh": "Standard PINN",
    "rar": "RAR-PINN",
    "stan": "Stan-PINN",
    "nmaf_aat": "NMAF-AAT",
}


SCRIPT_DIR = Path(__file__).resolve().parent

RESULT_ROOT = SCRIPT_DIR / "wave_multiseed_results"
ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"
PREDICTION_DIR = ANALYSIS_DIR / "predictions"
OUTPUT_DIR = ANALYSIS_DIR / "final_field_maps"

DEVICE = torch.device("cpu")
DTYPE = torch.float64


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
            OUTPUT_DIM,
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

        return self.output_layer(
            out
        )


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
            OUTPUT_DIM,
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

        return self.output_layer(
            out
        )


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
            + self.beta
            * z
            * tanh_z
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
            OUTPUT_DIM,
        )

        self.stan_activations = nn.ModuleList([
            StanActivation(
                hidden_neurons
            )
            for _ in range(
                hidden_layers + 1
            )
        ])

    def forward(self, x, t):
        inputs = torch.cat(
            [x, t],
            dim=1,
        )

        out = self.stan_activations[0](
            self.input_layer(
                inputs
            )
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

        return self.output_layer(
            out
        )


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

    return model.double().to(
        DEVICE
    )


def make_reference_grid():


    x = np.linspace(
        0.0,
        1.0,
        N_TEST,
        dtype=np.float64,
    )

    t = np.linspace(
        0.0,
        1.0,
        N_TEST,
        dtype=np.float64,
    )

    X, T = np.meshgrid(
        x,
        t,
        indexing="ij",
    )

    u_exact = (
        np.sin(
            2.0 * np.pi * X
        )
        * np.sin(
            2.0 * np.pi * T
        )
        + np.sin(
            10.0 * np.pi * X
        )
        * np.sin(
            10.0 * np.pi * T
        )
    )

    return (
        x,
        t,
        X,
        T,
        u_exact,
    )


def fixed_method_dir(method):
    path = (
        RESULT_ROOT
        / method
    )

    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing result directory: {path}"
        )

    return path


def _is_valid_rar_dir(path):
    if not path.is_dir():
        return False


    seed_dir = (
        path
        / f"seed_{REPRESENTATIVE_SEED}"
    )

    if not seed_dir.is_dir():
        return False

    matches = list(
        seed_dir.glob(
            f"wave_*_seed{REPRESENTATIVE_SEED}_{FINAL_CHECKPOINT_NAME}.pth"
        )
    )

    return any(
        "rar" in p.name.lower()
        for p in matches
    )


def locate_rar_method_dir():
    if (
        RAR_METHOD_DIR_OVERRIDE
        is not None
    ):
        path = (
            RESULT_ROOT
            / RAR_METHOD_DIR_OVERRIDE
        )

        if not _is_valid_rar_dir(
            path
        ):
            raise FileNotFoundError(
                "RAR_METHOD_DIR_OVERRIDE is not a valid Wave RAR directory: "
                f"{path}"
            )

        return path

    formal_path = (
        RESULT_ROOT
        / RAR_METHOD_DIR_NAME
    )

    if _is_valid_rar_dir(
        formal_path
    ):
        return formal_path

    candidates = [
        path
        for path in RESULT_ROOT.iterdir()
        if (
            path.is_dir()
            and "rar" in path.name.lower()
            and _is_valid_rar_dir(path)
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    candidate_text = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise RuntimeError(
        "Wave RAR result directory could not be identified uniquely.\n"
        "Set RAR_METHOD_DIR_OVERRIDE near the top of this script.\n"
        "Candidates:\n"
        f"{candidate_text if candidate_text else '  (none)'}"
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

        preferred = (
            seed_dir
            / (
                f"wave_{RAR_METHOD_DIR_NAME}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_{FINAL_CHECKPOINT_NAME}.pth"
            )
        )

        if preferred.exists():
            return preferred

        matches = sorted(
            seed_dir.glob(
                f"wave_*_seed{REPRESENTATIVE_SEED}_{FINAL_CHECKPOINT_NAME}.pth"
            )
        )

        rar_matches = [
            p
            for p in matches
            if "rar" in p.name.lower()
        ]

        if len(rar_matches) == 1:
            return rar_matches[0]

        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            "Could not uniquely locate final Wave RAR checkpoint.\n"
            f"Directory: {seed_dir}\n"
            f"Matches: {[p.name for p in matches]}"
        )

    seed_dir = (
        fixed_method_dir(
            method
        )
        / f"seed_{REPRESENTATIVE_SEED}"
    )

    preferred = (
        seed_dir
        / (
            f"wave_{method}"
            f"_seed{REPRESENTATIVE_SEED}"
            f"_{FINAL_CHECKPOINT_NAME}.pth"
        )
    )

    if preferred.exists():
        return preferred

    matches = sorted(
        seed_dir.glob(
            f"wave_*_seed{REPRESENTATIVE_SEED}_{FINAL_CHECKPOINT_NAME}.pth"
        )
    )

    method_matches = [
        p
        for p in matches
        if method in p.name.lower()
    ]

    if len(method_matches) == 1:
        return method_matches[0]

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Could not uniquely locate final Wave checkpoint for "
        f"{METHOD_LABELS[method]}.\n"
        f"Directory: {seed_dir}\n"
        f"Matches: {[p.name for p in matches]}"
    )


def prediction_npz_path(
    method,
):
    return (
        PREDICTION_DIR
        / (
            f"wave_{method}"
            f"_seed{REPRESENTATIVE_SEED}"
            f"_{FINAL_CHECKPOINT_NAME}"
            f"_prediction.npz"
        )
    )


def load_existing_prediction(
    method,
    x_ref,
    t_ref,
    u_exact_ref,
):
    path = prediction_npz_path(
        method
    )

    if not path.exists():
        return None

    try:
        data = np.load(
            path
        )
    except Exception:
        return None

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
        data["x"],
        dtype=np.float64,
    ).reshape(-1)

    t = np.asarray(
        data["t"],
        dtype=np.float64,
    ).reshape(-1)

    u_exact = np.asarray(
        data["u_exact"],
        dtype=np.float64,
    )

    u_pred = np.asarray(
        data["u_pred"],
        dtype=np.float64,
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

    if (
        u_pred.shape
        != u_exact_ref.shape
    ):
        return None

    print(
        "Using existing prediction NPZ for "
        f"{METHOD_LABELS[method]}: "
        f"{path.name}"
    )

    return u_pred


def _state_dict_to_double_cpu(
    state_dict,
):
    converted = {}

    for key, value in state_dict.items():
        if torch.is_tensor(
            value
        ):
            converted[
                key
            ] = value.to(
                dtype=DTYPE,
                device="cpu",
            )
        else:
            converted[
                key
            ] = value

    return converted


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

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "'model_state_dict' missing in "
            f"{checkpoint_path}"
        )

    model = build_model(
        method
    )

    model.load_state_dict(
        _state_dict_to_double_cpu(
            checkpoint[
                "model_state_dict"
            ]
        ),
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
    batch_size = 20000

    for start in range(
        0,
        x_flat.shape[0],
        batch_size,
    ):
        stop = min(
            start
            + batch_size,
            x_flat.shape[0],
        )

        x_tensor = torch.as_tensor(
            x_flat[
                start:stop
            ],
            dtype=DTYPE,
            device=DEVICE,
        )

        t_tensor = torch.as_tensor(
            t_flat[
                start:stop
            ],
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
        f"Rebuilt prediction for "
        f"{METHOD_LABELS[method]} from "
        f"{checkpoint_path.name}"
    )

    del model

    return u_pred


def save_png(
    fig,
    filename,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    print(
        f"Saved: {path.resolve()}"
    )

    return path


def plot_method_final_field(
    method,
    X,
    T,
    u_exact,
    u_pred,
    shared_solution_vmin,
    shared_solution_vmax,
    shared_error_vmax,
):


    abs_error = np.abs(
        u_exact
        - u_pred
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    titles = [
        "Reference solution",
        f"{METHOD_LABELS[method]} prediction",
        "Absolute error",
    ]

    fields = [
        u_exact,
        u_pred,
        abs_error,
    ]

    for i, (
        ax,
        title,
        field,
    ) in enumerate(
        zip(
            axes,
            titles,
            fields,
        )
    ):
        if i < 2:


            im = ax.pcolor(
                T,
                X,
                field,
                cmap="RdBu_r",
                shading="auto",
                vmin=shared_solution_vmin,
                vmax=shared_solution_vmax,
            )
        else:

            im = ax.pcolor(
                T,
                X,
                field,
                cmap="RdBu_r",
                shading="auto",
                vmin=0.0,
                vmax=shared_error_vmax,
            )

        ax.set_title(
            title,
            fontsize=PLOT_CONFIG[
                "label_size"
            ],
        )

        ax.set_xlabel(
            "$t$",
            fontsize=PLOT_CONFIG[
                "label_size"
            ],
        )

        ax.set_ylabel(
            "$x$",
            fontsize=PLOT_CONFIG[
                "label_size"
            ],
        )

        ax.tick_params(
            labelsize=PLOT_CONFIG[
                "tick_size"
            ]
        )

        cbar = plt.colorbar(
            im,
            ax=ax,
        )

        cbar.ax.tick_params(
            labelsize=PLOT_CONFIG[
                "colorbar_size"
            ]
        )

    fig.tight_layout()

    filename = (
        f"wave_{method}"
        f"_final_field_map"
        f"_seed{REPRESENTATIVE_SEED}"
        f"_{FINAL_CHECKPOINT_NAME}.png"
    )

    path = save_png(
        fig,
        filename,
    )

    print(
        "-" * 72
    )
    print(
        f"Method: {METHOD_LABELS[method]}"
    )
    print(
        f"Maximum absolute error: "
        f"{float(abs_error.max()):.6e}"
    )
    print(
        "Reference/Prediction colorbar: "
        f"[{shared_solution_vmin:.6e}, "
        f"{shared_solution_vmax:.6e}]"
    )
    print(
        "Absolute-error colorbar: "
        f"[0.000000e+00, "
        f"{shared_error_vmax:.6e}]"
    )
    print(
        "-" * 72
    )

    plt.show()

    plt.close(
        fig
    )

    return path


def main():
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            "Missing Wave result root: "
            f"{RESULT_ROOT}"
        )

    (
        x,
        t,
        X,
        T,
        u_exact,
    ) = make_reference_grid()

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


    shared_solution_vmin = float(
        np.min(
            u_exact
        )
    )

    shared_solution_vmax = float(
        np.max(
            u_exact
        )
    )


    method_max_errors = {
        method: float(
            np.max(
                np.abs(
                    u_exact
                    - predictions[
                        method
                    ]
                )
            )
        )
        for method
        in METHOD_ORDER
    }

    shared_error_vmax = max(
        method_max_errors.values()
    )

    print(
        "=" * 110
    )
    print(
        "Wave final four-method physical-field contour plotting"
    )
    print(
        "=" * 110
    )
    print(
        f"Device                 : {DEVICE}"
    )
    print(
        f"Seed                   : {REPRESENTATIVE_SEED}"
    )
    print(
        f"Final checkpoint       : {FINAL_CHECKPOINT_NAME}"
    )
    print(
        f"Evaluation grid        : {N_TEST} x {N_TEST}"
    )
    print(
        f"RAR directory          : {rar_dir}"
    )
    print(
        f"PNG dpi                : {IMAGE_DPI}"
    )
    print(
        "Reference/Prediction   : "
        f"[{shared_solution_vmin:.6e}, "
        f"{shared_solution_vmax:.6e}]"
    )
    print(
        "Shared error colorbar  : "
        f"[0.000000e+00, "
        f"{shared_error_vmax:.6e}]"
    )
    print(
        "-" * 110
    )

    for method in METHOD_ORDER:
        print(
            f"{METHOD_LABELS[method]:16s} "
            f"max absolute error = "
            f"{method_max_errors[method]:.6e}"
        )

    print(
        "=" * 110
    )

    output_paths = []

    for method in METHOD_ORDER:
        output_paths.append(
            plot_method_final_field(
                method,
                X,
                T,
                u_exact,
                predictions[
                    method
                ],
                shared_solution_vmin,
                shared_solution_vmax,
                shared_error_vmax,
            )
        )

    print(
        "\nGenerated lossless PNG files:"
    )

    for path in output_paths:
        print(
            f"  {path.resolve()}"
        )

    print(
        "\nCompleted."
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
