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
    "colorbar_size": 14,
    "font_family": "Times New Roman",
}

plt.rcParams["font.family"] = PLOT_CONFIG["font_family"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams.update({"font.size": 14})


IMAGE_DPI = 800

REPRESENTATIVE_SEED = 2026
FINAL_EVAL = 10000

REQUESTED_TIMES = (0.59, 0.79, 0.98)

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 100
OUTPUT_DIM = 2

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0


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

RESULT_ROOT = SCRIPT_DIR / "nlse_multiseed_results"
REFERENCE_FILE = SCRIPT_DIR / "NLS.mat"
ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"
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
        X = self.alpha * z + self.beta

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
            OUTPUT_DIM,
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

    data = scipy.io.loadmat(
        REFERENCE_FILE
    )

    if "x" not in data:
        raise KeyError(
            "NLS.mat must contain variable 'x'."
        )

    if "tt" in data:
        time_key = "tt"
    elif "t" in data:
        time_key = "t"
    else:
        raise KeyError(
            "NLS.mat must contain time variable 'tt' or 't'."
        )

    if "uu" not in data:
        raise KeyError(
            "NLS.mat must contain complex reference field 'uu'."
        )

    x = np.asarray(
        data["x"],
        dtype=np.float64,
    ).reshape(-1)

    t = np.asarray(
        data[time_key],
        dtype=np.float64,
    ).reshape(-1)

    h_exact_complex = np.asarray(
        data["uu"]
    )

    target_shape = (
        x.size,
        t.size,
    )

    if h_exact_complex.shape != target_shape:
        if h_exact_complex.T.shape == target_shape:
            h_exact_complex = h_exact_complex.T
        else:
            raise ValueError(
                f"Unexpected uu shape {h_exact_complex.shape}; "
                f"expected {target_shape} or its transpose."
            )

    h_exact_magnitude = np.abs(
        h_exact_complex
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
        h_exact_complex,
        h_exact_magnitude,
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

    return all(
        (
            path
            / f"seed_{seed}"
            / "fixed_monitor_history.csv"
        ).exists()
        for seed in (
            99,
            1234,
            2026,
        )
    )


def locate_rar_method_dir():
    if RAR_METHOD_DIR_OVERRIDE is not None:
        path = (
            RESULT_ROOT
            / RAR_METHOD_DIR_OVERRIDE
        )

        if not _is_valid_rar_dir(
            path
        ):
            raise FileNotFoundError(
                "RAR_METHOD_DIR_OVERRIDE is not a valid "
                f"RAR result directory: {path}"
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

    if len(
        rar_named
    ) == 1:
        return rar_named[0]

    if len(
        candidates
    ) == 1:
        return candidates[0]

    candidate_text = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise RuntimeError(
        "RAR result directory could not be identified uniquely.\n"
        "Set RAR_METHOD_DIR_OVERRIDE near the top of this script.\n"
        "Candidates:\n"
        f"{candidate_text if candidate_text else '  (none)'}"
    )


def exact_eval_from_filename(
    path,
):


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
            fixed_method_dir(
                method
            )
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
                f"nlse_{rar_dir.name}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pth"
            ),
            seed_dir
            / (
                f"nlse_{rar_dir.name}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pt"
            ),
        ])

    else:
        preferred.extend([
            seed_dir
            / (
                f"nlse_{method}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pth"
            ),
            seed_dir
            / (
                f"nlse_{method}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_eval{FINAL_EVAL:05d}.pt"
            ),
        ])

    for path in preferred:
        if path.exists():
            return path


    candidates = sorted(
        list(
            seed_dir.glob(
                "*.pth"
            )
        )
        + list(
            seed_dir.glob(
                "*.pt"
            )
        )
    )

    exact_matches = [
        path
        for path in candidates
        if exact_eval_from_filename(
            path
        ) == FINAL_EVAL
    ]

    if len(
        exact_matches
    ) == 1:
        return exact_matches[0]

    if len(
        exact_matches
    ) > 1:
        method_token = (
            "rar"
            if method == "rar"
            else method
        )

        method_named = [
            path
            for path in exact_matches
            if method_token
            in path.name.lower()
        ]

        if len(
            method_named
        ) == 1:
            return method_named[0]

    matches_text = "\n".join(
        f"  - {path.name}"
        for path in exact_matches
    )

    raise FileNotFoundError(
        "Could not uniquely locate final checkpoint for "
        f"{METHOD_LABELS[method]}, "
        f"seed={REPRESENTATIVE_SEED}, "
        f"eval={FINAL_EVAL}.\n"
        f"Seed directory: {seed_dir}\n"
        "Exact eval-token matches:\n"
        f"{matches_text if matches_text else '  (none)'}"
    )


def prediction_npz_candidates(
    method,
):


    return [
        ANALYSIS_DIR
        / (
            f"prediction_{method}"
            f"_seed{REPRESENTATIVE_SEED}"
            f"_eval{FINAL_EVAL:05d}.npz"
        ),

        fixed_method_dir(method)
        / f"seed_{REPRESENTATIVE_SEED}"
        / (
            f"nlse_{method}"
            f"_seed{REPRESENTATIVE_SEED}"
            f"_prediction_eval{FINAL_EVAL:05d}.npz"
        )
        if method != "rar"
        else Path("__RAR_PLACEHOLDER__"),
    ]


def _extract_prediction_from_npz(
    path,
    x_ref,
    t_ref,
    h_exact_complex_ref,
):
    data = np.load(
        path
    )

    if "x" not in data.files:
        return None

    if (
        "t" not in data.files
        and "tt" not in data.files
    ):
        return None

    x = np.asarray(
        data["x"]
    ).reshape(-1)

    t_key = (
        "t"
        if "t" in data.files
        else "tt"
    )

    t = np.asarray(
        data[t_key]
    ).reshape(-1)

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


    exact_key = None
    for key in (
        "h_exact",
        "u_exact",
        "exact",
    ):
        if key in data.files:
            exact_key = key
            break

    if exact_key is not None:
        stored_exact = np.asarray(
            data[exact_key]
        )


        if (
            stored_exact.shape
            == h_exact_complex_ref.shape
            and np.iscomplexobj(
                stored_exact
            )
        ):
            if not np.array_equal(
                stored_exact,
                h_exact_complex_ref,
            ):
                return None


    for key in (
        "h_pred",
        "prediction",
    ):
        if key in data.files:
            h_pred = np.asarray(
                data[key]
            )

            if h_pred.shape != h_exact_complex_ref.shape:
                return None

            return h_pred


    if (
        "u_pred" in data.files
        and "v_pred" in data.files
    ):
        u_pred = np.asarray(
            data["u_pred"]
        )

        v_pred = np.asarray(
            data["v_pred"]
        )

        if (
            u_pred.shape
            != h_exact_complex_ref.shape
            or v_pred.shape
            != h_exact_complex_ref.shape
        ):
            return None

        return (
            u_pred
            + 1j * v_pred
        )

    return None


def load_existing_prediction(
    method,
    rar_dir,
    x_ref,
    t_ref,
    h_exact_complex_ref,
):
    candidates = []


    candidates.append(
        ANALYSIS_DIR
        / (
            f"prediction_{method}"
            f"_seed{REPRESENTATIVE_SEED}"
            f"_eval{FINAL_EVAL:05d}.npz"
        )
    )


    if method == "rar":
        seed_dir = (
            rar_dir
            / f"seed_{REPRESENTATIVE_SEED}"
        )
        candidates.extend(
            sorted(
                seed_dir.glob(
                    f"*prediction*eval{FINAL_EVAL:05d}*.npz"
                )
            )
        )
    else:
        seed_dir = (
            fixed_method_dir(
                method
            )
            / f"seed_{REPRESENTATIVE_SEED}"
        )
        candidates.extend([
            seed_dir
            / (
                f"nlse_{method}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_prediction_eval{FINAL_EVAL:05d}.npz"
            )
        ])
        candidates.extend(
            sorted(
                seed_dir.glob(
                    f"*prediction*eval{FINAL_EVAL:05d}*.npz"
                )
            )
        )

    seen = set()

    for path in candidates:
        if (
            str(path)
            == "__RAR_PLACEHOLDER__"
            or not path.exists()
        ):
            continue

        key = str(
            path.resolve()
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        h_pred = _extract_prediction_from_npz(
            path,
            x_ref,
            t_ref,
            h_exact_complex_ref,
        )

        if h_pred is not None:
            print(
                f"Using existing prediction for "
                f"{METHOD_LABELS[method]}: "
                f"{path}"
            )

            return h_pred

    return None


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

    state_dict = {}

    for key, value in checkpoint[
        "model_state_dict"
    ].items():
        if torch.is_tensor(
            value
        ):
            state_dict[
                key
            ] = value.to(
                dtype=DTYPE,
                device="cpu",
            )
        else:
            state_dict[
                key
            ] = value

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

    prediction = np.vstack(
        outputs
    )

    u_pred = prediction[
        :,
        0,
    ].reshape(
        X.shape
    )

    v_pred = prediction[
        :,
        1,
    ].reshape(
        X.shape
    )

    h_pred = (
        u_pred
        + 1j * v_pred
    )

    print(
        f"Rebuilt prediction for "
        f"{METHOD_LABELS[method]} from "
        f"{checkpoint_path.name}"
    )

    return h_pred


def load_all_final_predictions():
    (
        x,
        t,
        X,
        T,
        h_exact_complex,
        h_exact_magnitude,
    ) = load_reference()

    rar_dir = locate_rar_method_dir()

    predictions = {}

    for method in METHOD_ORDER:
        h_pred_complex = (
            load_existing_prediction(
                method,
                rar_dir,
                x,
                t,
                h_exact_complex,
            )
        )

        if h_pred_complex is None:
            h_pred_complex = predict_from_checkpoint(
                method,
                rar_dir,
                X,
                T,
            )

        predictions[
            method
        ] = np.abs(
            h_pred_complex
        )

    return (
        x,
        t,
        h_exact_magnitude,
        predictions,
        rar_dir,
    )


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
    h_exact_mag,
    h_pred_mag,
    shared_solution_vmin,
    shared_solution_vmax,
    shared_error_vmax,
):


    abs_error = np.abs(
        h_exact_mag
        - h_pred_mag
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

    datas = [
        h_exact_mag,
        h_pred_mag,
        abs_error,
    ]


    discrete_cmap = plt.get_cmap(
        "RdBu_r",
        50,
    )

    for i, (
        ax,
        title,
        data,
    ) in enumerate(
        zip(
            axes,
            titles,
            datas,
        )
    ):
        if i < 2:


            im = ax.pcolor(
                T,
                X,
                data,
                cmap=discrete_cmap,
                shading="auto",
                vmin=shared_solution_vmin,
                vmax=shared_solution_vmax,
            )
        else:

            im = ax.pcolor(
                T,
                X,
                data,
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
        f"nlse_{method}"
        f"_final_field_map"
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
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            f"Missing NLSE result root: "
            f"{RESULT_ROOT}"
        )

    (
        x,
        t,
        X,
        T,
        h_exact_complex,
        h_exact_mag,
    ) = load_reference()

    rar_dir = locate_rar_method_dir()

    predictions_mag = {}

    for method in METHOD_ORDER:
        h_pred_complex = load_existing_prediction(
            method,
            rar_dir,
            x,
            t,
            h_exact_complex,
        )

        if h_pred_complex is None:
            h_pred_complex = predict_from_checkpoint(
                method,
                rar_dir,
                X,
                T,
            )

        predictions_mag[
            method
        ] = np.abs(
            h_pred_complex
        )


    shared_solution_vmin = float(
        np.min(
            h_exact_mag
        )
    )

    shared_solution_vmax = float(
        np.max(
            h_exact_mag
        )
    )


    method_max_errors = {
        method: float(
            np.max(
                np.abs(
                    h_exact_mag
                    - predictions_mag[
                        method
                    ]
                )
            )
        )
        for method in METHOD_ORDER
    }

    shared_error_vmax = max(
        method_max_errors.values()
    )

    print(
        "=" * 110
    )
    print(
        "NLSE final four-method physical-field contour plotting"
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
        f"Final checkpoint       : {FINAL_EVAL}"
    )
    print(
        f"Reference file         : {REFERENCE_FILE}"
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
            f"max magnitude error = "
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
                h_exact_mag,
                predictions_mag[
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
