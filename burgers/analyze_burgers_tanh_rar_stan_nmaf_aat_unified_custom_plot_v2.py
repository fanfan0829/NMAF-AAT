import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn


TOTAL_LOSS_YLIM = (5.0e-7, 1.0e2)


LOSS_FIG_HEIGHT = 5.0
LOSS_FIG_WIDTH = 7.0


FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TITLE_SIZE = 13
LEGEND_SIZE = 11
TICK_LABEL_SIZE = 12


SEEDS = [99, 1234, 2026]
REPRESENTATIVE_SEED = 99
CHECKPOINT_EVALS = (2500, 5000)

RESULT_ROOT = Path("burgers_multiseed_results")
REFERENCE_FILE = Path("burgers_shock.mat")
ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"

IMAGE_DPI = 800

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 4
HIDDEN_NEURONS = 64

INITIAL_ALPHA = 1.0
INITIAL_BETA = 0.0
STAN_INITIAL_BETA = 1.0


RAR_METHOD_DIR_OVERRIDE = None

METHOD_ORDER = ["tanh", "rar", "stan", "nmaf_aat"]

METHOD_LABELS = {
    "tanh": "Standard PINN",
    "nmaf_aat": "NMAF-AAT",
    "rar": "RAR-PINN",
    "stan": "Stan-PINN",
}

LINESTYLES = {
    "tanh": "-",
    "nmaf_aat": "-",
    "rar": "-",
    "stan": "-",
}
MARKERS = {
    "tanh": "o",
    "nmaf_aat": "s",
    "rar": "^",
    "stan": "D",
}


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "lines.linewidth": 1.7,
    })


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


class AdaptiveActivation(nn.Module):


    def __init__(self, in_features, initial_alpha=INITIAL_ALPHA, initial_beta=INITIAL_BETA):
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


def build_model(method):
    if method == "tanh":
        model = TanhPINN()
    elif method == "nmaf_aat":
        model = NMAFAATPINN()
    elif method == "stan":
        model = StanPINN()
    elif method == "rar":

        model = TanhPINN()
    else:
        raise ValueError(f"Unknown method: {method}")

    return model.double().to(DEVICE)


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mean_and_sample_std(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(values) == 0:
        return np.nan, np.nan, 0
    if len(values) == 1:
        return float(values.iloc[0]), np.nan, 1
    return float(values.mean()), float(values.std(ddof=1)), int(len(values))


def format_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{mean_value:.6e}"
    return f"{mean_value:.6e} ± {std_value:.6e}"


def relative_l2(exact, pred):
    exact = np.asarray(exact, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    return float(
        np.linalg.norm(exact - pred, ord=2)
        / np.linalg.norm(exact, ord=2)
    )


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE.resolve()}"
        )

    data = scipy.io.loadmat(REFERENCE_FILE)
    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    exact = np.asarray(data["usol"], dtype=np.float64)

    target_shape = (x.size, t.size)
    if exact.shape != target_shape:
        if exact.T.shape == target_shape:
            exact = exact.T
        else:
            raise ValueError(
                f"Unexpected usol shape {exact.shape}; expected {target_shape}."
            )

    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, exact


@torch.no_grad()
def predict(model, X, T, output_shape, batch_size=65536):
    model.eval()
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    chunks = []

    for start in range(0, len(x_flat), batch_size):
        stop = min(start + batch_size, len(x_flat))
        x_tensor = torch.as_tensor(
            x_flat[start:stop], dtype=DTYPE, device=DEVICE
        )
        t_tensor = torch.as_tensor(
            t_flat[start:stop], dtype=DTYPE, device=DEVICE
        )
        chunks.append(model(x_tensor, t_tensor).cpu().numpy())

    return np.vstack(chunks).reshape(output_shape)


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' missing in {path}")

    state_dict = {
        key: (
            value.double()
            if torch.is_tensor(value) and value.dtype == torch.float32
            else value
        )
        for key, value in checkpoint["model_state_dict"].items()
    }

    return checkpoint, state_dict


def fixed_method_dir(method):
    path = RESULT_ROOT / method
    if not path.exists():
        raise FileNotFoundError(f"Missing result directory: {path}")
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
        for seed in SEEDS
    )


def locate_rar_method_dir():
    if RAR_METHOD_DIR_OVERRIDE is not None:
        path = RESULT_ROOT / RAR_METHOD_DIR_OVERRIDE
        if not _is_valid_rar_dir(path):
            raise FileNotFoundError(
                "RAR_METHOD_DIR_OVERRIDE is not a valid RAR result directory: "
                f"{path}"
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
        if path.name not in excluded and _is_valid_rar_dir(path)
    ]

    rar_named = [
        path for path in candidates
        if "rar" in path.name.lower()
    ]

    if len(rar_named) == 1:
        return rar_named[0]
    if len(candidates) == 1:
        return candidates[0]

    text = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "RAR result directory could not be identified uniquely.\n"
        "Set RAR_METHOD_DIR_OVERRIDE near the top of this script.\n"
        f"Candidates:\n{text if text else '  (none)'}"
    )


def fixed_checkpoint_path(method, seed, checkpoint_eval):
    return (
        fixed_method_dir(method)
        / f"seed_{seed}"
        / f"burgers_{method}_seed{seed}_eval{checkpoint_eval:05d}.pth"
    )


def find_rar_checkpoint(rar_dir, seed, checkpoint_eval):
    seed_dir = rar_dir / f"seed_{seed}"
    pattern = f"burgers_*_seed{seed}_eval{checkpoint_eval:05d}.pth"
    matches = sorted(seed_dir.glob(pattern))

    rar_matches = [
        p for p in matches
        if "rar" in p.name.lower()
    ]

    if len(rar_matches) == 1:
        return rar_matches[0]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely locate RAR checkpoint in {seed_dir}\n"
        f"Pattern: {pattern}\nMatches: {[str(p) for p in matches]}"
    )


def load_fixed_history(method, seed):
    path = fixed_method_dir(method) / f"seed_{seed}" / "lbfgs_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "lbfgs_eval",
        "total_loss",
        "residual_loss",
        "data_loss",
        "bc_loss",
        "ic_loss",
        "elapsed_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame.loc[
        frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS))
    ].copy()

    return frame.drop_duplicates("lbfgs_eval", keep="last")


def load_rar_monitor_history(rar_dir, seed):
    path = rar_dir / f"seed_{seed}" / "fixed_monitor_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "lbfgs_eval",
        "fixed_monitor_total_loss",
        "fixed_monitor_residual_loss",
        "fixed_monitor_data_loss",
        "fixed_monitor_bc_loss",
        "fixed_monitor_ic_loss",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame.loc[
        frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS))
    ].copy()

    return frame.drop_duplicates("lbfgs_eval", keep="last")


def load_rar_optimization_history(rar_dir, seed):
    path = rar_dir / f"seed_{seed}" / "lbfgs_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "lbfgs_eval",
        "optimization_total_loss",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    frame = frame.loc[
        frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS))
    ].copy()

    return frame.drop_duplicates("lbfgs_eval", keep="last")


def history_row(frame, actual_eval):
    row = frame.loc[frame["lbfgs_eval"] == int(actual_eval)]
    if row.empty:
        raise ValueError(f"No history row for actual_eval={actual_eval}")
    return row.iloc[-1]


def rebuild_individual_results(rar_dir):
    x, t, X, T, exact = load_reference()
    rows = []

    for method in METHOD_ORDER:
        for seed in SEEDS:
            if method == "rar":
                history = load_rar_monitor_history(rar_dir, seed)
            else:
                history = load_fixed_history(method, seed)

            for checkpoint_eval in CHECKPOINT_EVALS:
                if method == "rar":
                    checkpoint_path = find_rar_checkpoint(
                        rar_dir, seed, checkpoint_eval
                    )
                else:
                    checkpoint_path = fixed_checkpoint_path(
                        method, seed, checkpoint_eval
                    )

                checkpoint, state_dict = load_checkpoint(checkpoint_path)
                model = build_model(method)
                model.load_state_dict(state_dict, strict=True)

                pred = predict(model, X, T, exact.shape)
                abs_error = np.abs(exact - pred)

                actual_eval = int(
                    checkpoint.get(
                        "actual_lbfgs_eval",
                        checkpoint.get("lbfgs_eval", checkpoint_eval),
                    )
                )

                selected = history_row(history, actual_eval)

                if method == "rar":
                    common_total_loss = float(
                        selected["fixed_monitor_total_loss"]
                    )
                    residual_loss = float(
                        selected["fixed_monitor_residual_loss"]
                    )
                    data_loss = float(
                        selected["fixed_monitor_data_loss"]
                    )
                    bc_loss = float(
                        selected["fixed_monitor_bc_loss"]
                    )
                    ic_loss = float(
                        selected["fixed_monitor_ic_loss"]
                    )

                    latest = checkpoint.get("latest_metrics", {})
                    optimization_total_loss = float(
                        latest.get("optimization_total_loss", np.nan)
                    )


                    lbfgs_core_time = float(
                        checkpoint.get(
                            "core_lbfgs_time_seconds_estimate",
                            checkpoint.get("elapsed_seconds", np.nan),
                        )
                    )
                    wall_time = float(
                        checkpoint.get("elapsed_seconds", np.nan)
                    )
                    loss_role = "original shared 5000-point monitor"
                else:
                    common_total_loss = float(selected["total_loss"])
                    optimization_total_loss = common_total_loss
                    residual_loss = float(selected["residual_loss"])
                    data_loss = float(selected["data_loss"])
                    bc_loss = float(selected["bc_loss"])
                    ic_loss = float(selected["ic_loss"])
                    lbfgs_core_time = float(
                        checkpoint.get(
                            "elapsed_seconds",
                            selected["elapsed_seconds"],
                        )
                    )
                    wall_time = lbfgs_core_time
                    loss_role = "original shared 5000-point training loss"

                prediction_name = (
                    f"prediction_{method}_seed{seed}"
                    f"_eval{checkpoint_eval:05d}.npz"
                )

                np.savez_compressed(
                    ANALYSIS_DIR / prediction_name,
                    x=x,
                    t=t,
                    u_exact=exact,
                    u_pred=pred,
                    abs_error=abs_error,
                )

                rows.append({
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": int(seed),
                    "checkpoint_eval": int(checkpoint_eval),
                    "actual_lbfgs_eval": int(actual_eval),
                    "common_total_loss": common_total_loss,
                    "optimization_total_loss": optimization_total_loss,
                    "residual_loss": residual_loss,
                    "data_loss": data_loss,
                    "bc_loss": bc_loss,
                    "ic_loss": ic_loss,
                    "relative_l2_error": relative_l2(exact, pred),
                    "maximum_absolute_error": float(abs_error.max()),
                    "lbfgs_core_time_seconds": lbfgs_core_time,
                    "wall_time_to_checkpoint_seconds": wall_time,
                    "loss_role": loss_role,
                    "checkpoint_file": str(checkpoint_path),
                    "prediction_file": prediction_name,
                })

                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    combined = pd.DataFrame(rows)

    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    combined["_method_rank"] = combined["method"].map(method_rank)
    combined = (
        combined
        .sort_values(["checkpoint_eval", "_method_rank", "seed"])
        .drop(columns="_method_rank")
        .reset_index(drop=True)
    )

    combined.to_csv(
        ANALYSIS_DIR / "four_method_individual_checkpoint_results.csv",
        index=False,
    )

    return combined


def create_checkpoint_statistics(combined):
    metrics = [
        "common_total_loss",
        "optimization_total_loss",
        "residual_loss",
        "data_loss",
        "relative_l2_error",
        "maximum_absolute_error",
        "lbfgs_core_time_seconds",
        "wall_time_to_checkpoint_seconds",
    ]

    rows = []

    for (method, checkpoint_eval), group in combined.groupby(
        ["method", "checkpoint_eval"]
    ):
        row = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "checkpoint_eval": int(checkpoint_eval),
            "number_of_seeds": int(group["seed"].nunique()),
        }

        for metric in metrics:
            mean_value, std_value, n = mean_and_sample_std(group[metric])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sample_std"] = std_value
            row[f"{metric}_valid_n"] = n

        rows.append(row)

    stats = pd.DataFrame(rows)

    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    stats["_method_rank"] = stats["method"].map(method_rank)
    stats = (
        stats
        .sort_values(["checkpoint_eval", "_method_rank"])
        .drop(columns="_method_rank")
        .reset_index(drop=True)
    )

    stats.to_csv(
        ANALYSIS_DIR / "four_method_mean_std_statistics.csv",
        index=False,
    )

    paper_rows = []
    for _, row in stats.iterrows():
        paper_rows.append({
            "Method": row["method_label"],
            "Checkpoint": int(row["checkpoint_eval"]),
            "Seeds": int(row["number_of_seeds"]),
            "Common total loss": format_mean_std(
                row["common_total_loss_mean"],
                row["common_total_loss_sample_std"],
            ),
            "Relative L2 error": format_mean_std(
                row["relative_l2_error_mean"],
                row["relative_l2_error_sample_std"],
            ),
            "Maximum absolute error": format_mean_std(
                row["maximum_absolute_error_mean"],
                row["maximum_absolute_error_sample_std"],
            ),
        })

    pd.DataFrame(paper_rows).to_csv(
        ANALYSIS_DIR / "four_method_paper_ready_mean_std_table.csv",
        index=False,
    )

    return stats


def log_statistics_from_seed_frames(frames, loss_column):
    merged = None

    for seed in SEEDS:
        frame = frames[seed][["lbfgs_eval", loss_column]].copy()
        frame = (
            frame
            .drop_duplicates("lbfgs_eval", keep="last")
            .rename(columns={loss_column: f"loss_seed_{seed}"})
        )

        merged = (
            frame
            if merged is None
            else merged.merge(frame, on="lbfgs_eval", how="inner")
        )

    if merged is None or merged.empty:
        raise RuntimeError("No common evaluation indices across the three seeds.")

    seed_columns = [f"loss_seed_{seed}" for seed in SEEDS]
    values = merged[seed_columns].to_numpy(dtype=float)

    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"Invalid values in '{loss_column}'.")

    log_values = np.log10(values)
    log_mean = np.mean(log_values, axis=1)
    log_std = np.std(log_values, axis=1, ddof=1)

    stats = pd.DataFrame({
        "lbfgs_eval": merged["lbfgs_eval"].to_numpy(dtype=int),
        "geometric_mean_total_loss": 10.0 ** log_mean,
        "log10_sample_std": log_std,
        "lower_band": 10.0 ** (log_mean - log_std),
        "upper_band": 10.0 ** (log_mean + log_std),
    })

    for seed in SEEDS:
        stats[f"total_loss_seed_{seed}"] = merged[
            f"loss_seed_{seed}"
        ].to_numpy(dtype=float)

    return stats


def build_common_loss_statistics(rar_dir):
    outputs = []

    for method in METHOD_ORDER:
        if method == "rar":
            frames = {
                seed: load_rar_monitor_history(rar_dir, seed)
                for seed in SEEDS
            }
            loss_column = "fixed_monitor_total_loss"
        else:
            frames = {
                seed: load_fixed_history(method, seed)
                for seed in SEEDS
            }
            loss_column = "total_loss"

        stats = log_statistics_from_seed_frames(frames, loss_column)
        stats.insert(0, "method", method)
        stats.insert(1, "method_label", METHOD_LABELS[method])
        stats.insert(2, "loss_role", "original shared 5000-point loss")
        outputs.append(stats)

    combined = pd.concat(outputs, ignore_index=True)

    combined.to_csv(
        ANALYSIS_DIR / "four_method_common_loss_log_statistics.csv",
        index=False,
    )

    return combined


def build_rar_optimization_statistics(rar_dir):
    frames = {
        seed: load_rar_optimization_history(rar_dir, seed)
        for seed in SEEDS
    }

    stats = log_statistics_from_seed_frames(
        frames,
        "optimization_total_loss",
    )

    stats.to_csv(
        ANALYSIS_DIR / "rar_optimization6000_loss_log_statistics.csv",
        index=False,
    )

    return stats


def plot_common_loss(common_stats):
    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))

    for method in METHOD_ORDER:
        frame = common_stats.loc[
            common_stats["method"] == method
        ].copy()

        line, = ax.plot(
            frame["lbfgs_eval"],
            frame["geometric_mean_total_loss"],
            linestyle=LINESTYLES[method],
            linewidth=1.9,
            label=METHOD_LABELS[method],
            zorder=3,
        )

        ax.fill_between(
            frame["lbfgs_eval"],
            frame["lower_band"],
            frame["upper_band"],
            color=line.get_color(),
            alpha=0.14,
            linewidth=0.0,
            zorder=1,
        )

    ax.set_yscale("log")
    ax.set_xlim(1, max(CHECKPOINT_EVALS))
    if TOTAL_LOSS_YLIM is not None:
        ax.set_ylim(*TOTAL_LOSS_YLIM)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")


    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, numticks=100)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=1000,
        )
    )


    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.55,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.32,
    )


    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )

    ax.legend(
    frameon=True,
    handlelength=2.5,
    handletextpad=0.8,
)
    fig.tight_layout()

    path = ANALYSIS_DIR / "burgers_four_method_common_total_loss.png"
    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    return path


def plot_checkpoint_metric(combined, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))
    offsets = dict(zip(METHOD_ORDER, np.linspace(-75, 75, len(METHOD_ORDER))))

    for method in METHOD_ORDER:
        sub = combined.loc[combined["method"] == method]
        mean_x = []
        mean_y = []

        for checkpoint_eval in CHECKPOINT_EVALS:
            group = sub.loc[sub["checkpoint_eval"] == checkpoint_eval]
            mean_value, _, _ = mean_and_sample_std(group[metric])

            x_center = checkpoint_eval + offsets[method]
            mean_x.append(x_center)
            mean_y.append(mean_value)


            seed_offsets = np.linspace(-16, 16, len(SEEDS))
            for seed, seed_offset in zip(SEEDS, seed_offsets):
                value = group.loc[group["seed"] == seed, metric]
                if not value.empty:
                    ax.scatter(
                        x_center + seed_offset,
                        float(value.iloc[-1]),
                        marker=MARKERS[method],
                        s=26,
                        alpha=0.30,
                        zorder=2,
                    )

        ax.plot(
            mean_x,
            mean_y,
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=6.5,
            linewidth=1.8,
            label=METHOD_LABELS[method],
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_xticks(list(CHECKPOINT_EVALS))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.40)
    ax.legend(frameon=True)
    fig.tight_layout()

    path = ANALYSIS_DIR / filename
    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    return path


def plot_accuracy_figures(combined):
    l2_path = plot_checkpoint_metric(
        combined,
        "relative_l2_error",
        "Relative L2 Error",
        "burgers_four_method_relative_l2.png",
    )

    max_path = plot_checkpoint_metric(
        combined,
        "maximum_absolute_error",
        "Maximum Absolute Error",
        "burgers_four_method_max_abs_error.png",
    )

    return l2_path, max_path


def load_prediction(method, checkpoint_eval):
    path = (
        ANALYSIS_DIR
        / f"prediction_{method}_seed{REPRESENTATIVE_SEED}"
        f"_eval{checkpoint_eval:05d}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path)
    return {key: data[key] for key in data.files}


def plot_representative_profiles(
    checkpoint_eval,
    requested_times=(0.25, 0.50, 0.75),
):
    predictions = {
        method: load_prediction(method, checkpoint_eval)
        for method in METHOD_ORDER
    }

    reference = predictions["tanh"]
    x = reference["x"].reshape(-1)
    t = reference["t"].reshape(-1)
    exact = reference["u_exact"]

    fig, axes = plt.subplots(
        1,
        len(requested_times),
        figsize=(16.8, 4.8),
        sharey=True,
    )

    if len(requested_times) == 1:
        axes = [axes]

    for ax, requested_time in zip(axes, requested_times):
        tid = int(np.argmin(np.abs(t - requested_time)))
        actual_time = float(t[tid])

        ax.plot(
            x,
            exact[:, tid],
            linestyle="-",
            linewidth=2.1,
            label="Reference",
        )

        for method in METHOD_ORDER:
            ax.plot(
                x,
                predictions[method]["u_pred"][:, tid],
                linestyle=LINESTYLES[method],
                linewidth=1.6,
                label=METHOD_LABELS[method],
            )

        ax.set_xlabel("x")
        ax.set_title(f"t = {actual_time:.3f}")
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel("u(x,t)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    path = (
        ANALYSIS_DIR
        / f"burgers_four_method_profiles_eval{checkpoint_eval:05d}.png"
    )

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    return path


def plot_rar_monitor_vs_optimization(common_stats, rar_optimization_stats):
    monitor = common_stats.loc[
        common_stats["method"] == "rar"
    ].copy()

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    line_monitor, = ax.plot(
        monitor["lbfgs_eval"],
        monitor["geometric_mean_total_loss"],
        linestyle="-",
        linewidth=1.9,
        label="RAR: original 5000-point monitor",
    )
    ax.fill_between(
        monitor["lbfgs_eval"],
        monitor["lower_band"],
        monitor["upper_band"],
        color=line_monitor.get_color(),
        alpha=0.14,
        linewidth=0.0,
    )

    line_opt, = ax.plot(
        rar_optimization_stats["lbfgs_eval"],
        rar_optimization_stats["geometric_mean_total_loss"],
        linestyle="--",
        linewidth=1.9,
        label="RAR: actual 6000-point objective",
    )
    ax.fill_between(
        rar_optimization_stats["lbfgs_eval"],
        rar_optimization_stats["lower_band"],
        rar_optimization_stats["upper_band"],
        color=line_opt.get_color(),
        alpha=0.14,
        linewidth=0.0,
    )

    ax.set_yscale("log")
    ax.set_xlim(1, max(CHECKPOINT_EVALS))
    if TOTAL_LOSS_YLIM is not None:
        ax.set_ylim(*TOTAL_LOSS_YLIM)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total loss")

    ax.yaxis.set_major_locator(
        LogLocator(base=10.0, numticks=100)
    )
    ax.yaxis.set_minor_locator(
        LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=1000,
        )
    )

    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.55,
    )
    ax.grid(
        True,
        which="minor",
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.32,
    )
    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )

    ax.legend(frameon=True)
    fig.tight_layout()

    path = ANALYSIS_DIR / "rar_common5000_vs_optimization6000_loss.png"
    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    return path


def print_summary(stats, rar_dir):
    print("\n" + "=" * 118)
    print("Burgers unified four-method statistics")
    print("=" * 118)
    print(f"Seeds: {SEEDS}")
    print(f"RAR result directory: {rar_dir}")
    print(
        "Primary loss: all four methods evaluated on the original shared "
        "5000 PDE points; RAR uses fixed_monitor_total_loss."
    )
    print(
        "RAR 6000-point optimization loss is diagnostic only. "
        "RAR point-selection time is not included in the reported L-BFGS core time."
    )

    for checkpoint_eval in CHECKPOINT_EVALS:
        print("\n" + "-" * 118)
        print(f"Checkpoint = {checkpoint_eval}")
        print("-" * 118)

        sub = stats.loc[stats["checkpoint_eval"] == checkpoint_eval]

        for method in METHOD_ORDER:
            row = sub.loc[sub["method"] == method].iloc[0]
            print(
                f"{METHOD_LABELS[method]:18s} | "
                f"Loss={format_mean_std(row['common_total_loss_mean'], row['common_total_loss_sample_std'])} | "
                f"L2={format_mean_std(row['relative_l2_error_mean'], row['relative_l2_error_sample_std'])} | "
                f"Max={format_mean_std(row['maximum_absolute_error_mean'], row['maximum_absolute_error_sample_std'])}"
            )

    print("=" * 118)


def main():
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            f"Missing result root: {RESULT_ROOT.resolve()}"
        )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    rar_dir = locate_rar_method_dir()

    combined = rebuild_individual_results(rar_dir)
    checkpoint_stats = create_checkpoint_statistics(combined)

    common_stats = build_common_loss_statistics(rar_dir)
    rar_optimization_stats = build_rar_optimization_statistics(rar_dir)

    loss_figure = plot_common_loss(common_stats)
    l2_figure, max_figure = plot_accuracy_figures(combined)

    profile_figures = [
        str(plot_representative_profiles(checkpoint_eval))
        for checkpoint_eval in CHECKPOINT_EVALS
    ]

    rar_diagnostic = plot_rar_monitor_vs_optimization(
        common_stats,
        rar_optimization_stats,
    )

    manifest = {
        "seeds": SEEDS,
        "representative_seed": REPRESENTATIVE_SEED,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "rar_method_directory": str(rar_dir),
        "primary_loss_rule": (
            "All four methods are compared on the original shared 5000 PDE "
            "points; RAR uses fixed_monitor_total_loss."
        ),
        "rar_optimization_rule": (
            "RAR 6000-point optimization loss is diagnostic only."
        ),
        "loss_curve_statistics": (
            "geometric mean +/- 1 sample standard deviation in log10(loss) space"
        ),
        "checkpoint_statistics": (
            "arithmetic mean +/- sample standard deviation on original scale"
        ),
        "runtime_note": (
            "RAR lbfgs_core_time excludes point-selection time and is not an "
            "end-to-end RAR runtime."
        ),
        "figures": {
            "common_loss": str(loss_figure),
            "relative_l2": str(l2_figure),
            "maximum_absolute_error": str(max_figure),
            "profiles": profile_figures,
            "rar_diagnostic": str(rar_diagnostic),
        },
    }

    save_json(
        ANALYSIS_DIR / "analysis_manifest.json",
        manifest,
    )

    print_summary(checkpoint_stats, rar_dir)
    print(f"\nUnified analysis completed: {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
