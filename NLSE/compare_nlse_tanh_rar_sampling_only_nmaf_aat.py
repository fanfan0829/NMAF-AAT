import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn


TOTAL_LOSS_YLIM = None


LOSS_FIG_WIDTH = 7.0
LOSS_FIG_HEIGHT = 5.0


FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TITLE_SIZE = 13
LEGEND_SIZE = 11
TICK_LABEL_SIZE = 12


LEGEND_HANDLE_LENGTH = 3.5
LEGEND_HANDLE_TEXT_PAD = 0.8


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

SEEDS = [99, 1234, 2026]
REPRESENTATIVE_SEED = 99
CHECKPOINT_EVALS = (3000, 6000, 10000)

RESULT_ROOT = SCRIPT_DIR / "nlse_multiseed_results"
REFERENCE_FILE = SCRIPT_DIR / "NLS.mat"
ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"

IMAGE_DPI = 800
SHOW_FIGURES = True


DEVICE = torch.device("cpu")
DTYPE = torch.float64

HIDDEN_LAYERS = 6
HIDDEN_NEURONS = 100
OUTPUT_DIM = 2

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


LINESTYLES = {
    "tanh": "-",
    "rar": "-",
    "stan": "-",
    "nmaf_aat": "-",
}

MARKERS = {
    "tanh": "o",
    "rar": "^",
    "stan": "D",
    "nmaf_aat": "s",
}


def configure_plot_style():

    plt.rcParams["path.simplify"] = False

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
        "lines.linewidth": 1.5,
    })


def finish_figure(fig, path):


    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    print(f"Saved PNG figure: {path.resolve()}")

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)
    return path


class TanhPINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_neurons)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_neurons, hidden_neurons)
            for _ in range(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)
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
        self.alpha = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_alpha)
        self.beta = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_beta)

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
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)

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
        self.beta = nn.Parameter(torch.ones(in_features, dtype=DTYPE) * initial_beta)

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
        self.output_layer = nn.Linear(hidden_neurons, OUTPUT_DIM)

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
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Missing reference file: {REFERENCE_FILE.resolve()}")
    data = scipy.io.loadmat(REFERENCE_FILE)
    if "x" not in data:
        raise KeyError("NLS.mat must contain 'x'.")
    if "tt" in data:
        t_key = "tt"
    elif "t" in data:
        t_key = "t"
    else:
        raise KeyError("NLS.mat must contain 'tt' or 't'.")
    if "uu" not in data:
        raise KeyError("NLS.mat must contain complex reference field 'uu'.")

    x = np.asarray(data["x"]).reshape(-1).astype(np.float64)
    t = np.asarray(data[t_key]).reshape(-1).astype(np.float64)
    h_exact = np.asarray(data["uu"])
    target_shape = (x.size, t.size)
    if h_exact.shape != target_shape:
        if h_exact.T.shape == target_shape:
            h_exact = h_exact.T
            print(f"Reference solution transposed to {target_shape}.")
        else:
            raise ValueError(
                f"Unexpected NLS.mat uu shape {h_exact.shape}; expected {target_shape} or transpose."
            )
    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, h_exact


def predict_complex(model, X, T, output_shape, batch_size=20000):
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=DTYPE, device=DEVICE)
    t_flat = torch.tensor(T.reshape(-1, 1), dtype=DTYPE, device=DEVICE)
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x_flat.shape[0], batch_size):
            end = min(start + batch_size, x_flat.shape[0])
            parts.append(model(x_flat[start:end], t_flat[start:end]).detach().cpu().numpy())
    sol = np.vstack(parts)
    u = sol[:, 0].reshape(output_shape)
    v = sol[:, 1].reshape(output_shape)
    h = u + 1j * v
    return h, u, v


def relative_l2(h_exact, h_pred):
    exact = np.asarray(h_exact).reshape(-1)
    pred = np.asarray(h_pred).reshape(-1)
    denom = np.linalg.norm(exact, 2)
    if denom == 0.0:
        raise ZeroDivisionError("Reference field has zero L2 norm.")
    return float(np.linalg.norm(exact - pred, 2) / denom)


def mean_and_sample_std(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, 0
    if len(s) == 1:
        return float(s.iloc[0]), np.nan, 1
    return float(s.mean()), float(s.std(ddof=1)), int(len(s))


def format_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{mean_value:.6e}"
    return f"{mean_value:.6e} ± {std_value:.6e}"


def fixed_method_dir(method):
    return RESULT_ROOT / method


def _is_valid_rar_dir(path):
    return path.is_dir() and all(
        (path / f"seed_{seed}" / "fixed_monitor_history.csv").exists()
        for seed in SEEDS
    )


def locate_rar_method_dir():
    if RAR_METHOD_DIR_OVERRIDE is not None:
        path = RESULT_ROOT / RAR_METHOD_DIR_OVERRIDE
        if not _is_valid_rar_dir(path):
            raise FileNotFoundError(
                f"RAR_METHOD_DIR_OVERRIDE is not a valid RAR result directory: {path}"
            )
        return path

    excluded = {
        "tanh", "nmaf_aat", "stan", "fixed_training_points",
        "four_method_unified_analysis",
    }
    candidates = [
        p for p in RESULT_ROOT.iterdir()
        if p.name not in excluded and _is_valid_rar_dir(p)
    ]
    rar_named = [p for p in candidates if "rar" in p.name.lower()]
    if len(rar_named) == 1:
        return rar_named[0]
    if len(candidates) == 1:
        return candidates[0]
    text = "\n".join(f"  - {p}" for p in candidates) or "  (none)"
    raise RuntimeError(
        "RAR result directory could not be identified uniquely.\n"
        "Set RAR_METHOD_DIR_OVERRIDE near the top of this script.\n"
        f"Candidates:\n{text}"
    )


def fixed_checkpoint_path(method, seed, checkpoint_eval):


    seed_dir = fixed_method_dir(method) / f"seed_{seed}"

    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Missing seed directory: {seed_dir}"
        )

    ev = f"{int(checkpoint_eval):05d}"

    preferred = [
        seed_dir / f"nlse_{method}_seed{seed}_eval{ev}.pth",
        seed_dir / f"nlse_{method}_seed{seed}_eval{ev}.pt",
        seed_dir / f"checkpoint_eval_{ev}.pth",
        seed_dir / f"checkpoint_eval_{ev}.pt",
    ]

    existing = [p for p in preferred if p.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:

        formal = [
            p for p in existing
            if f"nlse_{method}" in p.name.lower()
        ]
        if len(formal) == 1:
            return formal[0]
        return existing[0]

    matches = []
    for pattern in (
        f"*{method}*seed{seed}*eval{ev}*.pth",
        f"*{method}*seed{seed}*eval{ev}*.pt",
        f"*seed{seed}*eval{ev}*.pth",
        f"*seed{seed}*eval{ev}*.pt",
        f"*eval{ev}*.pth",
        f"*eval{ev}*.pt",
        f"*{ev}*.pth",
        f"*{ev}*.pt",
    ):
        matches.extend(seed_dir.glob(pattern))

    unique = []
    seen = set()
    for p in sorted(matches):
        key = str(p.resolve())
        if key not in seen:
            unique.append(p)
            seen.add(key)

    if len(unique) == 1:
        return unique[0]

    method_named = [
        p for p in unique
        if method.lower() in p.name.lower()
    ]
    if len(method_named) == 1:
        return method_named[0]

    raise FileNotFoundError(
        "Could not uniquely locate formal checkpoint for "
        f"method={METHOD_LABELS[method]}, seed={seed}, "
        f"checkpoint={checkpoint_eval}.\n"
        f"Seed directory: {seed_dir}\n"
        "Matches:\n"
        + ("\n".join(f"  - {p.name}" for p in unique) if unique else "  (none)")
    )


def find_rar_checkpoint(rar_dir, seed, checkpoint_eval):


    seed_dir = rar_dir / f"seed_{seed}"

    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Missing RAR seed directory: {seed_dir}"
        )

    target_eval = int(checkpoint_eval)


    candidates = sorted(
        list(seed_dir.glob("*.pth"))
        + list(seed_dir.glob("*.pt"))
    )

    exact_matches = []

    for path in candidates:
        name = path.name


        match = re.search(
            r"eval_?0*(\d+)(?!\d)",
            name,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        file_eval = int(match.group(1))

        if file_eval == target_eval:
            exact_matches.append(path)

    if len(exact_matches) == 1:
        return exact_matches[0]


    rar_named = [
        path
        for path in exact_matches
        if "rar" in path.name.lower()
    ]

    if len(rar_named) == 1:
        return rar_named[0]

    matches_text = (
        "\n".join(f"  - {p.name}" for p in exact_matches)
        if exact_matches
        else "  (none)"
    )

    raise FileNotFoundError(
        f"Could not uniquely locate RAR checkpoint for seed={seed}, "
        f"eval={checkpoint_eval} in {seed_dir}.\n"
        f"Exact eval-token matches:\n{matches_text}"
    )


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ck:
        raise KeyError(f"Checkpoint missing model_state_dict: {path}")
    state = {
        k: (v.double() if torch.is_tensor(v) and v.dtype == torch.float32 else v)
        for k, v in ck["model_state_dict"].items()
    }
    return ck, state


def checkpoint_actual_eval(ck, target_eval):
    value = ck.get("actual_lbfgs_eval", ck.get("lbfgs_eval", target_eval))
    try:
        return int(value)
    except Exception:
        return int(target_eval)


def checkpoint_elapsed(ck):
    for key in (
        "core_lbfgs_time_seconds_estimate", "elapsed_seconds",
        "elapsed_time_seconds", "lbfgs_time_seconds",
    ):
        if key in ck:
            try:
                return float(ck[key])
            except Exception:
                pass
    return np.nan


def _first_existing_column(frame, names, context, required=True):
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"{context}: none of columns {names} exist. Available={list(frame.columns)}")
    return None


def load_fixed_history(method, seed):
    path = fixed_method_dir(method) / f"seed_{seed}" / "lbfgs_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"lbfgs_eval", "total_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame.loc[frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS))].copy()
    return frame.drop_duplicates("lbfgs_eval", keep="last")


def load_rar_monitor_history(rar_dir, seed):
    path = rar_dir / f"seed_{seed}" / "fixed_monitor_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    eval_col = _first_existing_column(frame, ["lbfgs_eval", "eval", "evaluation"], str(path))
    loss_col = _first_existing_column(
        frame,
        ["fixed_monitor_total_loss", "monitor_total_loss", "common_total_loss"],
        str(path),
    )
    rename = {eval_col: "lbfgs_eval", loss_col: "fixed_monitor_total_loss"}
    optional = {
        "fixed_monitor_residual_loss": ["fixed_monitor_residual_loss", "monitor_residual_loss"],
        "fixed_monitor_data_loss": ["fixed_monitor_data_loss", "monitor_data_loss"],
    }
    for dst, candidates in optional.items():
        src = _first_existing_column(frame, candidates, str(path), required=False)
        if src is not None:
            rename[src] = dst
    frame = frame.rename(columns=rename)
    frame = frame.loc[frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS))].copy()
    return frame.drop_duplicates("lbfgs_eval", keep="last")


def load_rar_optimization_history(rar_dir, seed):
    path = rar_dir / f"seed_{seed}" / "lbfgs_history.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    eval_col = _first_existing_column(frame, ["lbfgs_eval", "eval", "evaluation"], str(path))
    loss_col = _first_existing_column(
        frame,
        ["optimization_total_loss", "total_loss", "training_total_loss"],
        str(path),
    )
    frame = frame.rename(columns={eval_col: "lbfgs_eval", loss_col: "optimization_total_loss"})
    frame = frame.loc[frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS))].copy()
    return frame.drop_duplicates("lbfgs_eval", keep="last")


def history_row(frame, actual_eval):
    exact = frame.loc[frame["lbfgs_eval"] == actual_eval]
    if not exact.empty:
        return exact.iloc[-1]
    prior = frame.loc[frame["lbfgs_eval"] <= actual_eval]
    if prior.empty:
        raise ValueError(f"No history row at or before evaluation {actual_eval}.")
    return prior.iloc[-1]


def rebuild_individual_results(rar_dir):
    x, t, X, T, h_exact = load_reference()
    rows = []
    prediction_dir = ANALYSIS_DIR / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    fixed_histories = {
        (method, seed): load_fixed_history(method, seed)
        for method in ("tanh", "nmaf_aat", "stan")
        for seed in SEEDS
    }
    rar_monitor = {seed: load_rar_monitor_history(rar_dir, seed) for seed in SEEDS}
    rar_opt = {seed: load_rar_optimization_history(rar_dir, seed) for seed in SEEDS}

    for method in METHOD_ORDER:
        for seed in SEEDS:
            for target_eval in CHECKPOINT_EVALS:
                if method == "rar":
                    ck_path = find_rar_checkpoint(rar_dir, seed, target_eval)
                else:
                    ck_path = fixed_checkpoint_path(method, seed, target_eval)
                ck, state = load_checkpoint(ck_path)
                model = build_model(method)
                model.load_state_dict(state, strict=True)
                h_pred, u_pred, v_pred = predict_complex(model, X, T, h_exact.shape)
                complex_abs_error = np.abs(h_exact - h_pred)
                magnitude_error = np.abs(np.abs(h_exact) - np.abs(h_pred))
                l2 = relative_l2(h_exact, h_pred)
                maxerr = float(np.max(complex_abs_error))
                actual_eval = checkpoint_actual_eval(ck, target_eval)

                if method == "rar":
                    common_row = history_row(rar_monitor[seed], actual_eval)
                    opt_row = history_row(rar_opt[seed], actual_eval)
                    common_total_loss = float(common_row["fixed_monitor_total_loss"])
                    optimization_total_loss = float(opt_row["optimization_total_loss"])
                else:
                    common_row = history_row(fixed_histories[(method, seed)], actual_eval)
                    common_total_loss = float(common_row["total_loss"])
                    optimization_total_loss = common_total_loss

                pred_path = (
                    prediction_dir /
                    f"nlse_{method}_seed{seed}_prediction_eval{target_eval:05d}.npz"
                )
                np.savez_compressed(
                    pred_path,
                    x=x,
                    t=t,
                    h_exact=h_exact,
                    h_pred=h_pred,
                    u_pred=u_pred,
                    v_pred=v_pred,
                    complex_abs_error=complex_abs_error,
                    magnitude_error=magnitude_error,
                )

                rows.append({
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": int(seed),
                    "checkpoint_eval": int(target_eval),
                    "actual_lbfgs_eval": int(actual_eval),
                    "common_total_loss": common_total_loss,
                    "optimization_total_loss": optimization_total_loss,
                    "relative_l2_error": l2,
                    "maximum_absolute_error": maxerr,
                    "lbfgs_core_time_seconds": checkpoint_elapsed(ck),
                    "checkpoint_path": str(ck_path),
                    "prediction_path": str(pred_path),
                })
                print(
                    f"{METHOD_LABELS[method]:18s} | seed={seed:4d} | eval={target_eval:5d} | "
                    f"common loss={common_total_loss:.6e} | L2={l2:.6e} | max={maxerr:.6e}"
                )

    combined = pd.DataFrame(rows)
    combined.to_csv(ANALYSIS_DIR / "four_method_individual_checkpoint_results.csv", index=False)
    return combined


def create_checkpoint_statistics(combined):
    rows = []
    for checkpoint_eval in CHECKPOINT_EVALS:
        for method in METHOD_ORDER:
            g = combined.loc[
                (combined["checkpoint_eval"] == checkpoint_eval)
                & (combined["method"] == method)
            ]
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "checkpoint_eval": checkpoint_eval,
                "number_of_seeds": int(g["seed"].nunique()),
            }
            for metric in (
                "common_total_loss", "relative_l2_error",
                "maximum_absolute_error", "lbfgs_core_time_seconds",
            ):
                mean, std, _ = mean_and_sample_std(g[metric])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sample_std"] = std
            rows.append(row)

    stats = pd.DataFrame(rows)
    stats.to_csv(ANALYSIS_DIR / "four_method_mean_std_statistics.csv", index=False)

    paper = stats[["method_label", "checkpoint_eval", "number_of_seeds"]].copy()
    paper["common_total_loss"] = [
        format_mean_std(a, b)
        for a, b in zip(stats["common_total_loss_mean"], stats["common_total_loss_sample_std"])
    ]
    paper["relative_l2_error"] = [
        format_mean_std(a, b)
        for a, b in zip(stats["relative_l2_error_mean"], stats["relative_l2_error_sample_std"])
    ]
    paper["maximum_absolute_error"] = [
        format_mean_std(a, b)
        for a, b in zip(stats["maximum_absolute_error_mean"], stats["maximum_absolute_error_sample_std"])
    ]
    paper.to_csv(ANALYSIS_DIR / "four_method_paper_ready_mean_std_table.csv", index=False)
    return stats


def log_statistics_from_seed_frames(frames, loss_column):
    merged = None
    for seed, frame in frames.items():
        part = frame[["lbfgs_eval", loss_column]].copy().rename(
            columns={loss_column: f"loss_seed_{seed}"}
        )
        merged = part if merged is None else merged.merge(part, on="lbfgs_eval", how="inner")
    cols = [f"loss_seed_{seed}" for seed in SEEDS]
    values = merged[cols].to_numpy(dtype=float)
    if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
        raise ValueError(f"Invalid loss values in {loss_column}.")
    logs = np.log10(values)
    mu = logs.mean(axis=1)
    sd = logs.std(axis=1, ddof=1)
    out = pd.DataFrame({
        "lbfgs_eval": merged["lbfgs_eval"].to_numpy(dtype=int),
        "geometric_mean_total_loss": 10.0 ** mu,
        "log10_sample_std": sd,
        "lower_band": 10.0 ** (mu - sd),
        "upper_band": 10.0 ** (mu + sd),
    })
    for seed in SEEDS:
        out[f"loss_seed_{seed}"] = merged[f"loss_seed_{seed}"].to_numpy(dtype=float)
    return out


def build_common_loss_statistics(rar_dir):
    all_stats = []
    for method in METHOD_ORDER:
        if method == "rar":
            frames = {seed: load_rar_monitor_history(rar_dir, seed) for seed in SEEDS}
            stats = log_statistics_from_seed_frames(frames, "fixed_monitor_total_loss")
        else:
            frames = {seed: load_fixed_history(method, seed) for seed in SEEDS}
            stats = log_statistics_from_seed_frames(frames, "total_loss")
        stats.insert(0, "method", method)
        stats.insert(1, "method_label", METHOD_LABELS[method])
        all_stats.append(stats)
    combined = pd.concat(all_stats, ignore_index=True)
    combined.to_csv(ANALYSIS_DIR / "four_method_common_loss_log_statistics.csv", index=False)
    return combined


def build_rar_optimization_statistics(rar_dir):
    frames = {seed: load_rar_optimization_history(rar_dir, seed) for seed in SEEDS}
    stats = log_statistics_from_seed_frames(frames, "optimization_total_loss")
    stats.to_csv(ANALYSIS_DIR / "rar_optimization12000_loss_log_statistics.csv", index=False)
    return stats


def plot_common_loss(common_stats):
    fig, ax = plt.subplots(
        figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT)
    )

    for method in METHOD_ORDER:
        st = common_stats.loc[
            common_stats["method"] == method
        ].copy()

        line, = ax.plot(
            st["lbfgs_eval"],
            st["geometric_mean_total_loss"],
            linestyle=LINESTYLES[method],
            linewidth=1.5,
            antialiased=True,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=METHOD_LABELS[method],
            zorder=3,
        )

        ax.fill_between(
            st["lbfgs_eval"],
            st["lower_band"],
            st["upper_band"],
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
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )

    fig.tight_layout()

    return finish_figure(
        fig,
        ANALYSIS_DIR / "nlse_four_method_common_total_loss.png",
    )


def plot_checkpoint_metric(combined, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT))
    xvals = np.arange(len(CHECKPOINT_EVALS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(METHOD_ORDER))
    for offset, method in zip(offsets, METHOD_ORDER):
        means, stds = [], []
        for ev in CHECKPOINT_EVALS:
            vals = combined.loc[
                (combined["method"] == method) & (combined["checkpoint_eval"] == ev),
                metric,
            ]
            mean, std, _ = mean_and_sample_std(vals)
            means.append(mean)
            stds.append(0.0 if np.isnan(std) else std)
        ax.errorbar(
            xvals + offset, means, yerr=stds,
            marker=MARKERS[method], linestyle="none", capsize=4,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(xvals, [str(v) for v in CHECKPOINT_EVALS])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()
    return finish_figure(fig, ANALYSIS_DIR / filename)


def plot_accuracy_figures(combined):
    l2 = plot_checkpoint_metric(
        combined, "relative_l2_error", "Relative $L_2$ Error",
        "nlse_four_method_relative_l2.png",
    )
    mx = plot_checkpoint_metric(
        combined, "maximum_absolute_error", "Maximum Absolute Error",
        "nlse_four_method_max_abs_error.png",
    )
    return l2, mx


def prediction_path(method, checkpoint_eval):
    return (
        ANALYSIS_DIR / "predictions" /
        f"nlse_{method}_seed{REPRESENTATIVE_SEED}_prediction_eval{checkpoint_eval:05d}.npz"
    )


def load_prediction(method, checkpoint_eval):
    path = prediction_path(method, checkpoint_eval)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def plot_representative_profiles(checkpoint_eval, requested_times=None):
    if requested_times is None:
        requested_times = (0.25, 0.50, 1.00)
    pred = {m: load_prediction(m, checkpoint_eval) for m in METHOD_ORDER}
    ref = pred["tanh"]
    x = ref["x"].reshape(-1)
    t = ref["t"].reshape(-1)
    h_exact = ref["h_exact"]

    fig, axes = plt.subplots(1, len(requested_times), figsize=(16.8, 4.8), sharey=True)
    if len(requested_times) == 1:
        axes = [axes]
    for ax, req in zip(axes, requested_times):
        tid = int(np.argmin(np.abs(t - req)))
        actual_t = float(t[tid])
        ax.plot(x, np.abs(h_exact[:, tid]), linewidth=2.1, label="Reference")
        for method in METHOD_ORDER:
            ax.plot(
                x, np.abs(pred[method]["h_pred"][:, tid]),
                linestyle=LINESTYLES[method], linewidth=1.6,
                label=METHOD_LABELS[method],
            )
        ax.set_xlabel("x")
        ax.set_title(f"t = {actual_t:.3f}")
        ax.grid(True, alpha=0.35)
    axes[0].set_ylabel(r"$|h(x,t)|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    return finish_figure(
        fig, ANALYSIS_DIR / f"nlse_four_method_profiles_eval{checkpoint_eval:05d}.png"
    )


def plot_representative_fields(checkpoint_eval):
    pred = {m: load_prediction(m, checkpoint_eval) for m in METHOD_ORDER}
    ref = pred["tanh"]
    x = ref["x"].reshape(-1)
    t = ref["t"].reshape(-1)
    exact_mag = np.abs(ref["h_exact"])
    extent = [float(t.min()), float(t.max()), float(x.min()), float(x.max())]

    mags = [exact_mag] + [np.abs(pred[m]["h_pred"]) for m in METHOD_ORDER]
    vmin = min(float(np.min(a)) for a in mags)
    vmax = max(float(np.max(a)) for a in mags)
    err_max = max(float(np.max(pred[m]["complex_abs_error"])) for m in METHOD_ORDER)

    fig, axes = plt.subplots(2, 5, figsize=(19.0, 7.6), constrained_layout=True)
    im0 = axes[0, 0].imshow(
        exact_mag, origin="lower", aspect="auto", extent=extent,
        vmin=vmin, vmax=vmax, cmap="jet",
    )
    axes[0, 0].set_title(r"Reference $|h|$")
    axes[1, 0].axis("off")

    for col, method in enumerate(METHOD_ORDER, start=1):
        axes[0, col].imshow(
            np.abs(pred[method]["h_pred"]), origin="lower", aspect="auto",
            extent=extent, vmin=vmin, vmax=vmax, cmap="jet",
        )
        axes[0, col].set_title(METHOD_LABELS[method])
        axes[1, col].imshow(
            pred[method]["complex_abs_error"], origin="lower", aspect="auto",
            extent=extent, vmin=0.0, vmax=err_max, cmap="Reds",
        )
        axes[1, col].set_title(r"$|h_{ref}-h_{pred}|$")

    for ax in axes[0, :]:
        ax.set_xlabel("t")
    for ax in axes[1, 1:]:
        ax.set_xlabel("t")
    axes[0, 0].set_ylabel("x")
    axes[1, 1].set_ylabel("x")
    cb0 = fig.colorbar(im0, ax=axes[0, :], shrink=0.85, pad=0.02)
    cb0.set_label(r"$|h|$")
    imerr = axes[1, 1].images[0]
    cb1 = fig.colorbar(imerr, ax=axes[1, 1:], shrink=0.85, pad=0.02)
    cb1.set_label("Complex-field absolute error")
    return finish_figure(
        fig, ANALYSIS_DIR / f"nlse_four_method_fields_eval{checkpoint_eval:05d}.png"
    )


def plot_rar_monitor_vs_optimization(common_stats, rar_opt_stats):
    monitor = common_stats.loc[
        common_stats["method"] == "rar"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(LOSS_FIG_WIDTH, LOSS_FIG_HEIGHT)
    )

    line1, = ax.plot(
        monitor["lbfgs_eval"],
        monitor["geometric_mean_total_loss"],
        linewidth=1.5,
        linestyle="-",
        label="RAR: original 10000-point monitor",
        zorder=3,
    )

    ax.fill_between(
        monitor["lbfgs_eval"],
        monitor["lower_band"],
        monitor["upper_band"],
        color=line1.get_color(),
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )

    line2, = ax.plot(
        rar_opt_stats["lbfgs_eval"],
        rar_opt_stats["geometric_mean_total_loss"],
        linestyle="--",
        linewidth=1.5,
        label="RAR: actual 12000-point objective",
        zorder=3,
    )

    ax.fill_between(
        rar_opt_stats["lbfgs_eval"],
        rar_opt_stats["lower_band"],
        rar_opt_stats["upper_band"],
        color=line2.get_color(),
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
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )

    fig.tight_layout()

    return finish_figure(
        fig,
        ANALYSIS_DIR / "rar_common10000_vs_optimization12000_loss.png",
    )


def print_summary(stats, rar_dir):
    print("\n" + "=" * 120)
    print("NLSE unified four-method statistics")
    print("=" * 120)
    print(f"Seeds: {SEEDS}")
    print(f"RAR result directory: {rar_dir}")
    print("Primary loss: original shared 10000-point objective for all methods; RAR uses fixed monitor.")
    print("RAR 12000-point optimization loss is diagnostic only.")
    for ev in CHECKPOINT_EVALS:
        print("\n" + "-" * 120)
        print(f"Checkpoint = {ev}")
        sub = stats.loc[stats["checkpoint_eval"] == ev]
        for method in METHOD_ORDER:
            row = sub.loc[sub["method"] == method].iloc[0]
            print(
                f"{METHOD_LABELS[method]:18s} | "
                f"Loss={format_mean_std(row['common_total_loss_mean'], row['common_total_loss_sample_std'])} | "
                f"L2={format_mean_std(row['relative_l2_error_mean'], row['relative_l2_error_sample_std'])} | "
                f"Max={format_mean_std(row['maximum_absolute_error_mean'], row['maximum_absolute_error_sample_std'])}"
            )
    print("=" * 120)


def main():
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(f"Missing result root: {RESULT_ROOT.resolve()}")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    rar_dir = locate_rar_method_dir()
    combined = rebuild_individual_results(rar_dir)
    checkpoint_stats = create_checkpoint_statistics(combined)
    common_stats = build_common_loss_statistics(rar_dir)
    rar_opt_stats = build_rar_optimization_statistics(rar_dir)

    loss_fig = plot_common_loss(common_stats)
    l2_fig, max_fig = plot_accuracy_figures(combined)
    profile_figs = [str(plot_representative_profiles(ev)) for ev in CHECKPOINT_EVALS]
    field_figs = [str(plot_representative_fields(ev)) for ev in CHECKPOINT_EVALS]
    rar_diag = plot_rar_monitor_vs_optimization(common_stats, rar_opt_stats)

    manifest = {
        "problem": "Nonlinear Schrodinger equation",
        "seeds": SEEDS,
        "representative_seed": REPRESENTATIVE_SEED,
        "checkpoint_evals": list(CHECKPOINT_EVALS),
        "hidden_layers_code_variable": HIDDEN_LAYERS,
        "actual_hidden_activation_locations": HIDDEN_LAYERS + 1,
        "hidden_neurons": HIDDEN_NEURONS,
        "rar_method_directory": str(rar_dir),
        "primary_loss_rule": "RAR uses fixed-monitor loss on original shared 10000 PDE points.",
        "rar_optimization_rule": "RAR actual 12000-point optimization loss is diagnostic only.",
        "loss_curve_statistics": "geometric mean +/- 1 sample std in log10(loss) space",
        "checkpoint_statistics": "arithmetic mean +/- sample std on original scale",
        "accuracy": "full complex-field relative L2 and max |h_ref-h_pred| on NLS.mat",
        "runtime_note": "RAR selection time may not be included; runtime is not in the paper-ready main table.",
        "outputs": {
            "common_loss_figure": str(loss_fig),
            "relative_l2_figure": str(l2_fig),
            "maximum_error_figure": str(max_fig),
            "profiles": profile_figs,
            "fields": field_figs,
            "rar_diagnostic": str(rar_diag),
        },
    }
    save_json(ANALYSIS_DIR / "analysis_manifest.json", manifest)
    print_summary(checkpoint_stats, rar_dir)
    print(f"\nAnalysis output directory: {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
