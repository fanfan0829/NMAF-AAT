import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch

from allen_cahn_common_fixed_points_continuous_lbfgs import (
    CHECKPOINT_EVALS,
    DEVICE,
    DTYPE,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    IMAGE_DPI,
    INITIAL_ALPHA,
    INITIAL_BETA,
    NMAFAATPINN,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    SEEDS,
    TanhPINN,
    adaptive_parameter_summary,
    method_label,
)


METHOD = "tanh"
METHOD_LABEL = method_label(METHOD)
METHOD_DIR = RESULT_ROOT / METHOD
ANALYSIS_DIR = METHOD_DIR / "analysis"
REFERENCE_FILE = Path("AC.mat")


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.6,
    })


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def build_model():
    if METHOD == "tanh":
        return TanhPINN(hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS).double().to(DEVICE)
    if METHOD == "nmaf_aat":
        return NMAFAATPINN(hidden_layers=HIDDEN_LAYERS, hidden_neurons=HIDDEN_NEURONS).double().to(DEVICE)
    raise ValueError(f"Unknown method: {METHOD}")


def load_reference():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Missing reference file: {REFERENCE_FILE}.")
    data = scipy.io.loadmat(REFERENCE_FILE)
    x = np.asarray(data["x"]).flatten()
    t_key = "tt"
    t = np.asarray(data[t_key]).flatten()
    u_exact = np.asarray(data["uu"])
    target_shape = (len(x), len(t))
    if u_exact.shape != target_shape:
        if u_exact.T.shape == target_shape:
            u_exact = u_exact.T
            print(f"Reference solution was transposed to match shape {target_shape}.")
        else:
            raise ValueError(f"Unexpected reference shape: {u_exact.shape}, expected {target_shape}.")
    X, T = np.meshgrid(x, t, indexing="ij")
    return x, t, X, T, u_exact


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k: (v.double() if v.dtype == torch.float32 else v) for k, v in state_dict.items()}
    saved_eval = checkpoint.get("actual_lbfgs_eval", checkpoint.get("lbfgs_eval", "NA"))
    target_eval = checkpoint.get("target_checkpoint_eval", "NA")
    elapsed_seconds = checkpoint.get("elapsed_seconds", "NA")
    latest_metrics = checkpoint.get("latest_metrics", {})
    aat_summary = checkpoint.get("adaptive_parameter_summary", {})
    diagnostic = (
        f"Loaded checkpoint: {path} | "
        f"target_eval={target_eval} | actual_eval={saved_eval} | "
        f"elapsed_seconds={elapsed_seconds} | "
        f"checkpoint_total_loss={latest_metrics.get('total_loss', 'NA')}"
    )
    if METHOD == "nmaf_aat":
        diagnostic += f" | alpha_mean={aat_summary.get('alpha_mean', 'NA')} | beta_mean={aat_summary.get('beta_mean', 'NA')}"
    print(diagnostic)
    return checkpoint, state_dict


def predict(model, X, T, output_shape):
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=DTYPE, device=DEVICE)
    t_flat = torch.tensor(T.reshape(-1, 1), dtype=DTYPE, device=DEVICE)
    model.eval()
    batch_size = 20000
    parts = []
    with torch.no_grad():
        for start in range(0, x_flat.shape[0], batch_size):
            end = start + batch_size
            parts.append(model(x_flat[start:end], t_flat[start:end]).detach().cpu().numpy())
    return np.vstack(parts).reshape(output_shape)


def relative_l2(u_exact, u_pred):
    exact_flat = np.asarray(u_exact).reshape(-1)
    pred_flat = np.asarray(u_pred).reshape(-1)
    return float(np.linalg.norm(exact_flat - pred_flat, 2) / np.linalg.norm(exact_flat, 2))


def mean_and_sample_std(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
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


def rebuild_seed_summary(seed):
    seed_dir = METHOD_DIR / f"seed_{seed}"
    history_path = seed_dir / "lbfgs_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    history = pd.read_csv(history_path)
    x, t, X, T, u_exact = load_reference()
    rows = []
    for checkpoint_eval in CHECKPOINT_EVALS:
        checkpoint_path = seed_dir / f"allen_cahn_{METHOD}_seed{seed}_eval{checkpoint_eval:05d}.pth"
        checkpoint, state_dict = load_checkpoint(checkpoint_path)
        model = build_model()
        model.load_state_dict(state_dict, strict=True)
        u_pred = predict(model, X, T, u_exact.shape)
        abs_error = np.abs(u_exact - u_pred)
        actual_eval = int(checkpoint.get("actual_lbfgs_eval", checkpoint.get("lbfgs_eval", checkpoint_eval)))
        selected = history.loc[history["lbfgs_eval"] == actual_eval]
        if selected.empty:
            raise ValueError(f"No history row exists for seed {seed} at actual eval {actual_eval}.")
        selected = selected.iloc[0]
        prediction_name = f"allen_cahn_{METHOD}_seed{seed}_prediction_eval{checkpoint_eval:05d}.npz"
        np.savez_compressed(seed_dir / prediction_name, x=x, t=t, u_exact=u_exact, u_pred=u_pred, abs_error=abs_error)
        metrics = {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "seed": int(seed),
            "checkpoint_eval": int(checkpoint_eval),
            "actual_lbfgs_eval": int(actual_eval),
            "total_loss": float(selected["total_loss"]),
            "residual_loss": float(selected["residual_loss"]),
            "data_loss": float(selected["data_loss"]),
            "bc_loss": float(selected["bc_loss"]),
            "ic_loss_weighted": float(selected["ic_loss_weighted"]),
            "relative_l2_error": float(relative_l2(u_exact, u_pred)),
            "maximum_absolute_error": float(abs_error.max()),
            "u_exact_min": float(np.min(u_exact)),
            "u_exact_max": float(np.max(u_exact)),
            "u_pred_min": float(np.min(u_pred)),
            "u_pred_max": float(np.max(u_pred)),
            "lbfgs_time_seconds": float(selected["elapsed_seconds"]),
            "prediction_file": prediction_name,
        }
        if METHOD == "nmaf_aat":
            aat = adaptive_parameter_summary(model)
            metrics.update({
                "alpha_mean": aat["alpha_mean"],
                "alpha_std": aat["alpha_std"],
                "alpha_min": aat["alpha_min"],
                "alpha_max": aat["alpha_max"],
                "beta_mean": aat["beta_mean"],
                "beta_std": aat["beta_std"],
                "beta_min": aat["beta_min"],
                "beta_max": aat["beta_max"],
                "delta_alpha_l2": aat["delta_alpha_l2"],
                "delta_beta_l2": aat["delta_beta_l2"],
            })
        save_json(seed_dir / f"metrics_eval{checkpoint_eval:05d}.json", metrics)
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(seed_dir / "seed_summary.csv", index=False)


def rebuild_all_seed_summaries():
    for seed in SEEDS:
        rebuild_seed_summary(seed)


def combine_individual_results():
    frames = [pd.read_csv(METHOD_DIR / f"seed_{seed}" / "seed_summary.csv") for seed in SEEDS]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["checkpoint_eval", "seed"]).reset_index(drop=True)
    combined.to_csv(ANALYSIS_DIR / "all_individual_runs.csv", index=False)
    return combined


def create_statistics_tables(combined):
    metrics = ["total_loss", "residual_loss", "data_loss", "relative_l2_error", "maximum_absolute_error", "lbfgs_time_seconds"]
    if METHOD == "nmaf_aat":
        metrics += ["delta_alpha_l2", "delta_beta_l2"]
    rows = []
    for checkpoint_eval, group in combined.groupby("checkpoint_eval"):
        row = {"method": METHOD, "method_label": METHOD_LABEL, "checkpoint_eval": int(checkpoint_eval), "number_of_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            mean_value, std_value, valid_n = mean_and_sample_std(group[metric])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sample_std"] = std_value
            row[f"{metric}_valid_n"] = valid_n
        rows.append(row)
    numeric = pd.DataFrame(rows)
    numeric.to_csv(ANALYSIS_DIR / "mean_std_statistics.csv", index=False)
    paper_rows = []
    for _, row in numeric.iterrows():
        paper_row = {"Method": row["method_label"], "Checkpoint": int(row["checkpoint_eval"]), "Seeds": int(row["number_of_seeds"])}
        for metric in metrics:
            paper_row[metric] = format_mean_std(row[f"{metric}_mean"], row[f"{metric}_sample_std"])
        paper_rows.append(paper_row)
    pd.DataFrame(paper_rows).to_csv(ANALYSIS_DIR / "paper_ready_mean_std_table.csv", index=False)
    return numeric


def print_individual_checkpoint_accuracy(combined):
    display_columns = [
        "seed", "checkpoint_eval", "actual_lbfgs_eval", "total_loss",
        "relative_l2_error", "maximum_absolute_error",
        "u_exact_min", "u_exact_max", "u_pred_min", "u_pred_max",
    ]
    if METHOD == "nmaf_aat":
        display_columns += ["delta_alpha_l2", "delta_beta_l2"]
    print("\n" + "=" * 116)
    print(f"{METHOD_LABEL} individual checkpoint accuracy")
    print("=" * 116)
    print(combined[display_columns].to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    print("=" * 116)


def print_l2_and_max_abs_statistics(stats):
    print("\n" + "=" * 116)
    print(f"{METHOD_LABEL} checkpoint accuracy statistics")
    print("=" * 116)
    for _, row in stats.iterrows():
        checkpoint_eval = int(row["checkpoint_eval"])
        l2_text = format_mean_std(row["relative_l2_error_mean"], row["relative_l2_error_sample_std"])
        max_abs_text = format_mean_std(row["maximum_absolute_error_mean"], row["maximum_absolute_error_sample_std"])
        line = f"Eval {checkpoint_eval:5d} | L2 = {l2_text} | Max abs error = {max_abs_text}"
        if METHOD == "nmaf_aat":
            da = format_mean_std(row["delta_alpha_l2_mean"], row["delta_alpha_l2_sample_std"])
            db = format_mean_std(row["delta_beta_l2_mean"], row["delta_beta_l2_sample_std"])
            line += f" | ||alpha-1||2 = {da} | ||beta||2 = {db}"
        print(line)
    print("=" * 116)


def load_seed_history(seed):
    path = METHOD_DIR / f"seed_{seed}" / "lbfgs_history.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["lbfgs_eval"].between(1, max(CHECKPOINT_EVALS)), ["lbfgs_eval", "total_loss"]].copy()
    return frame.drop_duplicates(subset="lbfgs_eval", keep="last")


def create_total_loss_curve():
    merged = None
    for seed in SEEDS:
        history = load_seed_history(seed).rename(columns={"total_loss": f"total_loss_seed_{seed}"})
        merged = history if merged is None else merged.merge(history, on="lbfgs_eval", how="inner")
    columns = [f"total_loss_seed_{seed}" for seed in SEEDS]
    values = merged[columns].to_numpy(dtype=float)
    if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
        raise ValueError("Total loss contains invalid values.")
    log_values = np.log10(values)
    log_mean = np.mean(log_values, axis=1)
    log_sample_std = np.std(log_values, axis=1, ddof=1)
    curve_data = pd.DataFrame({
        "lbfgs_eval": merged["lbfgs_eval"].to_numpy(dtype=int),
        "geometric_mean_total_loss": 10.0 ** log_mean,
        "log10_sample_std": log_sample_std,
        "lower_band": 10.0 ** (log_mean - log_sample_std),
        "upper_band": 10.0 ** (log_mean + log_sample_std),
    })
    for seed in SEEDS:
        curve_data[f"total_loss_seed_{seed}"] = merged[f"total_loss_seed_{seed}"].to_numpy(dtype=float)
    curve_data.to_csv(ANALYSIS_DIR / f"{METHOD}_total_loss_log_statistics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot(curve_data["lbfgs_eval"], curve_data["geometric_mean_total_loss"], linewidth=1.8, label=METHOD_LABEL)
    ax.fill_between(curve_data["lbfgs_eval"], curve_data["lower_band"], curve_data["upper_band"], color=line.get_color(), alpha=0.16, linewidth=0.0)
    for checkpoint_eval in CHECKPOINT_EVALS:
        ax.axvline(checkpoint_eval, color="0.35", linestyle="-.", linewidth=0.9, alpha=0.65)
    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS Function Evaluations")
    ax.set_ylabel("Total Loss")
    ax.grid(True, which="both", linestyle="-", alpha=0.5)
    ax.legend(frameon=True)
    ax.set_xlim(curve_data["lbfgs_eval"].min(), max(CHECKPOINT_EVALS))
    fig.tight_layout()
    figure_path = ANALYSIS_DIR / f"allen_cahn_{METHOD}_total_loss.jpg"
    fig.savefig(figure_path, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved total-loss curve: {figure_path}")
    return curve_data


def create_representative_fields():
    for checkpoint_eval in CHECKPOINT_EVALS:
        path = METHOD_DIR / f"seed_{REPRESENTATIVE_SEED}" / f"allen_cahn_{METHOD}_seed{REPRESENTATIVE_SEED}_prediction_eval{checkpoint_eval:05d}.npz"
        data = np.load(path)
        x, t = data["x"], data["t"]
        u_exact, u_pred, abs_error = data["u_exact"], data["u_pred"], data["abs_error"]
        X, T = np.meshgrid(x, t, indexing="ij")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        u_min, u_max = float(np.min(u_exact)), float(np.max(u_exact))
        im0 = axes[0].pcolor(T, X, u_exact, cmap="jet", shading="auto", vmin=u_min, vmax=u_max)
        im1 = axes[1].pcolor(T, X, u_pred, cmap="jet", shading="auto", vmin=u_min, vmax=u_max)
        im2 = axes[2].pcolor(T, X, abs_error, cmap="Reds", shading="auto")
        axes[0].set_title("Reference Solution")
        axes[1].set_title(f"{METHOD_LABEL} Prediction")
        axes[2].set_title("Absolute Error Map")
        for axis, image in zip(axes, [im0, im1, im2]):
            axis.set_xlabel("t")
            axis.set_ylabel("x")
            fig.colorbar(image, ax=axis)
        fig.tight_layout()
        figure_path = ANALYSIS_DIR / f"allen_cahn_{METHOD}_seed{REPRESENTATIVE_SEED}_field_eval{checkpoint_eval:05d}.jpg"
        fig.savefig(figure_path, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved representative field figure: {figure_path}")


def create_representative_aat_histograms():
    if METHOD != "nmaf_aat":
        return
    for checkpoint_eval in CHECKPOINT_EVALS:
        checkpoint_path = METHOD_DIR / f"seed_{REPRESENTATIVE_SEED}" / f"allen_cahn_{METHOD}_seed{REPRESENTATIVE_SEED}_eval{checkpoint_eval:05d}.pth"
        _, state_dict = load_checkpoint(checkpoint_path)
        alpha = state_dict["activation.alpha"].detach().cpu().numpy()
        beta = state_dict["activation.beta"].detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(alpha, bins=20)
        axes[0].axvline(INITIAL_ALPHA, linestyle="--", linewidth=1.2)
        axes[0].set_xlabel(r"$\alpha$")
        axes[0].set_ylabel("Count")
        axes[0].set_title(r"Learned $\alpha$")
        axes[1].hist(beta, bins=20)
        axes[1].axvline(INITIAL_BETA, linestyle="--", linewidth=1.2)
        axes[1].set_xlabel(r"$\beta$")
        axes[1].set_ylabel("Count")
        axes[1].set_title(r"Learned $\beta$")
        fig.tight_layout()
        figure_path = ANALYSIS_DIR / f"allen_cahn_{METHOD}_seed{REPRESENTATIVE_SEED}_aat_hist_eval{checkpoint_eval:05d}.jpg"
        fig.savefig(figure_path, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Saved representative AAT histogram: {figure_path}")


if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    rebuild_all_seed_summaries()
    combined_results = combine_individual_results()
    print_individual_checkpoint_accuracy(combined_results)
    statistics = create_statistics_tables(combined_results)
    print_l2_and_max_abs_statistics(statistics)
    create_total_loss_curve()
    create_representative_fields()
    create_representative_aat_histograms()
    print("\n" + "=" * 116)
    print(f"{METHOD_LABEL} analysis completed.")
    print(f"Output directory: {ANALYSIS_DIR.resolve()}")
    print("All figures are saved as 800-dpi JPG files.")
    print("=" * 116)
