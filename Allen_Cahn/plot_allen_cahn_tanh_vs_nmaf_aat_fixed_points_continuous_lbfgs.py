import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from allen_cahn_common_fixed_points_continuous_lbfgs import CHECKPOINT_EVALS, IMAGE_DPI, RESULT_ROOT

OUTPUT_DIR = RESULT_ROOT / "tanh_vs_nmaf_aat_comparison"

METHODS = {
    "tanh": {
        "label": "Standard PINN",
        "analysis_csv": "tanh_total_loss_log_statistics.csv",
        "color": "#1f77b4",
        "linestyle": "-",
        "zorder": 5,
    },
    "nmaf_aat": {
        "label": "NMAF-AAT",
        "analysis_csv": "nmaf_aat_total_loss_log_statistics.csv",
        "color": "#2ca02c",
        "linestyle": "--",
        "zorder": 6,
    },
}


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({"font.size": 12})


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


def load_loss_statistics(method_key):
    path = RESULT_ROOT / method_key / "analysis" / METHODS[method_key]["analysis_csv"]
    if not path.exists():
        raise FileNotFoundError(f"Missing loss statistics: {path}. Run method-specific analysis first.")
    return pd.read_csv(path)


def plot_total_loss_comparison():
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    exported = []
    for method_key, info in METHODS.items():
        stats = load_loss_statistics(method_key)
        method_export = stats.copy()
        method_export.insert(0, "method", method_key)
        method_export.insert(1, "method_label", info["label"])
        exported.append(method_export)
        ax.semilogy(stats["lbfgs_eval"], stats["geometric_mean_total_loss"], linewidth=1.8, color=info["color"], linestyle=info["linestyle"], label=info["label"], zorder=info["zorder"])
        ax.fill_between(stats["lbfgs_eval"], stats["lower_band"], stats["upper_band"], color=info["color"], alpha=0.14, linewidth=0.0, zorder=info["zorder"] - 1)
    pd.concat(exported, ignore_index=True).to_csv(OUTPUT_DIR / "allen_cahn_tanh_vs_nmaf_aat_total_loss_data.csv", index=False)
    for checkpoint_eval in CHECKPOINT_EVALS:
        ax.axvline(checkpoint_eval, color="0.35", linestyle="-.", linewidth=0.9, alpha=0.65)
    ax.set_xlabel("L-BFGS Function Evaluations")
    ax.set_ylabel("Total Loss")
    ax.grid(True, which="both", linestyle="-", alpha=0.5)
    ax.legend(frameon=True)
    ax.set_xlim(1, max(CHECKPOINT_EVALS))
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "allen_cahn_tanh_vs_nmaf_aat_total_loss.jpg"
    fig.savefig(figure_path, dpi=IMAGE_DPI, format="jpg", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return figure_path


def load_individual_results(method_key):
    path = RESULT_ROOT / method_key / "analysis" / "all_individual_runs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing individual results: {path}. Run method-specific analysis first.")
    data = pd.read_csv(path)
    data["method_key"] = method_key
    data["method_label"] = METHODS[method_key]["label"]
    return data


def calculate_error_statistics():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    individual = pd.concat([load_individual_results(method_key) for method_key in METHODS], ignore_index=True)
    individual = individual.sort_values(["checkpoint_eval", "method_key", "seed"]).reset_index(drop=True)
    individual.to_csv(OUTPUT_DIR / "allen_cahn_tanh_vs_nmaf_aat_individual_results.csv", index=False)
    rows = []
    for (method_key, checkpoint_eval), group in individual.groupby(["method_key", "checkpoint_eval"]):
        row = {
            "method_key": method_key,
            "method_label": METHODS[method_key]["label"],
            "checkpoint_eval": int(checkpoint_eval),
            "number_of_seeds": int(group["seed"].nunique()),
        }
        for metric in ["relative_l2_error", "maximum_absolute_error", "total_loss"]:
            mean_value, std_value, valid_n = mean_and_sample_std(group[metric])
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_sample_std"] = std_value
            row[f"{metric}_valid_n"] = valid_n
        rows.append(row)
    summary = pd.DataFrame(rows)
    method_order = {key: index for index, key in enumerate(METHODS.keys())}
    summary["method_order"] = summary["method_key"].map(method_order)
    summary = summary.sort_values(["checkpoint_eval", "method_order"]).drop(columns=["method_order"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "allen_cahn_tanh_vs_nmaf_aat_error_summary.csv", index=False)
    paper_ready = summary[["method_label", "checkpoint_eval", "number_of_seeds"]].copy()
    for metric in ["relative_l2_error", "maximum_absolute_error", "total_loss"]:
        paper_ready[metric] = [
            format_mean_std(mean_value, std_value)
            for mean_value, std_value in zip(summary[f"{metric}_mean"], summary[f"{metric}_sample_std"])
        ]
    paper_ready.to_csv(OUTPUT_DIR / "allen_cahn_tanh_vs_nmaf_aat_paper_ready_table.csv", index=False)
    return individual, summary, paper_ready


if __name__ == "__main__":
    loss_figure_path = plot_total_loss_comparison()
    individual, summary, paper_ready = calculate_error_statistics()
    print("=" * 116)
    print("Allen-Cahn Standard PINN vs NMAF-AAT comparison completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Loss figure: {loss_figure_path}")
    print("=" * 116)
    print("\nPaper-ready table:")
    print(paper_ready.to_string(index=False))
