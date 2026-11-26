from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "nlse_multiseed_results"

SEEDS = [99, 1234, 2026]


PREFERRED_RAR_DIRNAME = "rar_sampling_only_base10000_plus2000"


RAR_METHOD_DIR_OVERRIDE = None

OUTPUT_DIR_NAME = "rar_total_runtime_analysis"

IMAGE_DPI = 800
SHOW_FIGURES = True


FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TITLE_SIZE = 13
LEGEND_SIZE = 11
TICK_LABEL_SIZE = 12


def assert_cpu_only():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible != "":
        raise RuntimeError(
            "CPU-only guard failed: CUDA_VISIBLE_DEVICES must be an empty string."
        )


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


def mean_std(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if len(values) == 0:
        return np.nan, np.nan, 0
    if len(values) == 1:
        return float(values.iloc[0]), np.nan, 1

    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        int(len(values)),
    )


def fmt_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "NA"
    if pd.isna(std_value):
        return f"{mean_value:.6f}"
    return f"{mean_value:.6f} ± {std_value:.6f}"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_valid_rar_dir(path: Path):
    if not path.is_dir():
        return False

    found_any = False

    for seed in SEEDS:
        seed_dir = path / f"seed_{seed}"
        sampling = seed_dir / "sampling_stage_summary.json"
        formal = seed_dir / "formal_training_summary.json"

        if sampling.exists() and formal.exists():
            found_any = True

    return found_any


def locate_rar_dir():
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            f"Cannot find NLSE result root:\n{RESULT_ROOT.resolve()}"
        )

    if RAR_METHOD_DIR_OVERRIDE is not None:
        path = RESULT_ROOT / RAR_METHOD_DIR_OVERRIDE

        if not is_valid_rar_dir(path):
            raise FileNotFoundError(
                "RAR_METHOD_DIR_OVERRIDE is not a valid NLSE RAR directory:\n"
                f"{path}"
            )

        return path

    preferred = RESULT_ROOT / PREFERRED_RAR_DIRNAME

    if is_valid_rar_dir(preferred):
        return preferred

    candidates = [
        path
        for path in RESULT_ROOT.iterdir()
        if (
            path.is_dir()
            and "rar" in path.name.lower()
            and is_valid_rar_dir(path)
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No valid NLSE RAR result directory was found.\n"
            "Expected each available seed directory to contain:\n"
            "  sampling_stage_summary.json\n"
            "  formal_training_summary.json"
        )

    raise RuntimeError(
        "More than one valid NLSE RAR result directory was found.\n"
        "Set RAR_METHOD_DIR_OVERRIDE near the top of this script.\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def analyze_seed(rar_dir: Path, seed: int):
    seed_dir = rar_dir / f"seed_{seed}"

    sampling_path = seed_dir / "sampling_stage_summary.json"
    formal_path = seed_dir / "formal_training_summary.json"

    if not sampling_path.exists():
        raise FileNotFoundError(sampling_path)

    if not formal_path.exists():
        raise FileNotFoundError(formal_path)

    sampling = load_json(sampling_path)
    formal = load_json(formal_path)


    sampling_wall = float(
        sampling["sampling_stage_wall_seconds"]
    )

    adam_optimizer = float(
        sampling["adam_optimizer_seconds"]
    )

    rar_selection = float(
        sampling["rar_selection_seconds"]
    )


    lbfgs_wall = float(
        formal["lbfgs_wall_including_monitor_seconds"]
    )

    lbfgs_monitor = float(
        formal["lbfgs_monitor_seconds"]
    )

    lbfgs_core = float(
        formal["lbfgs_core_seconds_estimate"]
    )


    timing_values = {
        "sampling_stage_wall_seconds": sampling_wall,
        "sampling_adam_optimizer_seconds": adam_optimizer,
        "sampling_rar_selection_seconds": rar_selection,
        "lbfgs_wall_including_monitor_seconds": lbfgs_wall,
        "lbfgs_monitor_seconds": lbfgs_monitor,
        "lbfgs_core_seconds_estimate": lbfgs_core,
    }

    for key, value in timing_values.items():
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite timing for seed={seed}: {key}={value}"
            )
        if value < 0.0:
            raise ValueError(
                f"Negative timing for seed={seed}: {key}={value}"
            )


    stage_a_other_overhead = (
        sampling_wall
        - adam_optimizer
        - rar_selection
    )


    if stage_a_other_overhead < 0.0 and abs(stage_a_other_overhead) < 1.0e-8:
        stage_a_other_overhead = 0.0


    total_training = (
        sampling_wall
        + lbfgs_core
    )


    component_core = (
        adam_optimizer
        + rar_selection
        + lbfgs_core
    )


    literal_end_to_end = (
        sampling_wall
        + lbfgs_wall
    )


    saved_end_to_end = float(
        formal.get(
            "end_to_end_cost_seconds",
            np.nan,
        )
    )

    end_to_end_difference = (
        literal_end_to_end
        - saved_end_to_end
        if np.isfinite(saved_end_to_end)
        else np.nan
    )

    return {
        "seed": int(seed),


        "sampling_adam_optimizer_seconds": adam_optimizer,
        "sampling_rar_selection_seconds": rar_selection,
        "sampling_stage_other_overhead_seconds": stage_a_other_overhead,
        "sampling_stage_wall_seconds": sampling_wall,


        "lbfgs_core_seconds": lbfgs_core,
        "lbfgs_monitor_seconds": lbfgs_monitor,
        "lbfgs_wall_including_monitor_seconds": lbfgs_wall,


        "component_core_seconds": component_core,
        "total_training_seconds": total_training,
        "literal_end_to_end_wall_seconds": literal_end_to_end,


        "saved_end_to_end_cost_seconds": saved_end_to_end,
        "literal_minus_saved_end_to_end_seconds": end_to_end_difference,

        "sampling_summary_file": str(sampling_path),
        "formal_summary_file": str(formal_path),
    }


def build_tables(rar_dir: Path, out_dir: Path):
    rows = []
    skipped = []

    print("\n" + "=" * 112)
    print("NLSE RAR total-runtime analysis — CPU only")
    print("=" * 112)
    print(f"RAR directory : {rar_dir.resolve()}")
    print(f"Seeds         : {SEEDS}")
    print(
        "Primary total : sampling_stage_wall_seconds "
        "+ lbfgs_core_seconds_estimate"
    )
    print("=" * 112)

    for seed in SEEDS:
        try:
            row = analyze_seed(
                rar_dir,
                seed,
            )

            rows.append(row)

            print(
                f"seed={seed:4d} | "
                f"Stage-A wall={row['sampling_stage_wall_seconds']:.3f}s | "
                f"L-BFGS core={row['lbfgs_core_seconds']:.3f}s | "
                f"TOTAL={row['total_training_seconds']:.3f}s"
            )

        except Exception as exc:
            skipped.append({
                "seed": int(seed),
                "reason": repr(exc),
            })

            print(
                f"SKIPPED seed={seed:4d} | {exc}"
            )

    if not rows:
        raise RuntimeError(
            "No usable NLSE RAR timing results were found."
        )

    individual = pd.DataFrame(rows)

    individual.to_csv(
        out_dir / "nlse_rar_total_runtime_seedwise.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(skipped).to_csv(
        out_dir / "nlse_rar_total_runtime_skipped.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = [
        "sampling_adam_optimizer_seconds",
        "sampling_rar_selection_seconds",
        "sampling_stage_other_overhead_seconds",
        "sampling_stage_wall_seconds",
        "lbfgs_core_seconds",
        "lbfgs_monitor_seconds",
        "lbfgs_wall_including_monitor_seconds",
        "component_core_seconds",
        "total_training_seconds",
        "literal_end_to_end_wall_seconds",
    ]

    summary = {
        "number_of_available_seeds": int(
            len(individual)
        ),
        "available_seeds": ",".join(
            str(int(seed))
            for seed in individual["seed"]
        ),
    }

    for metric in metrics:
        mean_value, std_value, n = mean_std(
            individual[metric]
        )

        summary[
            f"{metric}_mean"
        ] = mean_value

        summary[
            f"{metric}_sample_std"
        ] = std_value

        summary[
            f"{metric}_valid_n"
        ] = n

        summary[
            f"{metric}_mean_plus_minus_sample_std"
        ] = fmt_mean_std(
            mean_value,
            std_value,
        )

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        out_dir / "nlse_rar_total_runtime_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paper = pd.DataFrame([{
        "Method": "RAR-PINN",
        "Seeds": summary[
            "available_seeds"
        ],
        "Stage-A Adam optimizer time (s)": summary[
            "sampling_adam_optimizer_seconds_mean_plus_minus_sample_std"
        ],
        "Stage-A RAR selection time (s)": summary[
            "sampling_rar_selection_seconds_mean_plus_minus_sample_std"
        ],
        "Stage-A complete wall time (s)": summary[
            "sampling_stage_wall_seconds_mean_plus_minus_sample_std"
        ],
        "L-BFGS core training time (s)": summary[
            "lbfgs_core_seconds_mean_plus_minus_sample_std"
        ],
        "RAR total training cost (s)": summary[
            "total_training_seconds_mean_plus_minus_sample_std"
        ],
        "Literal end-to-end wall time (s)": summary[
            "literal_end_to_end_wall_seconds_mean_plus_minus_sample_std"
        ],
    }])

    paper.to_csv(
        out_dir / "nlse_rar_total_runtime_paper_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return (
        individual,
        summary_df,
        paper,
    )


def plot_runtime_breakdown(
    individual: pd.DataFrame,
    out_dir: Path,
):
    configure_plot_style()

    adam_mean = float(
        individual[
            "sampling_adam_optimizer_seconds"
        ].mean()
    )

    selection_mean = float(
        individual[
            "sampling_rar_selection_seconds"
        ].mean()
    )

    overhead_mean = float(
        individual[
            "sampling_stage_other_overhead_seconds"
        ].mean()
    )

    lbfgs_mean = float(
        individual[
            "lbfgs_core_seconds"
        ].mean()
    )

    total_values = pd.to_numeric(
        individual[
            "total_training_seconds"
        ],
        errors="coerce",
    ).dropna()

    total_mean = float(
        total_values.mean()
    )

    total_std = (
        float(
            total_values.std(
                ddof=1
            )
        )
        if len(total_values) > 1
        else 0.0
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    x = np.array([0.0])
    width = 0.62
    bottom = 0.0

    parts = [
        (
            "Stage-A Adam optimizer",
            adam_mean,
        ),
        (
            "Stage-A RAR selection",
            selection_mean,
        ),
        (
            "Stage-A other overhead",
            overhead_mean,
        ),
        (
            "Stage-B L-BFGS core",
            lbfgs_mean,
        ),
    ]

    for label, value in parts:
        ax.bar(
            x,
            [value],
            width=width,
            bottom=[bottom],
            label=label,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.80,
        )

        bottom += value

    ax.errorbar(
        x,
        [total_mean],
        yerr=[total_std],
        fmt="none",
        capsize=5,
        linewidth=1.2,
        zorder=5,
    )

    seed_offsets = np.linspace(
        -0.12,
        0.12,
        len(individual),
    )

    for offset, (_, row) in zip(
        seed_offsets,
        individual.iterrows(),
    ):
        ax.scatter(
            [x[0] + offset],
            [
                row[
                    "total_training_seconds"
                ]
            ],
            s=38,
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
            label=f"seed {int(row['seed'])}",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["RAR-PINN"]
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    ax.set_title(
        "NLSE RAR total runtime: point acquisition + L-BFGS training"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.35,
    )

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    unique = {}

    for handle, label in zip(
        handles,
        labels,
    ):
        if label not in unique:
            unique[label] = handle

    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=True,
        fontsize=9,
    )

    ax.annotate(
        f"{total_mean:.2f} ± {total_std:.2f} s",
        xy=(
            x[0],
            total_mean,
        ),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        ),
    )

    fig.tight_layout()

    path = (
        out_dir
        / "nlse_rar_total_runtime_breakdown.png"
    )

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="png",
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)

    return path


def main():
    assert_cpu_only()

    rar_dir = locate_rar_dir()

    out_dir = (
        rar_dir
        / OUTPUT_DIR_NAME
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        individual,
        summary,
        paper,
    ) = build_tables(
        rar_dir,
        out_dir,
    )

    figure_path = (
        plot_runtime_breakdown(
            individual,
            out_dir,
        )
    )

    row = summary.iloc[0]

    print("\n" + "=" * 112)
    print("FINAL NLSE RAR RUNTIME STATISTICS")
    print("=" * 112)

    print(
        "Stage-A Adam optimizer       : "
        + row[
            "sampling_adam_optimizer_seconds_mean_plus_minus_sample_std"
        ]
        + " s"
    )

    print(
        "Stage-A RAR selection        : "
        + row[
            "sampling_rar_selection_seconds_mean_plus_minus_sample_std"
        ]
        + " s"
    )

    print(
        "Stage-A complete wall time   : "
        + row[
            "sampling_stage_wall_seconds_mean_plus_minus_sample_std"
        ]
        + " s"
    )

    print(
        "Stage-B L-BFGS core training : "
        + row[
            "lbfgs_core_seconds_mean_plus_minus_sample_std"
        ]
        + " s"
    )

    print("-" * 112)

    print(
        "RAR TOTAL TRAINING COST      : "
        + row[
            "total_training_seconds_mean_plus_minus_sample_std"
        ]
        + " s"
    )

    print(
        "Literal end-to-end wall time : "
        + row[
            "literal_end_to_end_wall_seconds_mean_plus_minus_sample_std"
        ]
        + " s"
    )

    print("=" * 112)

    print(
        "\nRecommended quantity for cross-method computational-cost comparison:"
    )

    print(
        "sampling_stage_wall_seconds "
        "+ lbfgs_core_seconds_estimate"
    )

    print(
        "= complete RAR point-acquisition stage "
        "+ true L-BFGS training"
    )

    print(
        "\nThe analysis itself is CPU-only and does not load model checkpoints."
    )

    print(
        "Output directory:",
        out_dir.resolve(),
    )

    print(
        "Figure:",
        figure_path,
    )


if __name__ == "__main__":
    main()
