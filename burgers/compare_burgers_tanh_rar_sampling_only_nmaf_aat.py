import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULT_ROOT = BASE_DIR / "burgers_multiseed_results"
REFERENCE_FILE = BASE_DIR / "burgers_shock.mat"
ANALYSIS_DIR = RESULT_ROOT / "four_method_unified_analysis"

from burgers_common_fixed_points_continuous_lbfgs import (
    SEEDS,
    RESULT_ROOT,
    REFERENCE_FILE,
    CHECKPOINT_EVALS,
    IMAGE_DPI,
)

from burgers_rar_sampling_only_common import (
    METHOD,
    METHOD_LABEL,
)


OUTPUT_DIR = (
    RESULT_ROOT
    / "tanh_vs_rar_sampling_only_vs_nmaf_aat"
)

METHODS = {
    "tanh": {
        "label": "Standard PINN",
        "loss_csv": (
            RESULT_ROOT
            / "tanh"
            / "analysis"
            / "tanh_total_loss_log_statistics.csv"
        ),
        "individual_csv": (
            RESULT_ROOT
            / "tanh"
            / "analysis"
            / "all_individual_runs.csv"
        ),
        "prediction_prefix": "burgers_tanh",
    },

    METHOD: {
        "label": METHOD_LABEL,
        "loss_csv": (
            RESULT_ROOT
            / METHOD
            / "analysis"
            / (
                f"{METHOD}"
                f"_total_loss_log_statistics.csv"
            )
        ),
        "individual_csv": (
            RESULT_ROOT
            / METHOD
            / "analysis"
            / "all_individual_runs.csv"
        ),
        "prediction_prefix": (
            f"burgers_{METHOD}"
        ),
    },

    "nmaf_aat": {
        "label": "NMAF-AAT",
        "loss_csv": (
            RESULT_ROOT
            / "nmaf_aat"
            / "analysis"
            / "nmaf_aat_total_loss_log_statistics.csv"
        ),
        "individual_csv": (
            RESULT_ROOT
            / "nmaf_aat"
            / "analysis"
            / "all_individual_runs.csv"
        ),
        "prediction_prefix": (
            "burgers_nmaf_aat"
        ),
    },
}


def configure_plot_style():
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.size": 12,
        "lines.linewidth": 1.8,
    })


def load_reference_arrays():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            REFERENCE_FILE
        )

    data = scipy.io.loadmat(
        REFERENCE_FILE
    )

    x = np.asarray(data["x"]).flatten()
    t = np.asarray(data["t"]).flatten()
    exact = np.asarray(data["usol"])

    target_shape = (
        len(x),
        len(t),
    )

    if exact.shape != target_shape:
        if exact.T.shape == target_shape:
            exact = exact.T
        else:
            raise ValueError(
                f"Reference shape={exact.shape}, "
                f"expected {target_shape}."
            )

    return x, t, exact


def prediction_path(
    method_key,
    seed,
    checkpoint_eval,
):
    info = METHODS[
        method_key
    ]

    return (
        RESULT_ROOT
        / method_key
        / f"seed_{seed}"
        / (
            f"{info['prediction_prefix']}"
            f"_seed{seed}"
            f"_prediction_eval{checkpoint_eval:05d}.npz"
        )
    )


def verify_identical_l2_grid():
    x_ref, t_ref, exact_ref = (
        load_reference_arrays()
    )

    rows = []

    for method_key in METHODS:
        for seed in SEEDS:
            for checkpoint_eval in CHECKPOINT_EVALS:
                path = prediction_path(
                    method_key,
                    seed,
                    checkpoint_eval,
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing prediction file: {path}. "
                        "Run the corresponding analysis first."
                    )

                data = np.load(path)

                x = np.asarray(
                    data["x"]
                ).reshape(-1)

                t = np.asarray(
                    data["t"]
                ).reshape(-1)

                exact = np.asarray(
                    data["u_exact"]
                )

                if not np.array_equal(
                    x,
                    x_ref,
                ):
                    raise RuntimeError(
                        f"x-grid mismatch: {path}"
                    )

                if not np.array_equal(
                    t,
                    t_ref,
                ):
                    raise RuntimeError(
                        f"t-grid mismatch: {path}"
                    )

                if not np.array_equal(
                    exact,
                    exact_ref,
                ):
                    raise RuntimeError(
                        f"u_exact mismatch: {path}"
                    )

                rows.append({
                    "method": method_key,
                    "seed": int(seed),
                    "checkpoint_eval": int(
                        checkpoint_eval
                    ),
                    "identical_x": True,
                    "identical_t": True,
                    "identical_u_exact": True,
                })

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_DIR
        / "verified_identical_l2_evaluation_grid.csv",
        index=False,
    )

    print(
        "PASS: all three methods use identical "
        "x/t/u_exact arrays for L2."
    )


def plot_common_loss():
    fig, ax = plt.subplots(
        figsize=(8.5, 5.3)
    )

    exports = []

    for method_key, info in METHODS.items():
        if not info[
            "loss_csv"
        ].exists():
            raise FileNotFoundError(
                info[
                    "loss_csv"
                ]
            )

        stats = pd.read_csv(
            info[
                "loss_csv"
            ]
        )

        export = stats.copy()
        export.insert(
            0,
            "method",
            method_key,
        )
        export.insert(
            1,
            "method_label",
            info[
                "label"
            ],
        )
        exports.append(export)

        line, = ax.plot(
            stats[
                "lbfgs_eval"
            ],
            stats[
                "geometric_mean_total_loss"
            ],
            label=info[
                "label"
            ],
        )

        ax.fill_between(
            stats[
                "lbfgs_eval"
            ],
            stats[
                "lower_band"
            ],
            stats[
                "upper_band"
            ],
            color=line.get_color(),
            alpha=0.14,
            linewidth=0.0,
        )

    pd.concat(
        exports,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR
        / "burgers_three_method_common_loss_data.csv",
        index=False,
    )

    for checkpoint_eval in CHECKPOINT_EVALS:
        ax.axvline(
            checkpoint_eval,
            color="0.35",
            linestyle="-.",
            linewidth=0.9,
            alpha=0.65,
        )

    ax.set_yscale("log")
    ax.set_xlabel(
        "L-BFGS Function Evaluations"
    )
    ax.set_ylabel(
        "Total Loss on Original Fixed Points"
    )
    ax.set_xlim(
        1,
        max(CHECKPOINT_EVALS),
    )
    ax.grid(
        True,
        which="both",
        alpha=0.5,
    )
    ax.legend(frameon=True)

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "burgers_tanh_rar_sampling_only_nmaf_aat_loss.jpg"
    )

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="jpg",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return path


def mean_and_sample_std(values):
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan, np.nan, 0

    if len(values) == 1:
        return float(values.iloc[0]), np.nan, 1

    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        int(len(values)),
    )


def format_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "NA"

    if pd.isna(std_value):
        return f"{mean_value:.6e}"

    return f"{mean_value:.6e} ± {std_value:.6e}"


def build_comparison_tables():
    frames = []

    for method_key, info in METHODS.items():
        path = info[
            "individual_csv"
        ]

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        frame = pd.read_csv(path)
        frame[
            "method_key"
        ] = method_key
        frame[
            "method_label"
        ] = info[
            "label"
        ]

        frames.append(frame)

    individual = pd.concat(
        frames,
        ignore_index=True,
    )

    individual.to_csv(
        OUTPUT_DIR
        / "burgers_three_method_individual_results.csv",
        index=False,
    )

    rows = []

    for (
        method_key,
        checkpoint_eval,
    ), group in individual.groupby(
        [
            "method_key",
            "checkpoint_eval",
        ]
    ):
        row = {
            "method_key": method_key,
            "method_label": METHODS[
                method_key
            ][
                "label"
            ],
            "checkpoint_eval": int(
                checkpoint_eval
            ),
            "number_of_seeds": int(
                group[
                    "seed"
                ].nunique()
            ),
        }

        for metric in [
            "relative_l2_error",
            "maximum_absolute_error",
            "total_loss",
            "lbfgs_time_seconds",
        ]:
            mean_value, std_value, valid_n = (
                mean_and_sample_std(
                    group[
                        metric
                    ]
                )
            )

            row[
                f"{metric}_mean"
            ] = mean_value
            row[
                f"{metric}_sample_std"
            ] = std_value
            row[
                f"{metric}_valid_n"
            ] = valid_n

        rows.append(row)

    summary = pd.DataFrame(rows)

    order = {
        "tanh": 0,
        METHOD: 1,
        "nmaf_aat": 2,
    }

    summary[
        "_order"
    ] = summary[
        "method_key"
    ].map(order)

    summary = summary.sort_values(
        [
            "checkpoint_eval",
            "_order",
        ]
    ).drop(
        columns="_order"
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        OUTPUT_DIR
        / "burgers_three_method_error_summary.csv",
        index=False,
    )

    paper = summary[
        [
            "method_label",
            "checkpoint_eval",
            "number_of_seeds",
        ]
    ].copy()

    for metric in [
        "relative_l2_error",
        "maximum_absolute_error",
        "total_loss",
        "lbfgs_time_seconds",
    ]:
        paper[
            metric
        ] = [
            format_mean_std(
                mean_value,
                std_value,
            )
            for mean_value, std_value in zip(
                summary[
                    f"{metric}_mean"
                ],
                summary[
                    f"{metric}_sample_std"
                ],
            )
        ]

    paper.to_csv(
        OUTPUT_DIR
        / "burgers_three_method_paper_ready_table.csv",
        index=False,
    )

    return (
        individual,
        summary,
        paper,
    )


def build_sampling_cost_table():
    rows = []

    for seed in SEEDS:
        seed_dir = (
            RESULT_ROOT
            / METHOD
            / f"seed_{seed}"
        )

        sampling = json.loads(
            (
                seed_dir
                / "sampling_stage_summary.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        formal = json.loads(
            (
                seed_dir
                / "formal_training_summary.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        rows.append({
            "seed": int(seed),
            "sampling_stage_wall_seconds": float(
                sampling[
                    "sampling_stage_wall_seconds"
                ]
            ),
            "sampling_adam_optimizer_seconds": float(
                sampling[
                    "adam_optimizer_seconds"
                ]
            ),
            "sampling_rar_selection_seconds": float(
                sampling[
                    "rar_selection_seconds"
                ]
            ),
            "formal_lbfgs_core_seconds": float(
                formal[
                    "lbfgs_core_seconds_estimate"
                ]
            ),
            "formal_lbfgs_wall_including_monitor_seconds": float(
                formal[
                    "lbfgs_wall_including_monitor_seconds"
                ]
            ),
            "end_to_end_cost_seconds": float(
                formal[
                    "end_to_end_cost_seconds"
                ]
            ),
        })

    frame = pd.DataFrame(rows)

    frame.to_csv(
        OUTPUT_DIR
        / "rar_sampling_cost_individual.csv",
        index=False,
    )

    return frame


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_plot_style()

    verify_identical_l2_grid()

    loss_path = plot_common_loss()

    (
        individual,
        summary,
        paper,
    ) = build_comparison_tables()

    sampling_cost = (
        build_sampling_cost_table()
    )

    print("=" * 122)
    print(
        "Standard PINN vs sampling-only RAR vs NMAF-AAT"
    )
    print("=" * 122)
    print(
        "Formal optimizer for all methods: "
        "fresh/same-seed continuous L-BFGS5000"
    )
    print(
        "RAR Adam stage is sampling design only; "
        "its model information is discarded."
    )
    print(
        f"Loss figure: {loss_path}"
    )
    print("-" * 122)
    print(
        paper.to_string(
            index=False
        )
    )
    print("=" * 122)
