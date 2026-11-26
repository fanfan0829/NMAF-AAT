import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nlse_common_fixed_points_continuous_lbfgs import (
    SEEDS,
    REPRESENTATIVE_SEED,
    RESULT_ROOT,
    DEVICE,
    DTYPE,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
    CHECKPOINT_EVALS,
    IMAGE_DPI,
)

from nlse_rar_sampling_only_common import (
    METHOD,
    METHOD_LABEL,
    METHOD_DIR,
    FINAL_N_RES,
    build_fresh_tanh_model,
    load_reference,
    predict,
    relative_l2,
    selected_points_path,
)


ANALYSIS_DIR = (
    METHOD_DIR
    / "analysis"
)


def configure_plot_style():
    plt.rcParams[
        "mathtext.fontset"
    ] = "custom"

    plt.rcParams[
        "mathtext.rm"
    ] = "Times New Roman"

    plt.rcParams[
        "mathtext.it"
    ] = "Times New Roman:italic"

    plt.rcParams[
        "font.family"
    ] = "Times New Roman"

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.6,
    })


def mean_and_sample_std(
    values,
):
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(
        values
    ) == 0:
        return (
            np.nan,
            np.nan,
            0,
        )

    if len(
        values
    ) == 1:
        return (
            float(
                values.iloc[
                    0
                ]
            ),
            np.nan,
            1,
        )

    return (
        float(
            values.mean()
        ),
        float(
            values.std(
                ddof=1
            )
        ),
        int(
            len(
                values
            )
        ),
    )


def format_mean_std(
    mean_value,
    std_value,
):
    if pd.isna(
        mean_value
    ):
        return "NA"

    if pd.isna(
        std_value
    ):
        return (
            f"{mean_value:.6e}"
        )

    return (
        f"{mean_value:.6e} "
        f"± {std_value:.6e}"
    )


def checkpoint_path(
    seed,
    checkpoint_eval,
):
    return (
        METHOD_DIR
        / f"seed_{seed}"
        / (
            f"nlse_{METHOD}"
            f"_seed{seed}"
            f"_eval{checkpoint_eval:05d}.pth"
        )
    )


def load_checkpoint(
    seed,
    checkpoint_eval,
):
    path = checkpoint_path(
        seed,
        checkpoint_eval,
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = {
        key: (
            value.double()
            if value.dtype
            == torch.float32
            else value
        )
        for key, value
        in checkpoint[
            "model_state_dict"
        ].items()
    }

    return (
        checkpoint,
        state_dict,
    )


def rebuild_individual_results():
    (
        x,
        t,
        X,
        T,
        h_exact,
    ) = load_reference()

    rows = []

    for seed in SEEDS:
        seed_dir = (
            METHOD_DIR
            / f"seed_{seed}"
        )

        monitor = pd.read_csv(
            seed_dir
            / "fixed_monitor_history.csv"
        )

        for checkpoint_eval in CHECKPOINT_EVALS:
            (
                checkpoint,
                state_dict,
            ) = load_checkpoint(
                seed,
                checkpoint_eval,
            )

            model = (
                build_fresh_tanh_model()
            )

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            (
                h_pred,
                u_pred,
                v_pred,
            ) = predict(
                model,
                X,
                T,
                h_exact.shape,
            )

            complex_abs_error = np.abs(
                h_exact
                - h_pred
            )

            magnitude_error = np.abs(
                np.abs(
                    h_exact
                )
                - np.abs(
                    h_pred
                )
            )

            actual_eval = int(
                checkpoint.get(
                    "actual_lbfgs_eval",
                    checkpoint.get(
                        "lbfgs_eval",
                        checkpoint_eval,
                    ),
                )
            )

            selected = monitor.loc[
                monitor[
                    "lbfgs_eval"
                ]
                == actual_eval
            ]

            if selected.empty:
                raise ValueError(
                    f"No RAR fixed-monitor row for "
                    f"seed={seed}, actual_eval={actual_eval}."
                )

            selected = (
                selected.iloc[
                    -1
                ]
            )

            prediction_name = (
                f"nlse_{METHOD}"
                f"_seed{seed}"
                f"_prediction_eval{checkpoint_eval:05d}.npz"
            )

            np.savez_compressed(
                seed_dir
                / prediction_name,
                x=x,
                t=t,
                h_exact=h_exact,
                h_pred=h_pred,
                u_pred=u_pred,
                v_pred=v_pred,
                complex_abs_error=complex_abs_error,
                magnitude_error=magnitude_error,
            )

            latest = checkpoint[
                "latest_metrics"
            ]

            rows.append({
                "method": METHOD,
                "method_label": METHOD_LABEL,
                "seed": int(
                    seed
                ),
                "checkpoint_eval": int(
                    checkpoint_eval
                ),
                "actual_lbfgs_eval": int(
                    actual_eval
                ),


                "total_loss": float(
                    selected[
                        "fixed_monitor_total_loss"
                    ]
                ),
                "residual_loss": float(
                    selected[
                        "fixed_monitor_residual_loss"
                    ]
                ),
                "data_loss": float(
                    selected[
                        "fixed_monitor_data_loss"
                    ]
                ),
                "bc_loss": float(
                    selected[
                        "fixed_monitor_bc_loss"
                    ]
                ),
                "ic_loss_unweighted": float(
                    selected[
                        "fixed_monitor_ic_loss_unweighted"
                    ]
                ),
                "ic_loss_weighted": float(
                    selected[
                        "fixed_monitor_ic_loss_weighted"
                    ]
                ),


                "optimization_total_loss": float(
                    latest[
                        "optimization_total_loss"
                    ]
                ),

                "relative_l2_error": float(
                    relative_l2(
                        h_exact,
                        h_pred,
                    )
                ),
                "maximum_absolute_error": float(
                    np.max(
                        complex_abs_error
                    )
                ),
                "maximum_magnitude_absolute_error": float(
                    np.max(
                        magnitude_error
                    )
                ),
                "h_exact_abs_min": float(
                    np.min(
                        np.abs(
                            h_exact
                        )
                    )
                ),
                "h_exact_abs_max": float(
                    np.max(
                        np.abs(
                            h_exact
                        )
                    )
                ),
                "h_pred_abs_min": float(
                    np.min(
                        np.abs(
                            h_pred
                        )
                    )
                ),
                "h_pred_abs_max": float(
                    np.max(
                        np.abs(
                            h_pred
                        )
                    )
                ),
                "lbfgs_time_seconds": float(
                    checkpoint[
                        "core_lbfgs_time_seconds_estimate"
                    ]
                ),
                "lbfgs_wall_including_monitor_seconds": float(
                    checkpoint[
                        "elapsed_seconds"
                    ]
                ),
                "prediction_file": (
                    prediction_name
                ),
            })

    combined = pd.DataFrame(
        rows
    ).sort_values(
        [
            "checkpoint_eval",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    combined.to_csv(
        ANALYSIS_DIR
        / "all_individual_runs.csv",
        index=False,
    )

    return combined


def create_statistics(
    combined,
):
    metrics = [
        "total_loss",
        "optimization_total_loss",
        "residual_loss",
        "data_loss",
        "relative_l2_error",
        "maximum_absolute_error",
        "maximum_magnitude_absolute_error",
        "lbfgs_time_seconds",
    ]

    rows = []

    for (
        checkpoint_eval,
        group,
    ) in combined.groupby(
        "checkpoint_eval"
    ):
        row = {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "checkpoint_eval": int(
                checkpoint_eval
            ),
            "number_of_seeds": int(
                group[
                    "seed"
                ].nunique()
            ),
        }

        for metric in metrics:
            (
                mean_value,
                std_value,
                valid_n,
            ) = mean_and_sample_std(
                group[
                    metric
                ]
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

        rows.append(
            row
        )

    stats = pd.DataFrame(
        rows
    )

    stats.to_csv(
        ANALYSIS_DIR
        / "mean_std_statistics.csv",
        index=False,
    )

    paper_rows = []

    for _, row in stats.iterrows():
        paper_row = {
            "Method": row[
                "method_label"
            ],
            "Checkpoint": int(
                row[
                    "checkpoint_eval"
                ]
            ),
            "Seeds": int(
                row[
                    "number_of_seeds"
                ]
            ),
        }

        for metric in metrics:
            paper_row[
                metric
            ] = format_mean_std(
                row[
                    f"{metric}_mean"
                ],
                row[
                    f"{metric}_sample_std"
                ],
            )

        paper_rows.append(
            paper_row
        )

    pd.DataFrame(
        paper_rows
    ).to_csv(
        ANALYSIS_DIR
        / "paper_ready_mean_std_table.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 122
    )

    print(
        "NLSE RAR sampling-only accuracy statistics"
    )

    print(
        "=" * 122
    )

    for _, row in stats.iterrows():
        print(
            f"Eval {int(row['checkpoint_eval']):5d} | "
            f"L2 = "
            f"{format_mean_std(row['relative_l2_error_mean'], row['relative_l2_error_sample_std'])} | "
            f"Max abs = "
            f"{format_mean_std(row['maximum_absolute_error_mean'], row['maximum_absolute_error_sample_std'])} | "
            f"Max magnitude abs = "
            f"{format_mean_std(row['maximum_magnitude_absolute_error_mean'], row['maximum_magnitude_absolute_error_sample_std'])}"
        )

    print(
        "=" * 122
    )

    return stats


def create_common_total_loss_curve():


    merged = None

    for seed in SEEDS:
        frame = pd.read_csv(
            METHOD_DIR
            / f"seed_{seed}"
            / "fixed_monitor_history.csv"
        )[
            [
                "lbfgs_eval",
                "fixed_monitor_total_loss",
            ]
        ].copy()

        frame = frame.drop_duplicates(
            subset="lbfgs_eval",
            keep="last",
        ).rename(
            columns={
                "fixed_monitor_total_loss":
                f"total_loss_seed_{seed}"
            }
        )

        merged = (
            frame
            if merged is None
            else merged.merge(
                frame,
                on="lbfgs_eval",
                how="inner",
            )
        )

    if (
        merged is None
        or merged.empty
    ):
        raise RuntimeError(
            "No common RAR monitor loss indices."
        )

    columns = [
        f"total_loss_seed_{seed}"
        for seed in SEEDS
    ]

    values = merged[
        columns
    ].to_numpy(
        dtype=float
    )

    if (
        np.any(
            values <= 0.0
        )
        or np.any(
            ~np.isfinite(
                values
            )
        )
    ):
        raise ValueError(
            "RAR fixed-monitor loss contains invalid values."
        )

    log_values = np.log10(
        values
    )

    log_mean = np.mean(
        log_values,
        axis=1,
    )

    log_std = np.std(
        log_values,
        axis=1,
        ddof=1,
    )

    curve = pd.DataFrame({
        "lbfgs_eval": merged[
            "lbfgs_eval"
        ].to_numpy(
            dtype=int
        ),
        "geometric_mean_total_loss": (
            10.0 ** log_mean
        ),
        "log10_sample_std": (
            log_std
        ),
        "lower_band": (
            10.0
            ** (
                log_mean
                - log_std
            )
        ),
        "upper_band": (
            10.0
            ** (
                log_mean
                + log_std
            )
        ),
    })

    for seed in SEEDS:
        curve[
            f"total_loss_seed_{seed}"
        ] = merged[
            f"total_loss_seed_{seed}"
        ].to_numpy(
            dtype=float
        )

    curve.to_csv(
        ANALYSIS_DIR
        / (
            f"{METHOD}"
            f"_total_loss_log_statistics.csv"
        ),
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    line, = ax.plot(
        curve[
            "lbfgs_eval"
        ],
        curve[
            "geometric_mean_total_loss"
        ],
        label=METHOD_LABEL,
    )

    ax.fill_between(
        curve[
            "lbfgs_eval"
        ],
        curve[
            "lower_band"
        ],
        curve[
            "upper_band"
        ],
        color=line.get_color(),
        alpha=0.16,
        linewidth=0.0,
    )

    for checkpoint_eval in CHECKPOINT_EVALS:
        ax.axvline(
            checkpoint_eval,
            color="0.35",
            linestyle="-.",
            linewidth=0.9,
            alpha=0.65,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "L-BFGS Function Evaluations"
    )

    ax.set_ylabel(
        "Total Loss on Original Fixed Points"
    )

    ax.set_xlim(
        1,
        max(
            CHECKPOINT_EVALS
        ),
    )

    ax.grid(
        True,
        which="both",
        linestyle="-",
        alpha=0.5,
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    path = (
        ANALYSIS_DIR
        / "nlse_rar_sampling_only_common_loss.jpg"
    )

    fig.savefig(
        path,
        dpi=IMAGE_DPI,
        format="jpg",
        bbox_inches="tight",
    )

    plt.show()

    plt.close(
        fig
    )

    return curve


def create_representative_fields():
    for checkpoint_eval in CHECKPOINT_EVALS:
        path = (
            METHOD_DIR
            / f"seed_{REPRESENTATIVE_SEED}"
            / (
                f"nlse_{METHOD}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_prediction_eval{checkpoint_eval:05d}.npz"
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        data = np.load(
            path
        )

        x = data[
            "x"
        ]

        t = data[
            "t"
        ]

        h_exact = data[
            "h_exact"
        ]

        h_pred = data[
            "h_pred"
        ]

        magnitude_error = data[
            "magnitude_error"
        ]

        X, T = np.meshgrid(
            x,
            t,
            indexing="ij",
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(
                18,
                5,
            ),
        )

        exact_mag = np.abs(
            h_exact
        )

        pred_mag = np.abs(
            h_pred
        )

        h_min = float(
            np.min(
                exact_mag
            )
        )

        h_max = float(
            np.max(
                exact_mag
            )
        )

        im0 = axes[
            0
        ].pcolor(
            T,
            X,
            exact_mag,
            cmap="jet",
            shading="auto",
            vmin=h_min,
            vmax=h_max,
        )

        im1 = axes[
            1
        ].pcolor(
            T,
            X,
            pred_mag,
            cmap="jet",
            shading="auto",
            vmin=h_min,
            vmax=h_max,
        )

        im2 = axes[
            2
        ].pcolor(
            T,
            X,
            magnitude_error,
            cmap="Reds",
            shading="auto",
        )

        axes[
            0
        ].set_title(
            r"Reference $|h(t,x)|$"
        )

        axes[
            1
        ].set_title(
            rf"{METHOD_LABEL} prediction $|h(t,x)|$"
        )

        axes[
            2
        ].set_title(
            r"Absolute magnitude error"
        )

        for (
            axis,
            image,
        ) in zip(
            axes,
            [
                im0,
                im1,
                im2,
            ],
        ):
            axis.set_xlabel(
                "t"
            )

            axis.set_ylabel(
                "x"
            )

            fig.colorbar(
                image,
                ax=axis,
            )

        fig.tight_layout()

        figure_path = (
            ANALYSIS_DIR
            / (
                f"nlse_{METHOD}"
                f"_seed{REPRESENTATIVE_SEED}"
                f"_field_eval{checkpoint_eval:05d}.jpg"
            )
        )

        fig.savefig(
            figure_path,
            dpi=IMAGE_DPI,
            format="jpg",
            bbox_inches="tight",
        )

        plt.show()

        plt.close(
            fig
        )


def plot_selected_points():
    seed = REPRESENTATIVE_SEED

    data = np.load(
        selected_points_path(
            seed
        )
    )

    x = data[
        "x_res"
    ].reshape(
        -1
    )

    t = data[
        "t_res"
    ].reshape(
        -1
    )

    source = data[
        "source_round"
    ].reshape(
        -1
    )

    if len(
        x
    ) != FINAL_N_RES:
        raise RuntimeError(
            f"Selected point count={len(x)}, "
            f"expected {FINAL_N_RES}."
        )

    original = (
        source == 0
    )

    adaptive = (
        source > 0
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    ax.scatter(
        t[
            original
        ],
        x[
            original
        ],
        s=3,
        alpha=0.20,
        label=(
            f"Original shared points "
            f"({int(original.sum())})"
        ),
    )

    scatter = ax.scatter(
        t[
            adaptive
        ],
        x[
            adaptive
        ],
        c=source[
            adaptive
        ],
        cmap="viridis",
        s=6,
        alpha=0.75,
        label=(
            f"RAR-added points "
            f"({int(adaptive.sum())})"
        ),
    )

    cbar = fig.colorbar(
        scatter,
        ax=ax,
    )

    cbar.set_label(
        "RAR Round"
    )

    ax.set_xlabel(
        "t"
    )

    ax.set_ylabel(
        "x"
    )

    ax.set_title(
        "NLSE Strong RAR Final PDE Points "
        "(10000 + 2000, Seed 99)"
    )

    ax.legend(
        frameon=True
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        ANALYSIS_DIR
        / (
            "nlse_rar_seed99"
            "_base10000_plus2000_points.jpg"
        ),
        dpi=IMAGE_DPI,
        format="jpg",
        bbox_inches="tight",
    )

    plt.show()

    plt.close(
        fig
    )


def create_sampling_cost_table():
    rows = []

    for seed in SEEDS:
        seed_dir = (
            METHOD_DIR
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
            "seed": int(
                seed
            ),
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

    frame = pd.DataFrame(
        rows
    )

    frame.to_csv(
        ANALYSIS_DIR
        / "rar_sampling_cost_individual.csv",
        index=False,
    )

    return frame


if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_plot_style()

    combined = (
        rebuild_individual_results()
    )

    create_statistics(
        combined
    )

    create_common_total_loss_curve()

    create_representative_fields()

    plot_selected_points()

    create_sampling_cost_table()

    print(
        f"\nNLSE sampling-only RAR analysis completed: "
        f"{ANALYSIS_DIR.resolve()}"
    )
