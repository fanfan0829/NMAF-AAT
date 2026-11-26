from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


COMMON_PATH = SCRIPT_DIR / "nc_common.py"

if not COMMON_PATH.exists():
    raise FileNotFoundError(
        "Cannot find nc_common.py in the same folder as this script."
    )

spec = importlib.util.spec_from_file_location(
    "nc_common_local_ra6_rar_continue_lbfgs",
    COMMON_PATH,
)
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)


TARGET_RA_EXPONENT = 6
METHOD = "rar"


SEED = 2026


EXTRA_LBFGS_STEPS = 10000


OVERWRITE = False


if int(F.RA_EXPONENT) != TARGET_RA_EXPONENT:
    raise RuntimeError(
        f"This script is for Ra=1e6, but local common file has "
        f"RA_EXPONENT={F.RA_EXPONENT}."
    )

if int(F.ADAM_STEPS) != 20000:
    raise RuntimeError(
        f"Expected ADAM_STEPS=20000, got {F.ADAM_STEPS}."
    )

if int(F.LBFGS_STEPS) != 10000:
    raise RuntimeError(
        f"Expected original LBFGS_STEPS=10000, got {F.LBFGS_STEPS}."
    )

if int(F.RAR_ROUNDS) != 10 or int(F.RAR_ADD_PER_ROUND) != 200:
    raise RuntimeError(
        "RAR protocol mismatch: expected 10 rounds x 200 added points."
    )

if int(F.RAR_FINAL_N) != 12000:
    raise RuntimeError(
        f"Expected final RAR interior count=12000, got {F.RAR_FINAL_N}."
    )

CURVE_EVERY_EXTRA_LBFGS = int(F.CURVE_EVERY_LBFGS)
PRINT_EVERY_EXTRA_LBFGS = int(F.PRINT_EVERY_LBFGS)


def seed_dir() -> Path:
    return F.fixed_out_dir(METHOD, SEED)


def original_final_checkpoint_path() -> Path:
    return (
        seed_dir()
        / f"Ra1e{F.RA_EXPONENT}_{METHOD}_seed{SEED}_final.pth"
    )


def original_history_path() -> Path:
    return seed_dir() / "training_history.csv"


def rar_selected_points_path() -> Path:
    return seed_dir() / f"rar_selected_points_seed{SEED}.npz"


def continuation_dir() -> Path:
    return seed_dir() / "continued_lbfgs_extra10000"


def continuation_checkpoint_path() -> Path:
    return (
        continuation_dir()
        / f"Ra1e{F.RA_EXPONENT}_{METHOD}_seed{SEED}"
          "_continued_lbfgs_extra10000_final.pth"
    )


def sync():
    F.sync()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_original_checkpoint():
    path = original_final_checkpoint_path()

    if not path.exists():
        raise FileNotFoundError(
            f"Original RAR final checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=F.DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unexpected checkpoint type: {type(checkpoint)}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Original checkpoint has no model_state_dict."
        )

    optimizer_keys = [
        k for k in checkpoint
        if "optimizer" in str(k).lower()
    ]

    return checkpoint, path, optimizer_keys


def load_rar_final_points(base):


    path = rar_selected_points_path()

    if not path.exists():
        raise FileNotFoundError(
            f"RAR selected-point file not found:\n{path}"
        )

    with np.load(path) as data:
        required = {
            "original_interior",
            "selected_interior",
            "interior",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"RAR point file is missing keys: {sorted(missing)}"
            )

        saved_original = np.asarray(
            data["original_interior"],
            dtype=np.float64,
        ).copy()

        selected = np.asarray(
            data["selected_interior"],
            dtype=np.float64,
        ).copy()

        current = np.asarray(
            data["interior"],
            dtype=np.float64,
        ).copy()

    base_original = np.asarray(
        base["interior"],
        dtype=np.float64,
    )

    if saved_original.shape != base_original.shape:
        raise RuntimeError(
            "Saved RAR original interior shape does not match fixed base points."
        )

    if not np.array_equal(
        saved_original,
        base_original,
    ):
        raise RuntimeError(
            "Saved RAR original 10000-point set does not exactly match "
            "the fixed seed-2026 base points."
        )

    if current.shape[0] != int(F.RAR_FINAL_N):
        raise RuntimeError(
            f"Expected {F.RAR_FINAL_N} final RAR points, "
            f"found {current.shape[0]}."
        )

    expected_selected = int(
        F.RAR_ROUNDS * F.RAR_ADD_PER_ROUND
    )
    if selected.shape[0] != expected_selected:
        raise RuntimeError(
            f"Expected {expected_selected} selected RAR points, "
            f"found {selected.shape[0]}."
        )

    if not np.array_equal(
        current[:F.N_INTERIOR],
        base_original,
    ):
        raise RuntimeError(
            "The first 10000 points of the final RAR set are not the "
            "original fixed interior points."
        )

    if not np.array_equal(
        current[F.N_INTERIOR:],
        selected,
    ):
        raise RuntimeError(
            "The appended RAR points do not match selected_interior."
        )

    return current, selected, path


def build_train_model_from_final_checkpoint():
    checkpoint, path, optimizer_keys = load_original_checkpoint()


    model = F.build_model(
        METHOD,
        F.DTYPE_TRAIN,
    )

    state = checkpoint["model_state_dict"]

    if hasattr(F, "cast_state_dict"):
        state = F.cast_state_dict(
            state,
            F.DTYPE_TRAIN,
        )
    else:
        state = {
            k: (
                v.to(dtype=F.DTYPE_TRAIN)
                if torch.is_tensor(v) and v.is_floating_point()
                else v
            )
            for k, v in state.items()
        }

    model.load_state_dict(
        state,
        strict=True,
    )
    model.train()

    return model, checkpoint, path, optimizer_keys


def continue_rar_seed2026():
    out = continuation_dir()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_out = continuation_checkpoint_path()

    if final_out.exists() and not OVERWRITE:
        print(
            "SKIP | seed=2026 | continuation already exists:\n"
            f"       {final_out}"
        )
        return None


    base = F.load_points_np(SEED)
    original_interior = np.asarray(
        base["interior"],
        dtype=np.float64,
    ).copy()

    (
        current_interior,
        selected_interior,
        selected_file,
    ) = load_rar_final_points(
        base
    )


    train_pts = F.make_tensors(
        base,
        current_interior,
        F.DTYPE_TRAIN,
    )


    monitor_pts = F.make_tensors(
        base,
        original_interior,
        F.DTYPE_TRAIN,
    )


    (
        model,
        source_checkpoint,
        source_path,
        source_optimizer_keys,
    ) = build_train_model_from_final_checkpoint()

    source_summary = source_checkpoint.get(
        "summary",
        {},
    )

    original_actual_lbfgs_evals = int(
        source_summary.get(
            "lbfgs_closure_evaluations",
            -1,
        )
    )


    if original_actual_lbfgs_evals != int(F.LBFGS_STEPS):
        raise RuntimeError(
            "RAR seed 2026 did not record the full original L-BFGS budget. "
            f"Recorded closure evaluations={original_actual_lbfgs_evals}, "
            f"expected={F.LBFGS_STEPS}. "
            "This script is intentionally restricted to a fully completed "
            "original seed."
        )

    original_combined_step = int(
        F.ADAM_STEPS
        + original_actual_lbfgs_evals
    )


    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=F.LBFGS_LR,
        max_iter=EXTRA_LBFGS_STEPS,
        max_eval=EXTRA_LBFGS_STEPS,
        history_size=F.LBFGS_HISTORY_SIZE,
        tolerance_grad=F.LBFGS_TOLERANCE_GRAD,
        tolerance_change=F.LBFGS_TOLERANCE_CHANGE,
        line_search_fn=F.LBFGS_LINE_SEARCH_FN,
    )

    closure_counter = 0
    diagnostic_seconds = 0.0
    extra_history = []

    print("=" * 116)
    print(
        f"Ra=1e{F.RA_EXPONENT} | RAR-PINN | seed={SEED} | "
        f"continue extra L-BFGS {EXTRA_LBFGS_STEPS}"
    )
    print(f"Source checkpoint : {source_path}")
    print(f"RAR point file    : {selected_file}")
    print(
        f"Original L-BFGS closure evals = {original_actual_lbfgs_evals} "
        f"(required full budget)"
    )
    print(
        f"Optimization points = {len(current_interior)} "
        f"(10000 original + {len(selected_interior)} RAR)"
    )
    print(
        f"Common monitor points = {len(original_interior)}"
    )
    print(
        "Original checkpoint optimizer state: "
        f"{source_optimizer_keys if source_optimizer_keys else 'NONE'}"
    )
    print(
        "Continuation mode: final MODEL parameters + fresh L-BFGS history"
    )
    print(
        f"L-BFGS: lr={F.LBFGS_LR}, "
        f"max_iter=max_eval={EXTRA_LBFGS_STEPS}, "
        f"history_size={F.LBFGS_HISTORY_SIZE}, "
        f"tol_grad={F.LBFGS_TOLERANCE_GRAD}, "
        f"tol_change={F.LBFGS_TOLERANCE_CHANGE}, "
        f"line_search={F.LBFGS_LINE_SEARCH_FN}"
    )
    print(
        f"Curve every {CURVE_EVERY_EXTRA_LBFGS} L-BFGS eval(s); "
        "RAR common-set monitor is outside the formal timer."
    )
    print(
        f"Device={F.DEVICE} | train dtype={F.DTYPE_TRAIN}"
    )
    print("=" * 116)


    with torch.enable_grad():
        d_opt0 = F.loss_components(
            model,
            train_pts,
        )
        optimization_start_loss = float(
            d_opt0["total_loss"].detach().cpu()
        )
        del d_opt0

        d_common0 = F.loss_components(
            model,
            monitor_pts,
        )
        common_start_loss = float(
            d_common0["total_loss"].detach().cpu()
        )
        del d_common0

    extra_history.append({
        "stage": "lbfgs_continuation",
        "stage_step": 0,
        "combined_step": int(
            original_combined_step
        ),
        "formal_elapsed_seconds_extra": 0.0,
        "total_loss": float(
            common_start_loss
        ),
        "optimization_total_loss": float(
            optimization_start_loss
        ),
        "training_n_interior": int(
            len(current_interior)
        ),
        "common_monitor_n_interior": int(
            len(original_interior)
        ),
    })

    print(
        f"Continuation start | combined step={original_combined_step} | "
        f"optimization loss={optimization_start_loss:.6e} | "
        f"common loss={common_start_loss:.6e}"
    )


    sync()
    lbfgs_wall_start = time.perf_counter()

    def closure():
        nonlocal closure_counter, diagnostic_seconds

        optimizer_lbfgs.zero_grad(
            set_to_none=True
        )


        d = F.loss_components(
            model,
            train_pts,
        )
        loss = d["total_loss"]
        loss.backward()

        closure_counter += 1

        need_curve = (
            closure_counter
            % CURVE_EVERY_EXTRA_LBFGS
            == 0
        )

        if need_curve:
            sync()

            formal_lbfgs_so_far = (
                time.perf_counter()
                - lbfgs_wall_start
                - diagnostic_seconds
            )

            optimization_value = float(
                loss.detach().cpu()
            )


            diag_start = time.perf_counter()

            with torch.enable_grad():
                d_common = F.loss_components(
                    model,
                    monitor_pts,
                )
                common_value = float(
                    d_common[
                        "total_loss"
                    ].detach().cpu()
                )
                del d_common

            extra_history.append({
                "stage": "lbfgs_continuation",
                "stage_step": int(
                    closure_counter
                ),
                "combined_step": int(
                    original_combined_step
                    + closure_counter
                ),
                "formal_elapsed_seconds_extra": float(
                    formal_lbfgs_so_far
                ),
                "total_loss": float(
                    common_value
                ),
                "optimization_total_loss": float(
                    optimization_value
                ),
                "training_n_interior": int(
                    len(current_interior)
                ),
                "common_monitor_n_interior": int(
                    len(original_interior)
                ),
            })

            if (
                closure_counter
                % PRINT_EVERY_EXTRA_LBFGS
                == 0
            ):
                print(
                    f"Extra L-BFGS eval "
                    f"{closure_counter:6d}/{EXTRA_LBFGS_STEPS} | "
                    f"combined step="
                    f"{original_combined_step + closure_counter:6d} | "
                    f"optimization loss={optimization_value:.6e} | "
                    f"common loss={common_value:.6e}"
                )

            sync()
            diagnostic_seconds += (
                time.perf_counter()
                - diag_start
            )

        return loss

    optimizer_lbfgs.step(
        closure
    )

    sync()

    lbfgs_wall_elapsed = (
        time.perf_counter()
        - lbfgs_wall_start
    )

    extra_formal_time = max(
        0.0,
        lbfgs_wall_elapsed
        - diagnostic_seconds,
    )

    if closure_counter <= 0:
        raise RuntimeError(
            "No additional L-BFGS closure evaluation was performed."
        )


    if (
        int(extra_history[-1]["stage_step"])
        != int(closure_counter)
    ):
        with torch.enable_grad():
            d_opt = F.loss_components(
                model,
                train_pts,
            )
            final_opt_curve_loss = float(
                d_opt["total_loss"].detach().cpu()
            )
            del d_opt

            d_common = F.loss_components(
                model,
                monitor_pts,
            )
            final_common_curve_loss = float(
                d_common["total_loss"].detach().cpu()
            )
            del d_common

        extra_history.append({
            "stage": "lbfgs_continuation",
            "stage_step": int(
                closure_counter
            ),
            "combined_step": int(
                original_combined_step
                + closure_counter
            ),
            "formal_elapsed_seconds_extra": float(
                extra_formal_time
            ),
            "total_loss": float(
                final_common_curve_loss
            ),
            "optimization_total_loss": float(
                final_opt_curve_loss
            ),
            "training_n_interior": int(
                len(current_interior)
            ),
            "common_monitor_n_interior": int(
                len(original_interior)
            ),
        })


    (
        eval_model,
        common,
        optimization,
        quantities,
    ) = F.final_metrics(
        METHOD,
        model,
        base,
        current_interior,
    )


    extra_history_df = pd.DataFrame(
        extra_history
    )

    extra_history_df.to_csv(
        out
        / "continued_lbfgs_extra10000_history.csv",
        index=False,
    )


    source_history_file = original_history_path()

    if source_history_file.exists():
        old_history = pd.read_csv(
            source_history_file
        )

        extension = extra_history_df[
            extra_history_df["stage_step"] > 0
        ].copy()

        old_formal_time = float(
            source_summary.get(
                "formal_training_time_seconds",
                0.0,
            )
        )

        extension[
            "formal_elapsed_seconds"
        ] = (
            extension[
                "formal_elapsed_seconds_extra"
            ]
            + old_formal_time
        )


        combined_extension = pd.DataFrame({
            "stage": extension["stage"],
            "stage_step": extension["stage_step"],
            "combined_step": extension["combined_step"],
            "formal_elapsed_seconds":
                extension["formal_elapsed_seconds"],


            "total_loss": extension["total_loss"],
            "training_n_interior":
                extension["training_n_interior"],
        })


        for col in old_history.columns:
            if col not in combined_extension.columns:
                combined_extension[col] = np.nan

        combined_extension = combined_extension[
            old_history.columns
        ]

        combined_history = pd.concat(
            [
                old_history,
                combined_extension,
            ],
            ignore_index=True,
        )

        combined_history.to_csv(
            out
            / "training_history_original_plus_extra10000.csv",
            index=False,
        )


    continuation_summary = {
        "protocol": (
            "RAR seed2026 warm-start from original completed final model; "
            "fresh L-BFGS optimizer because original optimizer state was not saved"
        ),
        "Ra_exponent": int(
            F.RA_EXPONENT
        ),
        "Ra": float(
            F.RA
        ),
        "Pr": float(
            F.PR
        ),
        "method": METHOD,
        "method_label": F.METHOD_LABELS[
            METHOD
        ],
        "seed": int(
            SEED
        ),

        "source_final_checkpoint": str(
            source_path
        ),
        "source_rar_selected_points_file": str(
            selected_file
        ),
        "source_checkpoint_optimizer_keys": (
            source_optimizer_keys
        ),
        "source_contains_lbfgs_optimizer_state": bool(
            source_optimizer_keys
        ),

        "original_adam_steps": int(
            F.ADAM_STEPS
        ),
        "original_lbfgs_configured_budget": int(
            F.LBFGS_STEPS
        ),
        "original_lbfgs_recorded_closure_evaluations": int(
            original_actual_lbfgs_evals
        ),
        "original_combined_step": int(
            original_combined_step
        ),

        "extra_lbfgs_configured_max_iter": int(
            EXTRA_LBFGS_STEPS
        ),
        "extra_lbfgs_configured_max_eval": int(
            EXTRA_LBFGS_STEPS
        ),
        "extra_lbfgs_actual_closure_evaluations": int(
            closure_counter
        ),
        "final_combined_step": int(
            original_combined_step
            + closure_counter
        ),

        "lbfgs_lr": float(
            F.LBFGS_LR
        ),
        "lbfgs_history_size": int(
            F.LBFGS_HISTORY_SIZE
        ),
        "lbfgs_tolerance_grad": float(
            F.LBFGS_TOLERANCE_GRAD
        ),
        "lbfgs_tolerance_change": float(
            F.LBFGS_TOLERANCE_CHANGE
        ),
        "lbfgs_line_search_fn": (
            F.LBFGS_LINE_SEARCH_FN
        ),

        "curve_every_lbfgs": int(
            CURVE_EVERY_EXTRA_LBFGS
        ),
        "training_dtype": str(
            F.DTYPE_TRAIN
        ),
        "evaluation_dtype": str(
            F.DTYPE_EVAL
        ),
        "device": str(
            F.DEVICE
        ),

        "common_monitor_n_interior": int(
            len(original_interior)
        ),
        "optimization_n_interior": int(
            len(current_interior)
        ),
        "rar_selected_n_interior": int(
            len(selected_interior)
        ),

        "extra_lbfgs_wall_seconds": float(
            lbfgs_wall_elapsed
        ),
        "extra_lbfgs_diagnostic_seconds_excluded": float(
            diagnostic_seconds
        ),
        "extra_lbfgs_formal_time_seconds": float(
            extra_formal_time
        ),

        "continuation_start_common_total_loss": float(
            common_start_loss
        ),
        "continuation_start_optimization_total_loss": float(
            optimization_start_loss
        ),

        "final_common_total_loss": float(
            common["total_loss"]
        ),
        "final_common_pde_loss": float(
            common["pde_loss"]
        ),
        "final_common_bc_loss": float(
            common["bc_loss"]
        ),
        "final_optimization_total_loss": float(
            optimization["total_loss"]
        ),
        "physical_quantities": quantities,

        "important_note": (
            "The original final checkpoint did not store L-BFGS optimizer "
            "state. Model parameters are continued exactly, but the new "
            "L-BFGS quasi-Newton history starts fresh."
        ),
    }

    save_json(
        out
        / "continued_lbfgs_extra10000_summary.json",
        continuation_summary,
    )


    torch.save(
        {
            "model_state_dict": F.cpu_state_dict(
                eval_model
            ),
            "train_model_state_dict": F.cpu_state_dict(
                model
            ),
            "optimizer_state_dict":
                optimizer_lbfgs.state_dict(),
            "summary":
                continuation_summary,
            "rar_current_interior":
                current_interior,
            "rar_selected_interior":
                selected_interior,
        },
        final_out,
    )

    print("=" * 116)
    print(
        f"RAR CONTINUATION DONE | seed={SEED}"
    )
    print(
        f"Extra actual closure evals = {closure_counter}"
    )
    print(
        f"Final combined step = "
        f"{original_combined_step + closure_counter}"
    )
    print(
        f"Final optimization loss (12000 pts) = "
        f"{optimization['total_loss']:.6e}"
    )
    print(
        f"Final common loss (10000 pts)       = "
        f"{common['total_loss']:.6e}"
    )
    print(
        f"Saved:\n{final_out}"
    )
    print("=" * 116)

    del (
        model,
        eval_model,
        train_pts,
        monitor_pts,
        source_checkpoint,
    )
    gc.collect()

    if F.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return continuation_summary


if __name__ == "__main__":
    print(
        "This script intentionally continues ONLY RAR seed 2026."
    )
    continue_rar_seed2026()
