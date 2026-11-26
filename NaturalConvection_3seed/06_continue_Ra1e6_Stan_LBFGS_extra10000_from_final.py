from __future__ import annotations

import gc
import importlib.util
import json
import os
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
        f"Cannot find local nc_common.py:\n{COMMON_PATH}\n"
        "Put this script in the same Ra1e6 folder as nc_common.py."
    )

spec = importlib.util.spec_from_file_location(
    "nc_common_local_ra6_continue_lbfgs",
    COMMON_PATH,
)
F = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = F
spec.loader.exec_module(F)


TARGET_RA_EXPONENT = 6
METHOD = "stan"

if int(F.RA_EXPONENT) != TARGET_RA_EXPONENT:
    raise RuntimeError(
        f"This continuation script is for Ra=1e6, but local nc_common.py "
        f"has RA_EXPONENT={F.RA_EXPONENT}."
    )

if int(F.ADAM_STEPS) != 20000:
    raise RuntimeError(
        f"Expected original ADAM_STEPS=20000, got {F.ADAM_STEPS}."
    )

if int(F.LBFGS_STEPS) != 10000:
    raise RuntimeError(
        f"Expected original LBFGS_STEPS=10000, got {F.LBFGS_STEPS}."
    )


EXTRA_LBFGS_STEPS = 10000


SEEDS_TO_CONTINUE = list(F.SEEDS)


OVERWRITE = False


ORIGINAL_NOMINAL_TOTAL_STEP = int(F.ADAM_STEPS + F.LBFGS_STEPS)


CURVE_EVERY_EXTRA_LBFGS = int(F.CURVE_EVERY_LBFGS)
PRINT_EVERY_EXTRA_LBFGS = int(F.PRINT_EVERY_LBFGS)


def original_final_checkpoint_path(seed: int) -> Path:
    return (
        F.fixed_out_dir(METHOD, seed)
        / f"Ra1e{F.RA_EXPONENT}_{METHOD}_seed{int(seed)}_final.pth"
    )


def original_history_path(seed: int) -> Path:
    return F.fixed_out_dir(METHOD, seed) / "training_history.csv"


def continuation_dir(seed: int) -> Path:
    return (
        F.fixed_out_dir(METHOD, seed)
        / "continued_lbfgs_extra10000"
    )


def continuation_checkpoint_path(seed: int) -> Path:
    return (
        continuation_dir(seed)
        / f"Ra1e{F.RA_EXPONENT}_{METHOD}_seed{int(seed)}"
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


def load_original_checkpoint(seed: int):
    path = original_final_checkpoint_path(seed)

    if not path.exists():
        raise FileNotFoundError(
            f"Original final checkpoint not found:\n{path}"
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
            f"{path} has no model_state_dict."
        )


    optimizer_keys = [
        k for k in checkpoint.keys()
        if "optimizer" in str(k).lower()
    ]

    return checkpoint, path, optimizer_keys


def build_train_model_from_final_checkpoint(seed: int):
    checkpoint, path, optimizer_keys = load_original_checkpoint(seed)


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


def continue_one_seed(seed: int):
    seed = int(seed)

    out = continuation_dir(seed)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_out = continuation_checkpoint_path(seed)

    if final_out.exists() and not OVERWRITE:
        print(
            f"SKIP | seed={seed} | continuation already exists:\n"
            f"       {final_out}"
        )
        return None


    base = F.load_points_np(seed)
    original_interior = base["interior"].copy()

    train_pts = F.make_tensors(
        base,
        original_interior,
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
    ) = build_train_model_from_final_checkpoint(seed)

    source_summary = (
        source_checkpoint.get("summary", {})
        if isinstance(source_checkpoint, dict)
        else {}
    )

    original_actual_lbfgs_evals = int(
        source_summary.get(
            "lbfgs_closure_evaluations",
            F.LBFGS_STEPS,
        )
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

    print("=" * 112)
    print(
        f"Ra=1e{F.RA_EXPONENT} | {F.METHOD_LABELS[METHOD]} | seed={seed}"
    )
    print(f"Source final checkpoint: {source_path}")
    print(
        f"Source checkpoint optimizer keys: "
        f"{source_optimizer_keys if source_optimizer_keys else 'NONE'}"
    )
    print(
        "Continuation mode: final model parameters + NEW L-BFGS optimizer "
        "(original quasi-Newton memory unavailable)"
    )
    print(
        f"Original nominal total step = {ORIGINAL_NOMINAL_TOTAL_STEP} "
        f"(Adam {F.ADAM_STEPS} + L-BFGS {F.LBFGS_STEPS})"
    )
    print(
        f"Original recorded L-BFGS closure evals = "
        f"{original_actual_lbfgs_evals}"
    )
    print(
        f"Additional configured L-BFGS budget: "
        f"max_iter=max_eval={EXTRA_LBFGS_STEPS}"
    )
    print(
        f"L-BFGS settings: lr={F.LBFGS_LR}, "
        f"history_size={F.LBFGS_HISTORY_SIZE}, "
        f"tol_grad={F.LBFGS_TOLERANCE_GRAD}, "
        f"tol_change={F.LBFGS_TOLERANCE_CHANGE}, "
        f"line_search={F.LBFGS_LINE_SEARCH_FN}"
    )
    print(
        f"Device={F.DEVICE} | train dtype={F.DTYPE_TRAIN} | "
        f"N_interior={len(original_interior)}"
    )
    print("=" * 112)


    with torch.enable_grad():
        initial_d = F.loss_components(
            model,
            train_pts,
        )
        initial_loss = float(
            initial_d["total_loss"].detach().cpu()
        )
    del initial_d
    gc.collect()

    extra_history.append({
        "stage": "lbfgs_continuation",
        "stage_step": 0,
        "combined_step": ORIGINAL_NOMINAL_TOTAL_STEP,
        "formal_elapsed_seconds_extra": 0.0,
        "total_loss": initial_loss,
        "training_n_interior": int(len(original_interior)),
    })

    print(
        f"Continuation start | combined step={ORIGINAL_NOMINAL_TOTAL_STEP} | "
        f"loss={initial_loss:.6e}"
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
            closure_counter % CURVE_EVERY_EXTRA_LBFGS == 0
        )

        if need_curve:
            sync()

            formal_lbfgs_so_far = (
                time.perf_counter()
                - lbfgs_wall_start
                - diagnostic_seconds
            )

            diag_start = time.perf_counter()


            total_value = float(
                loss.detach().cpu()
            )

            extra_history.append({
                "stage": "lbfgs_continuation",
                "stage_step": int(closure_counter),
                "combined_step": int(
                    ORIGINAL_NOMINAL_TOTAL_STEP
                    + closure_counter
                ),
                "formal_elapsed_seconds_extra": float(
                    formal_lbfgs_so_far
                ),
                "total_loss": total_value,
                "training_n_interior": int(
                    len(original_interior)
                ),
            })

            if (
                closure_counter % PRINT_EVERY_EXTRA_LBFGS == 0
            ):
                print(
                    f"Extra L-BFGS eval "
                    f"{closure_counter:6d}/{EXTRA_LBFGS_STEPS} | "
                    f"combined step="
                    f"{ORIGINAL_NOMINAL_TOTAL_STEP + closure_counter:6d} | "
                    f"loss={total_value:.6e}"
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
            d_final = F.loss_components(
                model,
                monitor_pts,
            )
            final_curve_loss = float(
                d_final["total_loss"].detach().cpu()
            )
        del d_final

        extra_history.append({
            "stage": "lbfgs_continuation",
            "stage_step": int(closure_counter),
            "combined_step": int(
                ORIGINAL_NOMINAL_TOTAL_STEP
                + closure_counter
            ),
            "formal_elapsed_seconds_extra": float(
                extra_formal_time
            ),
            "total_loss": final_curve_loss,
            "training_n_interior": int(
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
        original_interior,
    )


    extra_history_df = pd.DataFrame(
        extra_history
    )

    extra_history_df.to_csv(
        out / "continued_lbfgs_extra10000_history.csv",
        index=False,
    )


    source_history_file = original_history_path(
        seed
    )

    if source_history_file.exists():
        old_history = pd.read_csv(
            source_history_file
        )

        extension_for_combined = (
            extra_history_df[
                extra_history_df["stage_step"] > 0
            ].copy()
        )

        extension_for_combined = (
            extension_for_combined.rename(
                columns={
                    "formal_elapsed_seconds_extra":
                    "formal_elapsed_seconds",
                }
            )
        )

        old_formal_time = float(
            source_summary.get(
                "formal_training_time_seconds",
                0.0,
            )
        )

        extension_for_combined[
            "formal_elapsed_seconds"
        ] = (
            extension_for_combined[
                "formal_elapsed_seconds"
            ]
            + old_formal_time
        )


        for col in old_history.columns:
            if col not in extension_for_combined.columns:
                extension_for_combined[col] = np.nan

        for col in extension_for_combined.columns:
            if col not in old_history.columns:
                old_history[col] = np.nan

        extension_for_combined = extension_for_combined[
            old_history.columns
        ]

        combined_history = pd.concat(
            [
                old_history,
                extension_for_combined,
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
            "warm-start from original final model parameters; "
            "fresh L-BFGS optimizer because original optimizer state "
            "was not saved"
        ),
        "Ra_exponent": int(F.RA_EXPONENT),
        "Ra": float(F.RA),
        "Pr": float(F.PR),
        "method": METHOD,
        "method_label": F.METHOD_LABELS[METHOD],
        "seed": int(seed),

        "source_final_checkpoint": str(
            source_path
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
        "original_nominal_total_step": int(
            ORIGINAL_NOMINAL_TOTAL_STEP
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
        "final_nominal_combined_step": int(
            ORIGINAL_NOMINAL_TOTAL_STEP
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

        "training_dtype": str(
            F.DTYPE_TRAIN
        ),
        "evaluation_dtype": str(
            F.DTYPE_EVAL
        ),
        "device": str(
            F.DEVICE
        ),
        "training_n_interior": int(
            len(original_interior)
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

        "continuation_start_total_loss": float(
            initial_loss
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
            "The original final checkpoint did not store the L-BFGS optimizer "
            "state. Therefore the model weights are continued exactly, but "
            "the L-BFGS quasi-Newton history is restarted."
        ),
    }

    save_json(
        out / "continued_lbfgs_extra10000_summary.json",
        continuation_summary,
    )


    continued_checkpoint = {
        "model_state_dict": F.cpu_state_dict(
            eval_model
        ),
        "train_model_state_dict": F.cpu_state_dict(
            model
        ),
        "optimizer_state_dict": (
            optimizer_lbfgs.state_dict()
        ),
        "summary": continuation_summary,
    }

    torch.save(
        continued_checkpoint,
        final_out,
    )

    print(
        f"CONTINUATION DONE | seed={seed} | "
        f"extra actual closure evals={closure_counter} | "
        f"nominal combined step="
        f"{ORIGINAL_NOMINAL_TOTAL_STEP + closure_counter} | "
        f"final common loss={common['total_loss']:.6e}"
    )

    print(
        f"Saved continuation checkpoint:\n{final_out}"
    )

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


def main():
    print("=" * 112)
    print(
        "Ra=1e6 Stan-PINN: continue final model with "
        "an additional 10000 L-BFGS budget"
    )
    print("=" * 112)
    print(
        "IMPORTANT: original final checkpoint has no saved L-BFGS optimizer "
        "state; this is a parameter warm-start with a fresh LBFGS history."
    )
    print(
        f"Seeds: {SEEDS_TO_CONTINUE}"
    )
    print(
        f"Device inherited from nc_common.py: {F.DEVICE}"
    )
    print("=" * 112)

    summaries = []

    for seed in SEEDS_TO_CONTINUE:
        try:
            summary = continue_one_seed(
                seed
            )
            if summary is not None:
                summaries.append(
                    summary
                )
        except Exception as exc:
            print(
                f"FAILED | seed={seed} | "
                f"{type(exc).__name__}: {exc}"
            )
            raise

    if summaries:
        summary_table = pd.DataFrame([
            {
                "seed": s["seed"],
                "source_contains_lbfgs_optimizer_state":
                    s["source_contains_lbfgs_optimizer_state"],
                "original_nominal_total_step":
                    s["original_nominal_total_step"],
                "extra_lbfgs_actual_closure_evaluations":
                    s["extra_lbfgs_actual_closure_evaluations"],
                "final_nominal_combined_step":
                    s["final_nominal_combined_step"],
                "continuation_start_total_loss":
                    s["continuation_start_total_loss"],
                "final_common_total_loss":
                    s["final_common_total_loss"],
                "extra_lbfgs_formal_time_seconds":
                    s["extra_lbfgs_formal_time_seconds"],
            }
            for s in summaries
        ])

        summary_table.to_csv(
            F.RESULT_ROOT
            / METHOD
            / "continued_lbfgs_extra10000_all_seeds_summary.csv",
            index=False,
        )

        print("\nContinuation summary:")
        print(
            summary_table.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
