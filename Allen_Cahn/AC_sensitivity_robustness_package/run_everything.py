from __future__ import annotations

import argparse

from ac_sensitivity_common import (
    TRAIN_EXPERIMENTS,
    SEEDS,
    METHODS,
    DEVICE,
    baseline_result_root,
    reference_file,
    verify_required_baseline_files,
    write_experiment_plan,
)


def print_plan():
    print("=" * 100)
    print("Allen-Cahn reviewer-response experiment plan")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print(f"Baseline root: {baseline_result_root()}")
    try:
        print(f"Reference file: {reference_file()}")
    except FileNotFoundError as exc:
        print(f"Reference file: NOT FOUND ({exc})")

    print("\nExisting clean baseline (NOT retrained):")
    print("  7 hidden layers x 128 neurons, N_RES=3000, N_BC=200, N_IC=500, noise=0%")

    print("\nNew configurations:")
    for spec in TRAIN_EXPERIMENTS:
        print(
            f"  {spec.name:12s} | study={spec.study:5s} | level={spec.level:g} | "
            f"depth={spec.depth} | width={spec.width} | "
            f"data={spec.data_factor:g}x | noise={spec.noise_pct:g}%"
        )

    total = len(TRAIN_EXPERIMENTS) * len(SEEDS) * len(METHODS)
    print(f"\nNew training runs: {len(TRAIN_EXPERIMENTS)} configs x "
          f"{len(SEEDS)} seeds x {len(METHODS)} methods = {total}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    verify_required_baseline_files(require_reference=False)
    write_experiment_plan()
    print_plan()

    if args.check_only:
        return

    from train_all_experiments import main as train_main
    from postprocess_results import main as postprocess_main

    train_main()
    postprocess_main()


if __name__ == "__main__":
    main()
