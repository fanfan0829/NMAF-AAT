from pathlib import Path

from nlse_seedwise_common import (
    PACKAGE_VERSION,
    REFERENCE_FILE,
    SEEDS,
    load_original_nmaf_seed_reference,
    load_training_points_np,
    locate_formal_rar_dir,
    find_formal_rar_points,
    rar_selection_overhead,
)


def main():
    print(f"Package version: {PACKAGE_VERSION}")
    print(f"Checker actually executed: {Path(__file__).resolve()}")
    print("=" * 110)

    hard_problems = []
    warnings = []

    if not REFERENCE_FILE.exists():
        hard_problems.append(f"Missing reference file: {REFERENCE_FILE}")
    else:
        print(f"FOUND NLS.mat: {REFERENCE_FILE}")

    for seed in SEEDS:
        try:
            base = load_training_points_np(seed)
            print(
                f"FOUND original fixed points seed {seed}: "
                f"Nres={len(base['x_res'])}, Nbc={len(base['t_bc'])}, Nic={len(base['x_ic'])}"
            )
        except Exception as exc:
            hard_problems.append(f"seed {seed} fixed points: {exc}")
            continue

        try:
            ref = load_original_nmaf_seed_reference(seed)
            print(
                f"FOUND original NMAF-AAT seed {seed}: target={ref['target_wall_time_seconds']:.6f}s | "
                f"loss={ref['nmaf_total_loss']:.6e} | L2={ref['nmaf_relative_l2_error']:.6e}"
            )
        except Exception as exc:
            hard_problems.append(f"seed {seed} NMAF-AAT reference: {exc}")


    try:
        rar_dir = locate_formal_rar_dir()
        print(f"FOUND formal RAR result directory: {rar_dir}")
        for seed in SEEDS:
            try:
                base = load_training_points_np(seed)
                xr, tr, source, mode = find_formal_rar_points(seed, base)
                overhead, overhead_source, known = rar_selection_overhead(seed)
                print(
                    f"RAR seed {seed}: points={len(xr)} | mode={mode} | source={source} | "
                    f"selection overhead={overhead:.6f}s | known={known}"
                )
                if not known:
                    warnings.append(
                        f"RAR seed {seed}: selection overhead unavailable; fixed-time RAR will be "
                        "optimistic Stage-B-only unless an override is supplied."
                    )
            except Exception as exc:
                warnings.append(f"RAR seed {seed} not ready: {exc}")
    except Exception as exc:
        warnings.append(f"Formal RAR directory not ready/unique: {exc}")

    print("=" * 110)
    if warnings:
        print("WARNINGS:")
        for item in warnings:
            print(" - " + item)

    if hard_problems:
        print("\nPRE-FLIGHT CHECK: NOT READY FOR CORE TANH/STAN/NMAF-REFERENCE EXPERIMENTS")
        for item in hard_problems:
            print(" - " + item)
        raise SystemExit(1)

    print("\nPRE-FLIGHT CHECK: CORE PASS")
    print("The NMAF reference and original fixed point files are ready.")
    if warnings:
        print("RAR may still require the formal selected-point file and/or selection-time metadata.")


if __name__ == "__main__":
    main()
