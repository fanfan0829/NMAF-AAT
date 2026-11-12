from seedwise_wave_common import (
    SEEDS, NEW_RESULT_ROOT, PACKAGE_VERSION, load_original_nmaf_seed_reference,
    training_points_path, method_label,
)
from pathlib import Path

if __name__ == "__main__":
    print(f"Package version: {PACKAGE_VERSION}")
    print("=" * 108)
    print("WAVE SEED-WISE NMAF-AAT MATCHED-TIME LAYOUT CHECK")
    print("=" * 108)
    ok = True
    print("\nOriginal NMAF-AAT reference + fixed points:")
    for seed in SEEDS:
        try:
            ref = load_original_nmaf_seed_reference(seed)
            print(f"  PASS seed {seed:4d} | NMAF target={ref['target_wall_time_seconds']:.6f} s | "
                  f"loss={ref['nmaf_total_loss']:.6e} | L2={ref['nmaf_relative_l2_error']:.6e}")
        except Exception as exc:
            ok = False
            print(f"  FAIL seed {seed:4d} | {exc}")
        p = training_points_path(seed)
        print(f"       points: {'PASS' if p.exists() else 'MISSING'} | {p}")
        ok = ok and p.exists()

    print("\nNew supplemental results (missing methods are allowed before analysis):")
    for method in ["tanh", "rar", "stan"]:
        present = []
        for seed in SEEDS:
            q = NEW_RESULT_ROOT / method / f"seed_{seed}" / "seedwise_matched_summary.json"
            if q.exists(): present.append(seed)
        missing = [s for s in SEEDS if s not in present]
        state = "READY" if not missing else "INCOMPLETE"
        print(f"  {state:10s} {method_label(method):20s} | present={present} | missing={missing}")

    print("\n" + ("SOURCE CHECK PASS" if ok else "SOURCE CHECK FAILED"))
    print("=" * 108)
