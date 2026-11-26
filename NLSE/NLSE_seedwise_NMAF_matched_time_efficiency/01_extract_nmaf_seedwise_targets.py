from pathlib import Path
from nlse_seedwise_common import PACKAGE_VERSION, SEEDS, export_nmaf_seedwise_targets

if __name__ == "__main__":
    print(f"Package version: {PACKAGE_VERSION}")
    print(f"Script actually executed: {Path(__file__).resolve()}")
    info = export_nmaf_seedwise_targets()
    print("\nNLSE NMAF-AAT seed-wise target times")
    print("=" * 88)
    for seed in SEEDS:
        ref = info["nmaf_reference_by_seed"][str(seed)]
        print(
            f"seed {seed:4d} | target = {ref['target_wall_time_seconds']:.6f} s | "
            f"loss = {ref['nmaf_total_loss']:.6e} | L2 = {ref['nmaf_relative_l2_error']:.6e}"
        )
    print("=" * 88)
    print("IMPORTANT: there is NO averaged training target in this protocol.")
