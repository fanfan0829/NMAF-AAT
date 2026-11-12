from seedwise_wave_common import SEEDS, PACKAGE_VERSION, export_nmaf_seedwise_targets

if __name__ == "__main__":
    print(f"Package version: {PACKAGE_VERSION}")
    info = export_nmaf_seedwise_targets()
    print("=" * 100)
    print("ORIGINAL NMAF-AAT SEED-WISE WALL-CLOCK TARGETS")
    print("IMPORTANT: there is NO averaged training target; each seed uses its own NMAF-AAT time.")
    for seed in SEEDS:
        ref = info["nmaf_reference_by_seed"][str(seed)]
        print(f"seed {seed:4d} | target={ref['target_wall_time_seconds']:.6f} s | "
              f"NMAF loss={ref['nmaf_total_loss']:.6e} | "
              f"NMAF L2={ref['nmaf_relative_l2_error']:.6e}")
    print("Saved target map to: nmaf_seedwise_targets.json")
    print("=" * 100)
