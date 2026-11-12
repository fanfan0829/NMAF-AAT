from burgers_seedwise_common import SEEDS, export_nmaf_seedwise_targets

if __name__ == "__main__":
    info = export_nmaf_seedwise_targets()
    print("=" * 100)
    print("Burgers original NMAF-AAT seed-wise timing references")
    print("NO averaged target is used; NMAF-AAT is not retrained.")
    print("-" * 100)
    for seed in SEEDS:
        ref = info["nmaf_reference_by_seed"][str(seed)]
        print(
            f"seed {seed:4d} | target={ref['target_wall_time_seconds']:.6f} s | "
            f"loss={ref['nmaf_total_loss']:.6e} | "
            f"L2={ref['nmaf_relative_l2_error']:.6e} | "
            f"max|e|={ref['nmaf_maximum_absolute_error']:.6e}"
        )
    print("=" * 100)
