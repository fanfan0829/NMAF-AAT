from ac_adaptive5_common import export_nmaf_seedwise_targets
if __name__ == "__main__":
    info = export_nmaf_seedwise_targets()
    print("Same-seed NMAF-AAT wall-clock targets:")
    for seed, value in info["target_wall_time_seconds_by_seed"].items():
        print(f"  seed {seed}: {value:.6f} s")
