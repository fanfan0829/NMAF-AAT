from ac_tanh_adaptive5_common import export_tanh_aat_seedwise_targets
if __name__ == "__main__":
    info = export_tanh_aat_seedwise_targets()
    print(info["target_wall_time_seconds_by_seed"])
