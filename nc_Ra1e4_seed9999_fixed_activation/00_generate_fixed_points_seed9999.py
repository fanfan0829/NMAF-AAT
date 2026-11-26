from nc_fixed_activation_common import generate_and_save_all_points, write_fairness_audit

if __name__ == "__main__":
    generate_and_save_all_points(overwrite=False)
    audit = write_fairness_audit(seed=9999)
    print("Fairness audit passed:", audit["all_initial_linear_hashes_identical"])
