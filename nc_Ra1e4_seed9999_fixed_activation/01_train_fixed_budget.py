import argparse
from nc_fixed_activation_common import (
    METHODS, FIXED_ROOT, train_fixed_seed, write_fairness_audit,
    points_path, release_method_memory,
)


FIXED_ALL_METHODS = list(METHODS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()

    if a.all and a.method:
        p.error("Use either --all or --method, not both")


    if not a.all and a.method is None:
        a.all = True
        print("No --method/--all argument detected: defaulting to fixed-budget --all.")
        print("All activations, including tanh, use the same 7x64 MLP.")

    if not points_path(9999).exists():
        raise FileNotFoundError("Run 00_generate_fixed_points_seed9999.py first")

    write_fairness_audit(9999)
    release_method_memory("after fairness audit")

    methods = FIXED_ALL_METHODS if a.all else [a.method]

    if a.all:
        print("FIXED-BUDGET --all methods (7x64 MLP):")
        print("  " + ", ".join(methods))

    for m in methods:
        try:
            s = FIXED_ROOT / m / "seed_9999" / "training_summary.json"
            if s.exists() and not a.force:
                print("KEEP existing fixed-budget result:", s)
                continue
            train_fixed_seed(m, 9999)
        finally:
            release_method_memory(f"fixed-budget {m}")


if __name__ == "__main__":
    main()
