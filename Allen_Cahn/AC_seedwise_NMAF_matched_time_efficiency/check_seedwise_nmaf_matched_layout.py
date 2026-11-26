from pathlib import Path
import torch

import ac_seedwise_common as ac


def linear_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith("input_layer.") or k.startswith("hidden_layers.") or k.startswith("output_layer.")
    }


def main():
    print(f"Package version: {ac.PACKAGE_VERSION}")
    print(f"Checker actually executed: {Path(__file__).resolve()}")
    problems = []

    if not ac.REFERENCE_FILE.exists():
        problems.append(f"Missing AC.mat: {ac.REFERENCE_FILE}")
    if not ac.OLD_RESULT_ROOT.exists():
        problems.append(f"Missing original result root: {ac.OLD_RESULT_ROOT}")

    print("\nOriginal NMAF-AAT seed-wise reference")
    print("-" * 100)
    for seed in ac.SEEDS:
        p = ac.training_points_path(seed)
        if not p.exists():
            problems.append(f"Missing fixed points for seed {seed}: {p}")
        try:
            target, src = ac.read_original_nmaf_time(seed)
            print(f"seed {seed:4d} | target={target:.6f}s | source={src}")
        except Exception as exc:
            problems.append(f"Cannot read NMAF target for seed {seed}: {exc}")

    print("\nSame-seed Linear-initialization check")
    print("-" * 100)
    for seed in ac.SEEDS:
        ac.set_seed(seed)
        tanh = ac.TanhPINN().double().cpu()
        ac.set_seed(seed)
        nmaf = ac.NMAFAATPINN().double().cpu()
        ac.set_seed(seed)
        stan = ac.StanPINN().double().cpu()
        a, b, c = linear_state(tanh), linear_state(nmaf), linear_state(stan)
        ok = a.keys() == b.keys() == c.keys() and all(
            torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in a
        )
        print(f"seed {seed:4d} | Linear init identical: {ok}")
        if not ok:
            problems.append(f"Linear initialization mismatch for seed {seed}")

    print("\nFormal RAR selected-point availability")
    print("-" * 100)
    try:
        rar_dir = ac.locate_formal_rar_dir()
        print(f"Detected old RAR directory: {rar_dir}")
        for seed in ac.SEEDS:
            try:
                base = ac.load_training_points_np(seed)
                x, t, src, mode = ac.find_formal_rar_points(seed, base)
                overhead, over_src, known = ac.rar_selection_overhead(seed)
                print(
                    f"seed {seed:4d} | points={len(x)} ({mode}) | {src} | "
                    f"selection overhead={overhead:.6f}s | known={known} | {over_src}"
                )
            except Exception as exc:
                print(f"seed {seed:4d} | RAR NOT READY: {exc}")
        print(
            "RAR not-ready messages do NOT prevent tanh/Stan runs. They only need to be resolved before running 03_train_rar..."
        )
    except Exception as exc:
        print(f"RAR directory not ready: {exc}")
        print("This does not prevent tanh/Stan seed-wise efficiency runs.")

    if problems:
        print("\nPRE-FLIGHT CHECK: REQUIRED INPUTS NOT READY")
        for item in problems:
            print(" - " + item)
        raise SystemExit(1)

    print("\nPRE-FLIGHT CHECK: PASS for required AC/NMAF/fixed-point inputs")
    print("No averaged wall-clock target will be used.")


if __name__ == "__main__":
    main()
