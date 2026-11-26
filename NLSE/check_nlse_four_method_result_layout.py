from pathlib import Path

SEEDS = [99, 1234, 2026]
CHECKPOINTS = [3000, 6000, 10000]
ROOT = Path("nlse_multiseed_results")


def main():
    problems = []
    if not Path("NLS.mat").exists():
        problems.append("Missing NLS.mat in current working directory.")
    if not ROOT.exists():
        problems.append(f"Missing result root: {ROOT}")
    else:
        for method in ["tanh", "nmaf_aat", "stan"]:
            for seed in SEEDS:
                sd = ROOT / method / f"seed_{seed}"
                if not sd.exists():
                    problems.append(f"Missing directory: {sd}")
                    continue
                if not (sd / "lbfgs_history.csv").exists():
                    problems.append(f"Missing history: {sd / 'lbfgs_history.csv'}")
                for ev in CHECKPOINTS:
                    ck = sd / f"nlse_{method}_seed{seed}_eval{ev:05d}.pth"
                    if not ck.exists():
                        problems.append(f"Missing checkpoint: {ck}")

        rar_candidates = []
        for p in ROOT.iterdir():
            if p.is_dir() and all(
                (p / f"seed_{s}" / "fixed_monitor_history.csv").exists()
                for s in SEEDS
            ):
                rar_candidates.append(p)
        if len(rar_candidates) == 0:
            problems.append(
                "No RAR result directory containing fixed_monitor_history.csv "
                "for all three seeds was found."
            )
        elif len(rar_candidates) > 1:
            problems.append(
                "Multiple RAR result directories found; set RAR_METHOD_DIR_OVERRIDE "
                "in the unified analysis script: " + ", ".join(str(p) for p in rar_candidates)
            )
        else:
            print(f"RAR directory detected: {rar_candidates[0]}")

    if problems:
        print("\nPRE-FLIGHT CHECK: NOT READY")
        for item in problems:
            print(" - " + item)
        raise SystemExit(1)

    print("\nPRE-FLIGHT CHECK: PASS")
    print("The NLSE four-method result layout is ready for unified analysis.")


if __name__ == "__main__":
    main()
