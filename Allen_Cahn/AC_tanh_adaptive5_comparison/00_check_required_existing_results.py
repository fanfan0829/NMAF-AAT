from ac_tanh_adaptive5_common import (
    REFERENCE_FILE, SEEDS, training_points_path, formal_standard_final_checkpoint,
    formal_standard_history, HIDDEN_LAYERS, HIDDEN_NEURONS, N_RES, N_BC, N_IC,
)

if __name__ == "__main__":
    problems = []
    print(f"AC.mat: {REFERENCE_FILE}")
    if not REFERENCE_FILE.exists(): problems.append(str(REFERENCE_FILE))
    for seed in SEEDS:
        p = training_points_path(seed)
        print(f"fixed points seed {seed}: {p} | exists={p.exists()}")
        if not p.exists(): problems.append(str(p))
        try:
            ck = formal_standard_final_checkpoint(seed)
            print(f"Standard checkpoint seed {seed}: {ck}")
        except Exception as e:
            problems.append(f"Standard checkpoint seed {seed}: {e}")
        try:
            h = formal_standard_history(seed)
            print(f"Standard history seed {seed}: {h}")
        except Exception as e:
            problems.append(f"Standard history seed {seed}: {e}")
    print(f"Architecture: actual {HIDDEN_LAYERS+1} hidden activation sites x {HIDDEN_NEURONS}")
    print(f"Points: PDE={N_RES}, BC={N_BC}, IC={N_IC}")
    if problems:
        print("\nMISSING/INVALID:")
        for x in problems: print(" -", x)
        raise SystemExit(1)
    print("\nPASS: required formal Standard results, AC.mat, and fixed points are available.")
