from ac_beta_ablation_common import (
    AC_ROOT, REFERENCE_FILE, SEEDS, training_points_path,
    full_history_path, full_checkpoint_path, full_summary_path,
)

missing = []
print(f"Detected Allen-Cahn root: {AC_ROOT}")
print(f"Reference file: {REFERENCE_FILE}")
if not REFERENCE_FILE.exists():
    missing.append(str(REFERENCE_FILE))

for seed in SEEDS:
    for path in [
        training_points_path(seed),
        full_history_path(seed),
        full_checkpoint_path(seed, 20000),
        full_summary_path(seed),
    ]:
        ok = path.exists()
        print(f"{'OK' if ok else 'MISSING':7s} | {path}")
        if not ok:
            missing.append(str(path))

if missing:
    raise SystemExit("\nRequired formal files are missing. See the paths above.")
print("\nPreflight passed: formal NMAF-AAT results and fixed points are available.")
