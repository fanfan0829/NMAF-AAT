from ac_adaptive5_common import (
    REFERENCE_FILE, SEEDS, training_points_path, fixed_history_path,
    formal_final_checkpoint, locate_standard_matched_summary,
)

print("Allen-Cahn adaptive-5 comparison prerequisite check")
print("=" * 88)
print(f"AC.mat: {REFERENCE_FILE} | exists={REFERENCE_FILE.exists()}")

ok = REFERENCE_FILE.exists()
for seed in SEEDS:
    p = training_points_path(seed)
    print(f"fixed points seed {seed}: {p} | exists={p.exists()}")
    ok &= p.exists()

for method in ("tanh", "nmaf_aat"):
    for seed in SEEDS:
        h = fixed_history_path(method, seed)
        try:
            c = formal_final_checkpoint(method, seed)
            ctext = str(c)
            c_ok = True
        except Exception as exc:
            ctext = f"ERROR: {exc}"
            c_ok = False
        print(f"{method:9s} seed {seed}: history={h.exists()} | checkpoint={ctext}")
        ok &= h.exists() and c_ok

print("\nPrevious Standard-PINN matched-time results:")
for seed in SEEDS:
    try:
        p = locate_standard_matched_summary(seed)
        print(f"  seed {seed}: FOUND {p}")
    except Exception as exc:
        print(f"  seed {seed}: MISSING ({exc})")

print("=" * 88)
if not ok:
    raise SystemExit("Required formal fixed-iteration inputs are incomplete.")
print("Core fixed-iteration prerequisites are complete.")
