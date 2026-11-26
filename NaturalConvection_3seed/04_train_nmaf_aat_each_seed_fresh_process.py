from __future__ import annotations

import subprocess
import sys
from pathlib import Path

METHOD = "nmaf_aat"
SEEDS = [9999, 1234, 2026]

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "_run_one_fixed_seed.py"

RUNNER_CODE = r'''import sys
from nc_common import train_fixed_seed, formal_summary_path

method = sys.argv[1]
seed = int(sys.argv[2])
summary = formal_summary_path(method, seed)
if summary.exists():
    print(f"SKIP | {method} | seed={seed} | existing summary: {summary}")
else:
    train_fixed_seed(method, seed)
'''

RUNNER.write_text(RUNNER_CODE, encoding="utf-8")

if __name__ == "__main__":
    for seed in SEEDS:
        print("=" * 112)
        print(f"Launching independent process: {METHOD} seed={seed}")
        print("=" * 112)
        subprocess.run(
            [sys.executable, str(RUNNER), METHOD, str(seed)],
            cwd=str(SCRIPT_DIR),
            check=True,
        )
