import argparse
import json
import subprocess
import sys
from pathlib import Path

from nc_fixed_activation_common import (
    METHODS, MATCHED_ROOT, formal_summary_path, matched_out_dir,
    train_matched_time_seed, write_fairness_audit, release_method_memory,
)

SEED = 9999


MATCHED_COMPETITORS = [m for m in METHODS if m != "nmaf"]


def verify_nmaf_target_exists():
    p = formal_summary_path("nmaf", SEED)
    if not p.exists():
        raise FileNotFoundError(
            f"Missing fixed-budget NMAF summary: {p}\n"
            "Run 01_train_fixed_budget.py first (or explicitly use --method nmaf)."
        )
    return p


def run_one_method(method: str, force: bool = False):


    verify_nmaf_target_exists()
    write_fairness_audit(SEED)
    release_method_memory("after fairness audit")

    if method == "nmaf":
        info = train_matched_time_seed("nmaf", SEED)
        print("NMAF matched-time reference reuses the fixed-budget checkpoint; no training performed.")
        return info

    summary_path = matched_out_dir(method, SEED) / "training_summary.json"
    if summary_path.exists() and not force:
        print("KEEP existing matched-time result:", summary_path)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    try:
        return train_matched_time_seed(method, SEED)
    finally:
        release_method_memory(f"matched-time {method}")


def launch_all_as_separate_processes(force: bool = False):

    target = verify_nmaf_target_exists()
    print("=" * 112)
    print("MATCHED-WALL-CLOCK PROCESS-ISOLATED LAUNCHER")
    print(f"NMAF target summary: {target}")
    print("NMAF is reused as the reference and is not retrained.")
    print("Each competitor below is launched in a fresh Python process:")
    print("  " + ", ".join(MATCHED_COMPETITORS))
    print("=" * 112)

    script = Path(__file__).resolve()
    completed = []
    for i, method in enumerate(MATCHED_COMPETITORS, start=1):
        cmd = [sys.executable, str(script), "--method", method]
        if force:
            cmd.append("--force")
        print("\n" + "#" * 112)
        print(f"PROCESS {i}/{len(MATCHED_COMPETITORS)} | {method}")
        print("Command:", " ".join(cmd))
        print("#" * 112)
        subprocess.run(cmd, check=True, cwd=str(script.parent))
        completed.append(method)

    manifest = {
        "seed": SEED,
        "protocol": "one activation method per fresh Python process",
        "nmaf_retrained": False,
        "nmaf_reference_summary": str(target),
        "completed_competitors": completed,
    }
    MATCHED_ROOT.mkdir(parents=True, exist_ok=True)
    (MATCHED_ROOT / "process_isolated_launcher_manifest_seed9999.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nAll matched-time competitor processes completed.")


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
        print("No --method/--all argument detected: defaulting to matched-time --all.")
        print("Each activation will run in a separate Python/CUDA child process.")

    if a.all:
        launch_all_as_separate_processes(force=a.force)
        return

    run_one_method(a.method, force=a.force)


if __name__ == "__main__":
    main()
