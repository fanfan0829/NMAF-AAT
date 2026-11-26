import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
import nc_fixed_activation_common as F

SEED = 9999
CHECKPOINT_NAME = "Ra1e4_tanh_seed9999_final.pth"


def validate_source(src):
    src = Path(src)
    for name in (CHECKPOINT_NAME, "training_summary.json"):
        if not (src / name).is_file():
            raise FileNotFoundError(src / name)
    summary = json.loads((src / "training_summary.json").read_text(encoding="utf-8"))
    expected = {
        "Ra_exponent": 4, "Ra": 1e4, "Pr": F.PR,
        "method": "tanh", "seed": SEED,
        "adam_steps": F.ADAM_STEPS,
        "lbfgs_configured_max_iter": F.LBFGS_MAX_ITER,
        "lbfgs_configured_max_eval": F.LBFGS_MAX_EVAL,
        "lbfgs_history_size": F.LBFGS_HISTORY_SIZE,
        "common_monitor_n_interior": F.N_INTERIOR,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Source summary {key} does not match the required protocol: expected {value}.")
    checkpoint = torch.load(src / CHECKPOINT_NAME, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Source checkpoint must contain model_state_dict.")
    state = checkpoint["model_state_dict"]
    torch.manual_seed(SEED)
    model = F.MLPPINN(kind="tanh", dtype=F.DTYPE_TRAIN)
    if summary.get("initial_linear_state_sha256") != F.linear_state_hash(model):
        raise ValueError("Source initialization does not match the required seed9999 MLP.")
    expected_state = model.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("Source model is not the required seven-hidden-layer, width64 MLP.")
    for key, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected_state[key].shape or not torch.isfinite(value).all():
            raise ValueError("Source weights do not match the required 7x64 MLP.")
    converted = F.cast_state_dict(state, F.DTYPE_TRAIN)
    if not all(torch.isfinite(value).all() for value in converted.values()):
        raise ValueError("Source weights are not finite in the required training precision.")
    model.load_state_dict(converted, strict=True)
    embedded = checkpoint.get("summary")
    if not isinstance(embedded, dict):
        raise ValueError("Source checkpoint must include its experiment summary.")
    for key, value in expected.items():
        if embedded.get(key) != value:
            raise ValueError("Checkpoint and JSON summary do not describe the same protocol.")
    if embedded.get("initial_linear_state_sha256") != summary.get("initial_linear_state_sha256"):
        raise ValueError("Checkpoint and JSON summary have different initialization records.")
    summary.update(F.architecture_metadata())
    summary["imported_model_sha256"] = hashlib.sha256((src / CHECKPOINT_NAME).read_bytes()).hexdigest()
    checkpoint = dict(checkpoint)
    checkpoint["summary"] = dict(summary)
    return checkpoint, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True, help="Compatible 7x64 MLP tanh result directory")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    src = Path(args.source_dir).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(src)
    checkpoint, summary = validate_source(src)
    dst = F.fixed_out_dir("tanh", SEED)
    if src == dst.resolve():
        raise ValueError("Source and destination directories must differ.")
    outputs = [dst / CHECKPOINT_NAME, dst / "training_summary.json", dst / "training_history.csv"]
    if not args.force and any(path.exists() for path in outputs):
        print("KEEP existing target files; nothing imported. Use --force to replace:", dst)
        return
    history = src / "training_history.csv"
    if not history.is_file() and (dst / history.name).exists():
        raise ValueError("Source history is missing but target history exists; refusing to mix result files.")
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst / CHECKPOINT_NAME)
    F.save_json(dst / "training_summary.json", summary)
    if history.is_file():
        shutil.copy2(history, dst / history.name)
    else:
        print("OPTIONAL training_history.csv not found.")
    print("Verified 7x64 MLP tanh result imported:", dst)
    print("No retraining was performed.")


if __name__ == "__main__":
    main()
