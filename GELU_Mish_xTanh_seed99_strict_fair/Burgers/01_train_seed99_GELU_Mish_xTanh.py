import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from burgers_fixed_activation_seed99_common import *

METHODS = ["gelu", "mish", "xtanh"]
SEED = 99


def _ensure_seed99_points():
    path = training_points_path(SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        status = "reused existing formal seed-99 training points"
    else:
        data = generate_points_for_seed(SEED)
        np.savez_compressed(path, **data)
        status = "generated seed-99 training points using the exact formal generator"
    return path, status


def _linear_sha256(model):
    h = hashlib.sha256()
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            count += 1
            for key, tensor in (("weight", module.weight), ("bias", module.bias)):
                h.update(f"{name}.{key}".encode("utf-8"))
                h.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return h.hexdigest(), count


def fairness_preflight():
    point_path, point_status = _ensure_seed99_points()
    point_hash = hashlib.sha256(point_path.read_bytes()).hexdigest()

    set_seed(SEED)
    reference = TanhPINN().double().to(DEVICE)
    ref_hash, ref_linear_layers = _linear_sha256(reference)
    ref_params = sum(p.numel() for p in reference.parameters())

    methods = {}
    for method in METHODS:
        set_seed(SEED)
        model = build_model(method)
        linear_hash, linear_layers = _linear_sha256(model)
        methods[method] = {
            "linear_sha256": linear_hash,
            "linear_layers": linear_layers,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "same_linear_initialization_as_formal_tanh": linear_hash == ref_hash,
            "same_parameter_count_as_formal_tanh": sum(p.numel() for p in model.parameters()) == ref_params,
        }
        if linear_hash != ref_hash:
            raise RuntimeError(f"Fairness check failed for {method}: initial Linear weights differ.")
        if methods[method]["total_parameters"] != ref_params:
            raise RuntimeError(f"Fairness check failed for {method}: parameter count differs.")

    audit = {
        "seed": SEED,
        "methods": METHODS,
        "training_points_file": str(point_path),
        "training_points_status": point_status,
        "training_points_sha256": point_hash,
        "reference": "formal tanh backbone from the supplied common training code",
        "formal_tanh_linear_sha256": ref_hash,
        "formal_tanh_linear_layers": ref_linear_layers,
        "formal_tanh_total_parameters": ref_params,
        "method_checks": methods,
        "fairness_rule": (
            "same seed, identical Linear initialization, identical network depth/width, "
            "same fixed training points, same PDE/BC/IC loss, same optimizer schedule and budget; "
            "only the fixed activation function differs"
        ),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fairness_audit_seed99.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def main():
    if SEEDS != [99]:
        raise RuntimeError(f"This package must use SEEDS=[99], got {SEEDS}")
    fairness_preflight()
    for method in METHODS:
        print("\n" + "=" * 100)
        print(f"TRAINING {method_label(method)} | seed=99")
        print("=" * 100)
        train_single_seed(method, SEED)


if __name__ == "__main__":
    main()
