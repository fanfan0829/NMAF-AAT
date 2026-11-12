import torch
from ac_adaptive5_common import (
    SEEDS, set_seed, build_model, count_parameters, count_adaptive_parameters,
    method_label,
)

METHODS = ["tanh", "gaaf", "llaaf", "nlaaf", "nmaf_aat"]


def linear_state(model):
    out = {}
    out["input.weight"] = model.input_layer.weight.detach().cpu()
    out["input.bias"] = model.input_layer.bias.detach().cpu()
    for i, layer in enumerate(model.hidden_layers):
        out[f"hidden.{i}.weight"] = layer.weight.detach().cpu()
        out[f"hidden.{i}.bias"] = layer.bias.detach().cpu()
    out["output.weight"] = model.output_layer.weight.detach().cpu()
    out["output.bias"] = model.output_layer.bias.detach().cpu()
    return out

for seed in SEEDS:
    states = {}
    for method in METHODS:
        set_seed(seed)
        model = build_model(method, device=torch.device("cpu"))
        states[method] = linear_state(model)
        print(
            f"seed={seed:4d} | {method_label(method):13s} | "
            f"total={count_parameters(model):7d} | adaptive={count_adaptive_parameters(model):4d}"
        )
    ref = states["tanh"]
    for method in METHODS[1:]:
        same = all(torch.equal(ref[k], states[method][k]) for k in ref)
        if not same:
            raise RuntimeError(f"Linear initialization mismatch: seed={seed}, method={method}")
    print(f"seed={seed}: all five methods have identical Linear initialization.\n")

print("Expected adaptive parameter counts: GAAF=1, L-LAAF=7, N-LAAF=896, NMAF-AAT=256.")
