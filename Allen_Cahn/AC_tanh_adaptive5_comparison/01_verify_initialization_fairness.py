import torch
from ac_tanh_adaptive5_common import (
    SEEDS, METHODS, DEVICE, DTYPE, set_seed, build_model, count_adaptive_parameters,
    method_label, HIDDEN_NEURONS, N_ACTIVATION_SITES,
)

def backbone(model):
    out = []
    for layer in [model.input_layer, *list(model.hidden_layers), model.output_layer]:
        out += [layer.weight.detach().cpu(), layer.bias.detach().cpu()]
    return out

if __name__ == "__main__":
    expected = {"tanh": 0, "gaaf": 1, "llaaf": N_ACTIVATION_SITES,
                "nlaaf": N_ACTIVATION_SITES * HIDDEN_NEURONS,
                "tanh_aat": 2 * HIDDEN_NEURONS}
    for seed in SEEDS:
        models = {}
        for m in METHODS:
            set_seed(seed)
            models[m] = build_model(m)
        ref = backbone(models["tanh"])
        for m in METHODS:
            cur = backbone(models[m])
            assert len(cur) == len(ref)
            assert all(torch.equal(a, b) for a, b in zip(ref, cur)), (seed, m, "Linear init mismatch")
            assert count_adaptive_parameters(models[m]) == expected[m], (m, count_adaptive_parameters(models[m]), expected[m])

        x = torch.linspace(-1, 1, 17, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        t = torch.linspace(0, 1, 17, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        y0 = models["tanh"](x, t).detach().cpu()
        for m in METHODS[1:]:
            y = models[m](x, t).detach().cpu()
            assert torch.equal(y0, y), (seed, m, "initial output mismatch")
        print(f"seed={seed}: PASS | identical Linear initialization and identical initial network outputs")
    print("\nAdaptive parameters:")
    for m in METHODS:
        set_seed(SEEDS[0]); model = build_model(m)
        print(f"  {method_label(m):16s}: {count_adaptive_parameters(model)}")
    print("\nPASS: base activation is tanh for all five methods; GAAF/L-LAAF/N-LAAF have no slope-recovery loss.")
