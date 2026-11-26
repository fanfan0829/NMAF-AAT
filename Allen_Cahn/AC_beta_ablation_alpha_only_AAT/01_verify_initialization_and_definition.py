import torch
from ac_beta_ablation_common import (
    SEEDS, HIDDEN_NEURONS, N_ACTIVATION_SITES,
    set_seed, build_alpha_only_model, build_full_aat_model,
    backbone_state, count_adaptive_parameters,
)

for seed in SEEDS:
    set_seed(seed)
    alpha_only = build_alpha_only_model(device=torch.device("cpu"))
    set_seed(seed)
    full = build_full_aat_model(device=torch.device("cpu"))

    a_backbone = backbone_state(alpha_only)
    f_backbone = backbone_state(full)
    assert a_backbone.keys() == f_backbone.keys()
    for key in a_backbone:
        if not torch.equal(a_backbone[key], f_backbone[key]):
            raise AssertionError(f"Linear initialization mismatch for seed={seed}, key={key}")

    if "activation.beta" in alpha_only.state_dict():
        raise AssertionError("Alpha-only model unexpectedly contains beta")
    if "activation.beta" not in full.state_dict():
        raise AssertionError("Full AAT model is missing beta")

    assert alpha_only.activation.alpha.numel() == HIDDEN_NEURONS
    assert full.activation.alpha.numel() == HIDDEN_NEURONS
    assert full.activation.beta.numel() == HIDDEN_NEURONS

    print(
        f"seed={seed}: Linear initialization identical; "
        f"alpha-only adaptive params={count_adaptive_parameters(alpha_only)}, "
        f"full AAT adaptive params={count_adaptive_parameters(full)}"
    )

print(f"\nArchitecture: {N_ACTIVATION_SITES} hidden activation sites x {HIDDEN_NEURONS} neurons")
print("PASS: alpha sharing is identical; the alpha-only model differs only by removing beta.")
