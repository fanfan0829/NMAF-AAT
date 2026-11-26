import torch
import nlse_common_fixed_points_continuous_lbfgs as core


def linear_state(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ("input_layer." in name or "hidden_layers." in name or "output_layer." in name)
    }


def main():
    print("Seeds:", core.SEEDS)
    print("Residual / BC / IC:", core.N_RES, "/", core.N_BC, "/", core.N_IC)
    print("L-BFGS history size:", core.LBFGS_HISTORY_SIZE)
    print("Checkpoints:", core.CHECKPOINT_EVALS)
    print("Stan beta init:", core.STAN_INITIAL_BETA)

    for seed in core.SEEDS:
        core.set_seed(seed)
        tanh = core.TanhPINN().double().cpu()
        core.set_seed(seed)
        stan = core.StanPINN().double().cpu()
        a = linear_state(tanh)
        b = linear_state(stan)
        assert a.keys() == b.keys()
        for key in a:
            if not torch.equal(a[key], b[key]):
                raise AssertionError(f"Linear initialization differs: seed={seed}, key={key}")
        print(f"seed {seed:4d}: Linear initialization IDENTICAL")

    core.set_seed(core.SEEDS[0])
    tanh = core.TanhPINN().double().cpu()
    core.set_seed(core.SEEDS[0])
    nmaf = core.NMAFAATPINN().double().cpu()
    core.set_seed(core.SEEDS[0])
    stan = core.StanPINN().double().cpu()

    n_tanh = core.count_parameters(tanh)
    n_nmaf = core.count_parameters(nmaf)
    n_stan = core.count_parameters(stan)
    n_beta = sum(a.beta.numel() for a in stan.stan_activations)

    assert len(stan.stan_activations) == core.HIDDEN_LAYERS + 1
    assert n_beta == (core.HIDDEN_LAYERS + 1) * core.HIDDEN_NEURONS
    for activation in stan.stan_activations:
        assert torch.all(activation.beta == core.STAN_INITIAL_BETA)

    print("tanh parameters     :", n_tanh)
    print("NMAF-AAT parameters :", n_nmaf)
    print("Stan parameters     :", n_stan)
    print("NMAF adaptive params:", core.count_adaptive_parameters(nmaf))
    print("Stan beta parameters:", n_beta)
    print("activation locations:", len(stan.stan_activations))


    x = torch.tensor([[0.13], [-0.42]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[0.21], [0.77]], dtype=torch.float64, requires_grad=True)
    y = stan(x, t)
    u = y[:, 0:1]
    ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
    loss = y.square().mean() + uxx.square().mean()
    loss.backward()
    grad_sum = sum(
        float(a.beta.grad.abs().sum())
        for a in stan.stan_activations
        if a.beta.grad is not None
    )
    assert torch.isfinite(uxx).all()
    assert grad_sum > 0.0
    print("Second derivative smoke test: PASS")
    print("Stan beta gradient sum:", grad_sum)
    print("\nFAIRNESS CHECK: PASS")


if __name__ == "__main__":
    main()
