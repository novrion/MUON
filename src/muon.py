import torch


def newton_schulz(P, n_iter=5):
    assert P.ndim == 2
    a, b, c = 3.4445, -4.7770, 2.0315
    G = P.float() / (P.norm() + 1e-7)

    transpose = G.shape[0] > G.shape[1]
    if transpose:
        G = G.T

    for _ in range(n_iter):
        A = G@G.T
        B = b*A + c*A@A
        G = a*G + B@G

    if transpose:
        G = G.T

    return G


def muon_it(grad, momentum, beta):
    momentum.mul_(beta).add_(grad)
    update = momentum.reshape(len(momentum), -1) if momentum.ndim == 4 else momentum
    return newton_schulz(update)


class MUON(torch.optim.Optimizer):
    def __init__(self, params, momentum_weight, lr):
        super().__init__(params, dict(lr=lr, momentum_weight=momentum_weight))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum_weight = group["momentum_weight"]

            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

                # Initialise momentum
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                # MUON algorithm
                orthogonalized = muon_it(
                    p.grad,
                    state["momentum_buffer"],
                    beta=momentum_weight
                )

                # Update weights
                p.add_(orthogonalized.reshape(p.shape), alpha=-lr)
