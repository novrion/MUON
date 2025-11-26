import torch


def newton_schulz(G, n_iter=5):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7770, 2.0315
    M = G.float() / (G.norm() + 1e-7)

    transpose = M.shape[0] > M.shape[1]
    if transpose:
        M = M.T

    for _ in range(n_iter):
        A = M@M.T
        B = b*A + c*A@A
        M = a*M + B@M

    if transpose:
        M = M.T

    return M


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
