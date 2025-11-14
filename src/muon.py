import torch


def NewtonSchulz(M, n_it):
    assert len(M.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M.bfloat16()

    # Transpose for smaller Gram (intermediate) matrix
    if X.size(0) > X.size(1):
        X = X.T

    # Apply quintic odd polynomial
    for _ in range(n_it):
        A = X@X.T
        B = b*A + c*A@A
        X = a*X + B@X

    # Reverse Transpose
    if X.size(0) > X.size(1):
        X = X.T

    return X


def muon_it(grad, momentum, beta=0.95, ns_it=5):
    # Update momentum
    momentum.lerp_(grad, 1-beta)

    # Normalize and apply Newton-Schulz
    update = momentum / (momentum.norm(p='fro') + 1e-10)
    update = NewtonSchulz(update.reshape(len(update), -1),
                          n_it=ns_it).view(update.shape)

    # Scale gradient by layer's size
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5

    return update


def adam_it(grad, momentum, vel, betas, step):
    momentum.lerp_(grad, (1-betas[0]))
    vel.lerp_(grad.square(), (1-betas[1]))
    m_hat = momentum / (1 - betas[0]**step)
    v_hat = vel / (1 - betas[1]**step)
    update = m_hat / (v_hat.sqrt() + 1e-10)
    return update


class MUON(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    update = muon_it(
                        p.grad,
                        momentum=state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_it=5
                    )
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])

            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["vel_buffer"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    update = adam_it(
                        p.grad,
                        momentum=state["momentum_buffer"],
                        vel=state["vel_buffer"],
                        betas=group["betas"],
                        step=state["step"]
                    )
                    p.add_(update, alpha=-group["lr"])

        return loss
