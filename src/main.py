import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

N_EPOCHS = 1000


def NewtonSchulz(M, n_it):
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Transpose for smaller Gram (intermediate) matrix
    if M.size(0) > M.size(1):
        M = M.T

    # Apply quintic odd polynomial
    for _ in range(n_it):
        A = M@M.T
        B = b*A + c*A@A
        M = a*M + B@M

    # Reverse Transpose
    if M.size(0) > M.size(1):
        M = M.T

    return M


def muon_it(grad, momentum, beta=0.95, ns_it=5):
    momentum.lerp_(grad, 1-beta)
    normalized = momentum / (momentum.norm(p='fro') + 1e-10)
    orthogonalized = NewtonSchulz(normalized, n_it=ns_it)
    orthogonalized *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return orthogonalized


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
            with torch.enable_grad:
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    orthogonalized = muon_it(
                        p.grad,
                        momentum=state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_it=5
                    )
                    p.add_(orthogonalized.reshape(p.shape), alpha=-group["lr"])

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


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


def get_dataset():
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0],
                      [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0],
                      [100.0, 101.0, 102.0, 103.0]])
    y = torch.tensor([[5.0], [6.0], [7.0], [11.0], [104.0]])
    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader


if __name__ == "__main__":

    # Neural net, loss function, and optimizer
    model = NN()
    loss_fn = nn.MSELoss()
    optimizer = MUON(
        param_groups=[
            dict(
                params=[p for p in model.parameters() if p.ndim >= 2],
                use_muon=True,
                lr=0.002,
            ),
            dict(
                params=[p for p in model.parameters() if p.ndim < 2],
                use_muon=False,
                lr=3e-4,
            ),
        ]
    )

    # Train
    dataloader = get_dataset()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0

        for X, y in dataloader:
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)

            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Logging
        if epoch % 100 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Loss: {avg_loss:.6f}")

    with torch.no_grad():
        print("\nPredictions:")
        print(f"[1,2,3,4] -> {model(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).item():.2f} (expected ~5)")
        print(f"[2,3,4,5] -> {model(torch.tensor([[2.0, 3.0, 4.0, 5.0]])).item():.2f} (expected ~6)")
        print(f"[3,4,5,6] -> {model(torch.tensor([[3.0, 4.0, 5.0, 6.0]])).item():.2f} (expected ~7)")
        print(f"[40,41,42,43] -> {model(torch.tensor([[40.0, 41.0, 42.0, 43.0]])).item():.2f} (expected ~44)")
        print(f"[101,102,103,104] -> {model(torch.tensor([[101.0, 102.0, 103.0, 104.0]])).item():.2f} (expected ~105)")
