import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from muon import MUON

N_EPOCHS = 10000


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
        print(f"[1001,1002,1003,1004] -> {model(torch.tensor([[1001.0, 1002.0, 1003.0, 1004.0]])).item():.2f} (expected ~1005)")
