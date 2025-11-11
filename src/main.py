import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

N_EPOCHS = 1000


class MUON(torch.optim.Optimizer):
    pass


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x):
        return self.lin(x)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    # Train
    dataloader = get_dataset()
    for epoch in range(N_EPOCHS):
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)

            # Logging
            if epoch % 100 == 0:
                print(f"Loss: {loss}")

            loss.backward()
            optimizer.step()

    with torch.no_grad():
        print(model(torch.tensor([[1.0, 2.0, 3.0, 4.0]])))
        print(model(torch.tensor([[2.0, 3.0, 4.0, 5.0]])))
        print(model(torch.tensor([[3.0, 4.0, 5.0, 6.0]])))
        print(model(torch.tensor([[40.0, 41.0, 42.0, 43.0]])))
        print(model(torch.tensor([[101.0, 102.0, 103.0, 104.0]])))
