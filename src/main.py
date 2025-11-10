import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

N_EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9


class MUON(torch.optim.Optimizer):
    pass


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 5)

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":

    # CIFAR dataset
    batch_size = 4
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)
                                    )])
    dataset = torchvision.datasets.CIFAR10(root="data", train=True,
                                           download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # Neural net, loss function, and optimizer
    model = NN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, momentum=0.9)

    # Train
    for epoch in range(N_EPOCHS):
        for i, (X, y) in enumerate(dataloader):

            optimizer.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
