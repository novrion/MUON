import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from muon import MUON

N_EPOCHS = 10


def load_cifar(train):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(
        root="./datasets", train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=True, num_workers=2)
    return dataloader


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = NN()
    loss_fn = torch.nn.CrossEntropyLoss()

    conv_params = [p for p in model.parameters() if len(p.shape) == 4]
    linear_params = [p for p in model.parameters() if len(p.shape) == 2]
    bias_params = [p for p in model.parameters() if len(p.shape) == 1]

    optimizer = MUON(
        param_groups=[
            dict(params=conv_params, use_muon=True, lr=0.02),
            dict(params=linear_params, use_muon=True, lr=0.01),
            dict(params=bias_params, use_muon=False, lr=3e-4)
        ]
    )

    # Get dataset
    dataloader = load_cifar(train=True)

    # Train
    for epoch in range(N_EPOCHS):
        running_loss = 0

        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)

            # Logging
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"EPOCH {epoch} | BATCH {i} | Loss: {
                      running_loss / 10:.4f}")
                running_loss = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Test
    trainloader = load_cifar(train=False)
    it = iter(trainloader)
    images, labels = next(it)

    with torch.no_grad():
        logits = model(images)
    _, pred = torch.max(logits, 1)
    score = (pred == labels).sum().item()
    tot = labels.size(0)

    print(f"model accuracy: {score/tot}")
