import torch
from data import get_cifar
from model import SimpleCNN
from datetime import datetime
import os

from muon import MUON

# Hyperparameters
batch_size = 256
learning_rate_conv = 5e-4
learning_rate_linear = 5e-4
learning_rate_bias = 5e-4
print_every = 10
epochs = 2
save_path = "./models/"

trainloader, testloader = get_cifar(batch_size=batch_size)
model = SimpleCNN(num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()

# TODO: Are there other layers that are missing here?
conv_params = [p for p in model.parameters() if len(p.shape) == 4]
linear_params = [p for p in model.parameters() if len(p.shape) == 2]
bias_params = [p for p in model.parameters() if len(p.shape) == 1]
optimizer = MUON(
        param_groups=[
            dict(params=conv_params, use_muon=True, lr=learning_rate_conv),
            dict(params=linear_params, use_muon=True, lr=learning_rate_linear),
            dict(params=bias_params, use_muon=False, lr=learning_rate_bias)
        ]
    )

for epoch in range(epochs):
    loss_sum = 0
    for i, (images, labels) in enumerate(trainloader):
        logits = model(images) # Get models predictions
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()
        # TODO: MUON clip? Clip gradient?
        optimizer.zero_grad()

        loss_sum += loss
        if i and i % print_every == 0:
            print(f"Iteration {i} | Loss {loss_sum / print_every}")
            loss_sum = 0

# --- Save model ---
path = os.path.join(save_path, datetime.now().strftime("%Y-%m-%d") + ".pth")
torch.save(model.state_dict(), path)
print(f"Done! Model Saved to {path}")

# TODO: Can implement continue training
