import torch
from data import get_cifar
from model import SimpleCNN
from datetime import datetime
import os

# Hyperparameters 
batch_size = 256
learning_rate = 5e-4
print_every = 10
epochs = 2
save_path = "./models/"

trainloader, testloader = get_cifar(batch_size=batch_size)
model = SimpleCNN(num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # TODO: Bring MUON into the mix


for epoch in range(epochs):
    loss_sum = 0
    for i, (images, labels) in enumerate(trainloader):
        logits = model(images) # Get models predictions
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()
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
