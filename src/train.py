import torch
from data import get_cifar
from model import SimpleCNN
from muon import MUON
from datetime import datetime
import os

# --- Hyperparameters --- 
batch_size = 1024
learning_rate_sgd = 5e-4
learning_rate_muon = 0.3
print_every = 10
epochs = 20
save_path = "./models/"
momentum_weight_muon = 0.7
momentum_weight_sgd = 0.9
weight_decay_sgd = 1e-6 * batch_size
label_smoothing = 0.2
# -----------------------

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# --- Initialization --- 
trainloader, testloader = get_cifar(batch_size=batch_size)
model = SimpleCNN(num_classes=10).to(device)
loss_function = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='sum')
muon_params = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
sgd_params = [p for p in model.parameters() if p.ndim != 4 and p.requires_grad]
optimizer_muon = MUON(muon_params, learning_rate=learning_rate_muon, momentum_weight=momentum_weight_muon)
optimizer_sgd = torch.optim.SGD(sgd_params, lr=learning_rate_sgd, momentum=momentum_weight_sgd, weight_decay=weight_decay_sgd/learning_rate_sgd) 
total_train_steps = epochs * len(trainloader)
current_step = 0

# --- Train Loop ---
for epoch in range(epochs):
    loss_sum = 0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images) 
        loss = loss_function(logits, labels)

        loss.backward()
        
        # --- Learning Rate Scheduler ---
        lr_scale = 1 - (current_step / total_train_steps)
        for param_group in optimizer_muon.param_groups:
            param_group['lr'] = learning_rate_muon * lr_scale
        for param_group in optimizer_sgd.param_groups:
            param_group['lr'] = learning_rate_sgd * lr_scale
        current_step += 1
        
        optimizer_muon.step()
        optimizer_sgd.step()
        
        model.zero_grad(set_to_none=True)
        
        loss_sum += loss
        if i and i % print_every == 0:
            print(f"Epoch {epoch} / {epochs} | Iteration {i} / {len(trainloader)} | Loss {loss_sum / print_every}")
            loss_sum = 0

# --- Save model ---
os.makedirs(save_path, exist_ok=True)
path = os.path.join(save_path, datetime.now().strftime("%Y-%m-%d") + ".pth")
torch.save(model.state_dict(), path)
print(f"Done! Model Saved to {path}")
