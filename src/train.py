import torch
import time
import sys
import os

from muon import MUON
from data import get_cifar
from model import SimpleCNN


MUON_TRAIN_METHOD = "muon"
ADAMW_TRAIN_METHOD = "adamw"
ADAM_TRAIN_METHOD = "adam"
TRAIN_METHODS = [MUON_TRAIN_METHOD, ADAMW_TRAIN_METHOD, ADAM_TRAIN_METHOD]


def train(train_method):
    epochs = 40
    print_every = 10
    batch_size = 2048
    learning_rate_muon = 0.02
    learning_rate_adamw = 1e-3
    learning_rate_adam = 1e-3
    momentum_weight_muon = 0.7
    label_smoothing = 0.2

    # --- Initialisation ---
    print("Intialising...")
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available()
                          else "cpu"))

    # Initialise dataset loader, model, and loss function
    trainloader, _ = get_cifar(batch_size=batch_size)
    model = SimpleCNN(num_classes=10).to(device)
    loss_function = torch.nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
        reduction='mean')

    # Initialise optimizers
    if train_method == MUON_TRAIN_METHOD:
        muon_params = [p for p in model.parameters() if p.ndim ==
                       4 and p.requires_grad]
        adamw_params = [p for p in model.parameters() if p.ndim !=
                      4 and p.requires_grad]

        muon_optimizer = MUON(
            muon_params,
            lr=learning_rate_muon,
            momentum_weight=momentum_weight_muon)
        adamw_muon_optimizer = torch.optim.AdamW(
                adamw_params,
                lr=learning_rate_adamw)

    elif train_method == ADAMW_TRAIN_METHOD:
        adamw_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate_adamw)

    elif train_method == ADAM_TRAIN_METHOD:
        adam_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate_adam)

    # --- Training loop ---
    print("Starting training...")

    start_time = time.perf_counter()
    total_train_steps = epochs * len(trainloader)
    current_step = 0

    it = 0
    loss_data = []

    for epoch in range(epochs):
        loss_sum = 0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()

            # --- Learning rate scheduler ---
            lr_scale = 1 - (current_step / total_train_steps)
            current_step += 1

            if train_method == MUON_TRAIN_METHOD:    
                for param_group in muon_optimizer.param_groups:
                    param_group['lr'] = learning_rate_muon * lr_scale
                for param_group in adamw_muon_optimizer.param_groups:
                    param_group['lr'] = learning_rate_adamw * lr_scale

                muon_optimizer.step()
                adamw_muon_optimizer.step()

            elif train_method == ADAMW_TRAIN_METHOD:
                for param_group in adamw_optimizer.param_groups:
                    param_group['lr'] = learning_rate_adamw * lr_scale
                adamw_optimizer.step()

            elif train_method == ADAM_TRAIN_METHOD:
                for param_group in adam_optimizer.param_groups:
                    param_group['lr'] = learning_rate_adam * lr_scale
                adam_optimizer.step()

            model.zero_grad(set_to_none=True)

            loss_sum += loss.item()
            if i and i % print_every == 0:
                print(f"EPOCH {epoch+1}/{epochs} | ITER {i}/{len(trainloader)} | LOSS {loss_sum / print_every:.4f}")
                loss_sum = 0

            it += 1
            loss_data.append((it, loss.item()))

    print(f"Done training in {time.perf_counter() - start_time:.2f}s")
    return model, loss_data


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write("it,loss\n")
        for p in data:
            f.write(f"{p[0]},{p[1]}\n")
    print(f"Loss data saved to {path}")


if __name__ == "__main__":
    train_method = sys.argv[1]
    if train_method not in TRAIN_METHODS:
        print(f"Invalid train method '{
              train_method}' (options: {TRAIN_METHODS})")
        sys.exit(0)
    print(f"Training {train_method}")
    model, data = train(train_method)
    save_model(model, f"./models/{train_method}_model.pth")
    save_data(data, f"./data/{train_method}_loss.csv")
