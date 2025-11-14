import torch
import torchvision
from model import SimpleCNN
from data import get_cifar
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Hyperparameters ---
load_path = "./models/2025-11-14.pth"
batch_size = 256
interactive = False

# ---- Initalization ---
model = SimpleCNN(num_classes=10)
model.load_state_dict(torch.load(load_path))
_, testloader = get_cifar(batch_size=batch_size)

if not interactive: 
    # --- Get Test Accuracy --- 
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(testloader)):
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy {correct / total}")

else: # Unsure if works...
    # --- Interactive Testing ---
    for images, labels in testloader:
        logits = model(images)
        
        print(f"Probability Predictions {torch.nn.functional.softmax(logits, dim=1) * 100} %")

        # Visualisation
        images = images / 2 + 0.5 # Unnormalize 
        img_grid = torchvision.utils.make_grid(images)
        npimg = img_grid.numpy()
        plt.figure(figsize=(8,4))
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # From [C, H, W] to [H, W, C] 
        plt.show()

        input("CONTINUE: ")
