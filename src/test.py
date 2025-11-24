from tqdm import tqdm
import torch
import sys

from data import get_cifar
from model import SimpleCNN

# --- Hyperparameters ---
load_path = sys.argv[1]
batch_size = 256

# --- Initialisation ---
device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available()
                      else "cpu"))

model = SimpleCNN(num_classes=10).to(device)
try:
    model.load_state_dict(torch.load(load_path))
except Exception:
    print(f"Could not load model from '{load_path}'")
    sys.exit(0)

_, testloader = get_cifar(batch_size=batch_size)

# --- Test accuracy ---
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in tqdm(enumerate(testloader)):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test accuracy {correct / total}")
