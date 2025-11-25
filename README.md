# MUON
An implementation of the MUON optimizer, applied for a classifier of the CIFAR-10 dataset, with tests to compare with AdamW.

## Setup
```bash
# Required packages:
#   - uvicorn
#   - python

# Optional packages (for loss_graph):
#   - g++

bash scripts/setup.sh
```

## Run
```bash
# Train
bash scripts/train.sh

# Test
bash scripts/test.sh

# Create LaTeX graph of loss
bash scripts/loss_graph.sh
```
