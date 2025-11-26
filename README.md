# MUON
An implementation of the MUON optimizer, applied for a classifier of the CIFAR-10 dataset, with tests to compare with AdamW.

## Setup
```bash
# Required packages:
#   - uvicorn
#   - python

# Optional packages (for loss graph):
#   - g++

# Normal install
bash scripts/setup.sh

# No cloud compute
bash scripts/setup.sh minimal
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
