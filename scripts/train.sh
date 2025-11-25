#!/bin/bash

echo "Training MUON model..."
uv run src/train.py "muon"

echo "Training AdamW model..."
uv run src/train.py "adamw"

echo "Training Adam model..."
uv run src/train.py "adam"
