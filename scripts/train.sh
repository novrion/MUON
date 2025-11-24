#!/bin/bash

echo "Training MUON model..."
uv run src/train.py "muon"

echo "Training AdamW model..."
uv run src/train.py "adamw"
