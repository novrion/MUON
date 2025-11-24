#!/bin/bash

echo "Testing MUON model..."
uv run src/test.py models/muon_model.pth

echo "Testing AdamW model..."
uv run src/test.py models/adamw_model.pth
