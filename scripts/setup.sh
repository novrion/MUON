#!/bin/bash
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <install type>"
	echo "<install type> : 'minimal', 'normal'"
	echo "Using default 'normal' compute type"
	set -- "normal"
fi

echo "Syncing packages..."
uv venv
uv sync

if [ "$1" = "normal" ]; then
	echo "Setting up Modal..."
	modal setup
fi
