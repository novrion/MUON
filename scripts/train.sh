#!/bin/bash
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <compute type>"
	echo "<compute type> : 'local', 'modal'"
	echo "Using default 'local' compute type"
fi

case $1 in
    modal)
		echo "----- Training on Modal compute server -----"

		echo "Training MUON model..."
		uv run modal run scripts/train_modal.py --method "muon"

      	echo "Training AdamW model..."
		uv run modal run scripts/train_modal.py --method "adamw"

		echo "Training Adam model..."
		uv run modal run scripts/train_modal.py --method "adam"
        ;;
    *)
		echo "----- Training on local machine -----"

		echo "Training MUON model..."
      	uv run src/train.py "muon"
      	
      	echo "Training AdamW model..."
      	uv run src/train.py "adamw"
      	
      	echo "Training Adam model..."
      	uv run src/train.py "adam"
        ;;
esac









