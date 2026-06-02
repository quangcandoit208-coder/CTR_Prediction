#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/kaggle_t4x2.yaml}"
MODEL="${2:-nafi}"
EPOCHS="${3:-}"
OUTPUT_DIR="$(python -c "import yaml; print((yaml.safe_load(open('$CONFIG')) or {}).get('paths', {}).get('output_dir', 'outputs'))")"
TRAIN_ARGS=(--config "$CONFIG" --model "$MODEL")
if [[ -n "$EPOCHS" ]]; then
  TRAIN_ARGS+=(--epochs "$EPOCHS")
fi

python scripts/prepare_avazu.py --config "$CONFIG"
python scripts/train.py "${TRAIN_ARGS[@]}"
python scripts/evaluate.py --config "$CONFIG" --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pt"
