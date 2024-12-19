#!/bin/bash

# Set environment variables
export MODEL_CHECKPOINT="$MODEL_DIR/checkpoints/best_model.pth"
export CONFIG_FILE="/app/config/market_maker_config.json"

# Run market maker
python market_maker/Rollercoasters_girls.py \
    --model_checkpoint $MODEL_CHECKPOINT \
    --config $CONFIG_FILE \
    --mode live \
    --risk_limit 1000 \
    --max_position 100 \
    --spread_multiplier 1.5 \
    --update_interval 1.0
