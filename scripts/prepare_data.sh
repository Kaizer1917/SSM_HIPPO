# scripts/prepare_data.sh
#!/bin/bash

# Set environment variables
export DATA_DIR="/app/data"
export MODEL_DIR="/app/models"

# Create necessary directories
mkdir -p $DATA_DIR/raw
mkdir -p $DATA_DIR/processed
mkdir -p $MODEL_DIR/checkpoints

# Download and prepare market data
python market_maker/model/prepare_data.py \
    --input_file $DATA_DIR/raw/market_data.json \
    --output_file $DATA_DIR/processed/market_data.npz \
    --input_length 96 \
    --forecast_length 32 \
    --features "price,volume,spread,depth" \
    --normalize true \
    --split_ratio 0.8
