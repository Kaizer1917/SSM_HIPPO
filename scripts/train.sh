# scripts/train.sh
#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME="ssm_hippo_v1"

# Train the model
python market_maker/model/train.py \
    --train_dataset $DATA_DIR/processed/market_data.npz \
    --test_dataset $DATA_DIR/processed/market_data.npz \
    --project_name "market-maker-ssm" \
    --model_name $MODEL_NAME \
    --seq_len 96 \
    --forecast_len 32 \
    --num_channels 24 \
    --d_model 128 \
    --n_layer 4 \
    --learning_rate 0.001 \
    --num_epochs 50 \
    --batch_size 32 \
    --checkpoint_dir $MODEL_DIR/checkpoints
