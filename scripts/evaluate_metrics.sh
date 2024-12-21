#!/bin/bash

# Create results directory if it doesn't exist
if [ ! -d "./results" ]; then
    mkdir ./results
fi

# Parse arguments
model_name=$1
dataset=$2

# Collect metrics from logs
python scripts/collect_metrics.py \
    --log_dir logs/LongForecasting \
    --model $model_name \
    --dataset $dataset \
    --output_file results/${model_name}_${dataset}_metrics.csv 