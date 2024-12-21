#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Default parameters
model_name="DLinear"
seq_len=336
batch_size=32
learning_rate=0.001
train_epochs=100
patience=10

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --model)
            model_name="$2"
            shift 2
            ;;
        --seq_len)
            seq_len="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Dataset specific configurations
case $dataset in
    "ETTh1")
        data_path="ETTh1.csv"
        enc_in=7
        features="M"
        pred_lengths=(96 192 336 720)
        learning_rate=0.005
        ;;
    "ETTh2")
        data_path="ETTh2.csv"
        enc_in=7
        features="M"
        pred_lengths=(96 192 336 720)
        learning_rate=0.05
        ;;
    "ETTm1")
        data_path="ETTm1.csv"
        enc_in=7
        features="M"
        pred_lengths=(96 192 336 720)
        learning_rate=0.0001
        batch_size=8
        ;;
    "electricity")
        data_path="electricity.csv"
        enc_in=321
        features="M"
        pred_lengths=(96 192 336 720)
        learning_rate=0.001
        ;;
    *)
        echo "Unknown dataset: $dataset"
        exit 1
        ;;
esac

# Run experiments for each prediction length
for pred_len in "${pred_lengths[@]}"; do
    log_file="logs/LongForecasting/${model_name}_${dataset}_${seq_len}_${pred_len}.log"
    
    python run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path $data_path \
        --model_id ${dataset}_${seq_len}_${pred_len} \
        --model $model_name \
        --data $dataset \
        --features $features \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --des 'Exp' \
        --itr 1 \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience > $log_file
done 