# scripts/train.sh
#!/bin/bash
python model/train.py \
    --train_dataset /app/data/train.npz \
    --test_dataset /app/data/test.npz \
    --project_name "ssm-hippo-project" \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --batch_size 32 \
    --seq_len 96 \
    --forecast_len 96 \
    --input_dim 17 \
    --hidden_dim 128 \
    --num_layers 4