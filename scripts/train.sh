# scripts/train.sh
#!/bin/bash
python model/train.py \
    --train_dataset /app/data/train.npz \
    --test_dataset /app/data/test.npz \
    --project_name "ssm-hippo-project" \
     --seq_len 100 \
    --forecast_len 24 \
    --input_dim 1 \
    --hidden_dim 64 \
    --num_layers 4 \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --catboost_iterations 1000 \
    --catboost_learning_rate 0.03 \
    --catboost_depth 6
