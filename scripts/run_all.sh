#!/bin/bash

# List of models and datasets to evaluate
models=("DLinear" "MoU")
datasets=("ETTh1" "ETTh2" "ETTm1" "electricity")

# Run experiments
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running experiments for $model on $dataset"
        ./scripts/run_experiment.sh --model $model --dataset $dataset
        
        # Evaluate metrics
        ./scripts/evaluate_metrics.sh $model $dataset
    done
done

# Generate final report
python scripts/generate_report.py --results_dir ./results 