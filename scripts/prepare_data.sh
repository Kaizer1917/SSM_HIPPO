# scripts/prepare_data.sh
#!/bin/bash
python model/prepare_data.py \
    --input_file /app/data/raw_data.json \
    --output_file /app/data/processed_data.npz \
    --input_length 96 \
    --forecast_length 96