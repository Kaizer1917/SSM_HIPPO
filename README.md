# SSM_HIPPO

## Features

- **Channel Mixup**: A data augmentation technique to enhance the robustness of the model.
- **Channel Attention**: Mechanism to dynamically weigh different channels based on their importance.
- **Patch-based Input Processing**: Efficiently handles long sequences by splitting them into patches.
- **Selective State-Space Model (SSM)**: Implements advanced state-space modeling for capturing long-range dependencies.
- **Dynamic HiPPO Transitions**: Implements various HiPPO operators including:
  - Legendre (translated and scaled)
  - Laguerre (translated and generalized)
  - Fourier-based transitions
  - LMU equivalents
- **Adaptive State Space Modeling**: 
  - Optimizes transition matrices during training
  - Supports progressive dimension expansion
  - Dynamic scaling based on training progress
- **Selective State-Space Model (SSM)**:
  - Implements NPLR (Normal Plus Low-Rank) form of HiPPO matrices
  - Efficient state-space computations with selective scanning
  - Supports multiple measure types (legs, legt, fourier, etc.)
- **Advanced Training Features**:
  - Gradient accumulation for large batch training
  - Mixed precision training support
  - Distributed training capabilities
  - Custom learning rate scheduling
- **Model Optimization**:
  - Model pruning and quantization options
  - Memory-efficient inference
  - ONNX export support
  - TorchScript compatibility

## Installation

To use this model, you need to have the following libraries installed:
- `torch`
- `einops`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `wandb`
-  `tvm (optional)`

You can install them using pip:

```bash
pip install torch einops numpy pandas scikit-learn matplotlib wandb
```

## Data Preparation

Before training the SSM_HIPPO model, you'll need to prepare your data. We provide a data preparation script that preprocesses your time series data, normalizes the features, and splits it into training and test sets.

### Data Preparation Script

The data preparation script performs the following steps:
1. Loads the input JSON file containing the time series data.
2. Converts timestamps to datetime format, removes duplicates, and sorts the data.
3. Selects numerical columns as features.
4. Normalizes the features using `StandardScaler`.
5. Creates input-output sequences based on specified input and forecast lengths.
6. Splits the data into training and test sets.
7. Saves the processed data to an NPZ file.

### Usage

To use the data preparation script, save it as `prepare_data.py` and run it from the command line with the appropriate arguments. Here are the arguments you need to provide:

- `--input_file`: Path to the input JSON file containing the time series data.
- `--output_file`: Path to the output file (NPZ format) where the processed data will be saved.
- `--input_length`: Length of the input sequences.
- `--forecast_length`: Length of the forecast sequences.
- `--test_size`: Proportion of the dataset to include in the test split (default is 0.2).

### Example Command

```python
chmod +x scripts/prepare_data.sh
./scripts/prepare_data.sh
```

This command will load the data from exported_data_transformed.json, process it, and save the training and test sets to prepared_data.npz.

## Training

After preparing your data, you can train the SSM_HIPPO model using the training script. The training script initializes the model, sets up the loss function and optimizer, and handles the training loop with logging and model saving.

### Training Script

The training script performs the following steps:

1. Initializes the WandB run for tracking.
2. Initializes the model, loss function, and optimizer.
3. Loads the training and validation data.
4. Runs the training loop with gradient clipping and loss calculation.
5. Logs various metrics, including batch loss, epoch loss, gradient norms, and weight statistics.
6. Saves the model checkpoints and final model.

#### Usage

To use the training script, save it as train_ssm_hippo.py and run it from the command line with the appropriate arguments. Here are the arguments you need to provide:

- `--train_dataset`: Path to the training dataset (NPZ format).
- `--test_dataset`: Path to the test dataset (NPZ format).
- `--project_name`: WandB project name.
- `--learning_rate`: Learning rate for the optimizer (default is 0.- 001).
- `--num_epochs`: Number of epochs for training (default is 10).
- `--batch_size`: Batch size for training (default is 32).
- `--seq_len`: Length of the input sequences.
- `--forecast_len`: Length of the forecast sequences.
- `--input_dim`: Input dimension for the model.
- `--hidden_dim`: Hidden dimension for the model.
- `--num_layers`: Number of layers in the model.


#### Example Command

```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

This command will train the SSM_HIPPO model using the provided parameters and log the training process to WandB.

### Training Script Arguments

Additional training arguments now include:

- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (default is 1)
- `--mixed_precision`: Enable mixed precision training (default is False)
- `--distributed`: Enable distributed training (default is False)
- `--num_workers`: Number of data loading workers (default is 4)
- `--weight_decay`: Weight decay for optimizer (default is 0.01)
- `--warmup_steps`: Number of warmup steps for learning rate scheduler
- `--max_grad_norm`: Maximum gradient norm for clipping

#### Example Command with Advanced Options

```bash
./scripts/train.sh \
    --mixed_precision true \
    --gradient_accumulation_steps 4 \
    --distributed true \
    --num_workers 8 \
    --warmup_steps 1000
```

## Model Export and Optimization

To optimize the model for production deployment, use the following scripts:

```bash
# Export to ONNX
python scripts/export_onnx.py --model_path checkpoints/model.pt --output_path model.onnx

# Quantize model
python scripts/quantize.py --model_path checkpoints/model.pt --output_path model_quantized.pt

# Benchmark inference
python scripts/benchmark.py --model_path model.onnx
```

An example of using the model in a market-making system is presented in the repository - https://github.com/Kaizer1917/Market_maker_ssm_hippo
