# Market Making with SSM-HIPPO

A sophisticated market making system that leverages State Space Models (SSM) with HiPPO operators for dynamic price prediction and market making strategies.

## Features

### Rollercoaster_girls.py Core Components

- **Predict Class**: Advanced prediction system for market making
  - Real-time price movement prediction using SSM-HIPPO
  - Dynamic state management for market conditions
  - Adaptive temporal coherence for price stability
  - Efficient forward-filling for time series alignment

### SSM Integration for Market Making

- **Dynamic State Tracking**:
  - Continuous monitoring of market state variables
  - Adaptive transition matrices for varying market conditions
  - Real-time state updates with new market data

- **Price Prediction Features**:
  - Multi-horizon forecasting capabilities
  - Channel attention for relevant feature selection
  - Patch-based processing for varying timeframes
  - Temporal coherence loss for stable predictions

- **Market Making Optimization**:
  - Dynamic bid-ask spread adjustment
  - State-aware position management
  - Risk-adjusted order sizing
  - Adaptive regularization for market volatility

## Installation

Required dependencies:
```bash
pip install torch einops numpy pandas polars scikit-learn tvm
```

## Usage

### Basic Usage

```python
from market_maker.Rollercoasters_girls import Predict

# Initialize predictor
predictor = Predict(
    lookback_window=96,
    forecast_horizon=32,
    num_features=24
)

# Make predictions
predictions = predictor.forward(market_data)
```

### Advanced Configuration

```python
# Configure SSM-HIPPO parameters
model_args = ModelArgs(
    d_model=128,          # Model dimension
    n_layer=4,            # Number of SSM layers
    seq_len=96,           # Input sequence length
    forecast_len=32,      # Prediction horizon
    num_channels=24,      # Number of market features
    patch_len=16,         # Patch size for processing
    stride=8              # Stride for patch sampling
)

# Initialize with custom configuration
predictor = Predict(
    model_args=model_args,
    use_tvm_optimization=True,
    adaptive_regularization=True
)
```

## Market Making Integration

The system is designed to work with market making strategies by providing:

1. **Real-time Predictions**:
   - Forward-filling missing data points
   - Handling irregular time series
   - Fast inference with TVM optimization

2. **State Management**:
   - Tracking market regimes
   - Managing position exposure
   - Monitoring risk metrics

3. **Adaptive Features**:
   - Dynamic learning rate adjustment
   - Adaptive regularization
   - Progressive state space expansion

## Performance Optimization

The implementation includes several optimizations:

- TVM acceleration for inference
- Efficient memory management
- Vectorized operations
- Adaptive computation paths

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
