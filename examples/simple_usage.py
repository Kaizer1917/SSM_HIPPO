import torch
from model.mamba import SSM_HIPPO, ModelArgs
from prepare_data import load_time_series_data

def simple_forecast_example():
    """
    Simple example of using SSM-HIPPO for time series forecasting.
    """
    # Define model parameters
    args = ModelArgs(
        d_model=128,          # Model dimension
        n_layer=4,            # Number of layers
        seq_len=96,           # Input sequence length
        num_channels=1,       # Number of input features
        patch_len=16,         # Length of each patch
        stride=8,             # Stride between patches
        forecast_len=24,      # Number of steps to forecast
        d_state=16,          # State dimension
        expand=2,            # Expansion factor
        dt_rank='auto'       # Rank for delta projection
    )

    # Initialize model
    model = SSM_HIPPO(args)
    
    # Load real data instead of random values
    x = load_time_series_data(seq_length=args.seq_len, batch_size=2)
    
    # Generate forecast
    with torch.no_grad():
        forecast = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Forecasted values:\n{forecast[0, 0, :5]}")  # Show first 5 predictions

if __name__ == "__main__":
    simple_forecast_example() 
