import torch
from model.mamba import SSM_HIPPO, ModelArgs
from prepare_data import load_time_series_data, TimeSeriesDataset
from train import train_model

def simple_forecast_example():
    """
    Simple example of using SSM-HIPPO for time series forecasting.
    """
    # Define model parameters
    args = ModelArgs(
        d_model=128,          
        n_layer=4,            
        seq_len=96,           
        num_channels=1,       
        patch_len=16,         
        stride=8,             
        forecast_len=24,      
        d_state=16,          
        expand=2,            
        dt_rank='auto'       
    )

    # Initialize model
    model = SSM_HIPPO(args)
    
    # Load and prepare training data
    train_dataset = TimeSeriesDataset(
        seq_length=args.seq_len,
        forecast_length=args.forecast_len
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True
    )
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        epochs=10,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load test data for inference
    x = load_time_series_data(seq_length=args.seq_len, batch_size=2)
    
    # Generate forecast
    with torch.no_grad():
        forecast = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Forecasted values:\n{forecast[0, 0, :5]}")  # Show first 5 predictions

if __name__ == "__main__":
    simple_forecast_example() 
