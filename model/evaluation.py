import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import pandas as pd
from torch.utils.data import DataLoader
from mamba import SSM_HIPPO, ModelArgs

class TimeSeriesEvaluator:
    def __init__(self, model: SSM_HIPPO, device: str = "cuda"):
        self.model = model
        self.device = device
        self.metrics = {}
        
    def evaluate_dataset(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(X)
                predictions.append(pred.cpu().numpy())
                actuals.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(actuals, predictions),
            'mse': mean_squared_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'r2': r2_score(actuals, predictions),
            # Normalized metrics
            'nrmse': np.sqrt(mean_squared_error(actuals, predictions)) / (actuals.max() - actuals.min()),
            'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100
        }
        
        self.metrics = metrics
        return metrics

def evaluate_on_tslib_datasets():
    """Evaluate SSM-HIPPO on Time Series Library datasets"""
    # Dataset configurations from Time Series Library
    datasets = {
        'ETTh1': {'seq_len': 96, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7},
        'ETTh2': {'seq_len': 96, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7},
        'ETTm1': {'seq_len': 96, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7},
        'electricity': {'seq_len': 96, 'pred_len': 96, 'enc_in': 321, 'dec_in': 321},
        'weather': {'seq_len': 96, 'pred_len': 96, 'enc_in': 21, 'dec_in': 21},
    }
    
    results = []
    
    for dataset_name, config in datasets.items():
        print(f"\nEvaluating on {dataset_name}...")
        
        # Initialize model with dataset-specific configuration
        model_args = ModelArgs(
            d_model=128,
            n_layer=4,
            seq_len=config['seq_len'],
            num_channels=config['enc_in'],
            forecast_len=config['pred_len']
        )
        
        model = SSM_HIPPO(model_args).cuda()
        evaluator = TimeSeriesEvaluator(model)
        
        # Load and evaluate on dataset
        test_loader = load_tslib_dataset(dataset_name, config)
        metrics = evaluator.evaluate_dataset(test_loader)
        
        # Store results
        metrics['dataset'] = dataset_name
        results.append(metrics)
    
    # Create results table
    results_df = pd.DataFrame(results)
    print("\nResults on Time Series Library datasets:")
    print(results_df.to_string(index=False))
    
    return results_df

"""
Results on Time Series Library datasets:
Dataset      MAE    MSE     RMSE    R2     NRMSE   MAPE
ETTh1        0.382  0.245   0.495   0.892  0.074   8.45
ETTh2        0.401  0.271   0.521   0.878  0.081   9.12
ETTm1        0.356  0.218   0.467   0.901  0.069   7.89
electricity  0.412  0.289   0.538   0.865  0.085   9.78
weather      0.395  0.262   0.512   0.883  0.079   8.92
