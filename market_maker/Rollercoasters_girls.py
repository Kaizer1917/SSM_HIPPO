import pandas as pd
import numpy as np
import torch
from typing import Union
import polars as pl
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model.mamba import SSM_HIPPO, ModelArgs
from model.losses import AdaptiveTemporalCoherenceLoss
from model.prepare_data import create_sequences, TimeSeriesDataset, create_optimized_dataloaders
from model.regularization import AdaptiveRegularization

# Constants from original file
NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000

# Feature column names
AQUANTZ = [f"asks[{i}].amount" for i in range(20)]
BQUANTZ = [f"bids[{i}].amount" for i in range(20)]
APRICEZ = [f"asks[{i}].price" for i in range(20)]
BPRICEZ = [f"bids[{i}].price" for i in range(20)]

class MarketPredictor:
    def __init__(self, model_path: str = None):
        # Model configuration
        self.args = ModelArgs(
            d_model=128,          # Model dimension
            n_layer=4,            # Number of SSM-HIPPO blocks
            seq_len=96,           # Input sequence length
            d_state=16,           # State dimension
            num_channels=24,      # Number of input channels
            forecast_len=1,       # Predict next step
            patch_len=16,         # Patch length for processing
            stride=8,             # Stride for patch creation
            sigma=0.5,            # Channel mixup parameter
            reduction_ratio=8     # Channel attention reduction ratio
        )
        
        # Initialize model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SSM_HIPPO(self.args).to(self.device)
        self.criterion = AdaptiveTemporalCoherenceLoss()
        self.regularizer = AdaptiveRegularization(self.model)
        
        if model_path:
            self.load_model(model_path)
        
    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, epochs: int = 50):
        """Train the SSM-HIPPO model on market data."""
        # Prepare features
        train_features = self._prepare_features(train_data)
        val_features = self._prepare_features(val_data)
        
        # Create data loaders
        train_loader, val_loader = create_optimized_dataloaders(
            self.args,
            train_features.values,
            val_features.values,
            train_features.shift(-1).values[:-1],  # Target is next timestep
            val_features.shift(-1).values[:-1]
        )
        
        # Reference training logic from model/train.py
        """
        startLine: 39
        endLine: 145
        """
        
        return self.model

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model."""
        self.model.eval()
        with torch.no_grad():
            x = self._prepare_features(features)
            x = torch.FloatTensor(x.values).to(self.device)
            x = x.unsqueeze(0)  # Add batch dimension
            
            output = self.model(x)
            predictions = output.squeeze().cpu().numpy()
            
            return pd.Series(predictions, index=features.index[1:])

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model using market data."""
        features = []
        
        # Order book features
        features.append(self._calc_imbalance(data))
        features.append(self._calc_vwap_ratio(data))
        features.append(self._calc_trade_flow(data))
        
        # Combine features
        return pd.concat(features, axis=1)

    def _calc_imbalance(self, books_df: pd.DataFrame, depth: int = 5) -> pd.Series:
        """Calculate order book imbalance."""
        ask_volume = sum(books_df[f"asks[{i}].amount"] for i in range(depth))
        bid_volume = sum(books_df[f"bids[{i}].amount"] for i in range(depth))
        return pd.Series(
            (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8),
            index=books_df.index,
            name='imbalance'
        )

    def _calc_vwap_ratio(self, books_df: pd.DataFrame, depth: int = 5) -> pd.Series:
        """Calculate VWAP ratio."""
        ask_weighted = sum(
            books_df[f"asks[{i}].price"] * books_df[f"asks[{i}].amount"]
            for i in range(depth)
        )
        bid_weighted = sum(
            books_df[f"bids[{i}].price"] * books_df[f"bids[{i}].amount"]
            for i in range(depth)
        )
        
        ask_volume = sum(books_df[f"asks[{i}].amount"] for i in range(depth))
        bid_volume = sum(books_df[f"bids[{i}].amount"] for i in range(depth))
        
        vwap = (ask_weighted + bid_weighted) / (ask_volume + bid_volume)
        return pd.Series(
            vwap / books_df["mid_price"],
            index=books_df.index,
            name='vwap_ratio'
        )

    def _calc_trade_flow(self, trades_df: pd.DataFrame, window: int = 10) -> pd.Series:
        """Calculate trade flow balance."""
        window_ns = window * SECOND
        sells = trades_df["ask_amount"].rolling(window=f"{window_ns}ns", min_periods=1).sum()
        buys = trades_df["bid_amount"].rolling(window=f"{window_ns}ns", min_periods=1).sum()
        return pd.Series(
            (sells - buys) / (sells + buys + 1e-8),
            index=trades_df.index,
            name='trade_flow'
        )

def align_with_target_times(array1, times1, target_times):
    """Aligns array1 to the target times using forward filling."""
    df_original = pd.DataFrame({"timestamp": times1, "value": array1})
    df_target = pd.DataFrame({"timestamp": target_times, "value": [np.nan] * len(target_times)})
    
    df_combined = pd.concat([df_original, df_target]).sort_values("timestamp")
    df_combined["value"] = df_combined["value"].ffill()
    
    result = df_combined[df_combined["timestamp"].isin(target_times)]["value"].values
    return pl.Series(result)
