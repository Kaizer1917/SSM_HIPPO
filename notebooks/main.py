import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple
import random
from typing import Tuple, List, Optional
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass


class DataLoader:
    @staticmethod
    def load_data(trades_path: str, lob_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and clean trading and limit order book data
        """
        trades = pd.read_csv(trades_path)
        lob = pd.read_csv(lob_path)
        
        # Remove unnamed columns
        trades.drop(columns=["Unnamed: 0"], inplace=True)
        lob.drop(columns=["Unnamed: 0"], inplace=True)
        
        return trades, lob

class OrderBookAnalyzer:
    def __init__(self, lob_data: pd.DataFrame):
        self.lob = lob_data
        self.calculate_mid_price()
    
    def calculate_mid_price(self) -> None:
        """Calculate mid price from best bid and ask"""
        self.lob['mid_price'] = (self.lob['asks[0].price'] + self.lob['bids[0].price']) / 2
    
    def plot_order_book(self, length: int = 20) -> None:
        """
        Plot order book data for specified length
        
        Args:
            length: Number of data points to plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot ask prices
        for i in range(3):
            plt.plot(
                self.lob[f'asks[{i}].price'][-length:],
                label=f'ask{i}',
                color='red',
                alpha=0.7
            )
        
        # Plot mid price
        plt.plot(
            self.lob.mid_price[-length:],
            label='mid price',
            color='black',
            linewidth=2
        )
        
        # Plot bid prices
        for i in range(3):
            plt.plot(
                self.lob[f'bids[{i}].price'][-length:],
                label=f'bid{i}',
                color='green',
                alpha=0.7
            )
        
        plt.title('Order Book Price Levels')
        plt.xlabel('Time')
        plt.ylabel('Price (log scale)')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()

class TimeSeriesPreprocessor:
    """Class for preprocessing time series data."""
    
    def __init__(self, window_size: int, prediction_horizon: int):
        """
        Initialize TimeSeriesPreprocessor.
        
        Args:
            window_size: Size of the sliding window
            prediction_horizon: Number of steps ahead to predict
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
    
    def create_windows(self, data: pd.DataFrame) -> np.ndarray:
        """Create sliding windows from time series data."""
        return np.lib.stride_tricks.sliding_window_view(
            data['mid_price'].values, 
            self.window_size
        )
    
    def prepare_data(self, lob: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare time series data with sliding windows and targets.
        
        Args:
            lob: Limit order book DataFrame
            
        Returns:
            Processed DataFrame with features and targets
        """
        # Create windows and add target
        df = pd.DataFrame(self.create_windows(lob))
        df['target'] = lob['mid_price'].shift(-(self.prediction_horizon + self.window_size))
        
        # Clean and normalize data
        df = df.rename(str, axis="columns")
        df = df[:-self.prediction_horizon-1]
        df = df.apply(self._normalize_row, axis=1, result_type='expand')
        df.rename(columns={str(self.window_size): "target"}, inplace=True)
        
        return df
    
    @staticmethod
    def _normalize_row(row: pd.Series) -> np.ndarray:
        """Normalize a single row using StandardScaler."""
        scaler = StandardScaler()
        return scaler.fit_transform(row.values.reshape(-1, 1)).flatten()

def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        predictions: Predicted values
        targets: Actual values
        
    Returns:
        RMSE value
    """
    mse = np.square(np.subtract(predictions, targets)).mean()
    return np.sqrt(mse)


class LightGBMTrainer:
    """Class for training and evaluating LightGBM models."""
    
    def __init__(self, test_size: int = 10000):
        """
        Initialize LightGBM trainer.
        
        Args:
            test_size: Number of samples to use for testing
        """
        self.test_size = test_size
        self.model = None
        self.feature_cols: List[str] = []
        self.default_params = {
            "boosting_type": "gbdt",
            "metrics": "rmse",
            "objective": "regression",
            "max_depth": 50,
            "learning_rate": 0.15,
            "n_estimators": 3500,
            "colsample_bytree": 0.7,
            "colsample_bynode": 0.7,
            "verbose": -1,
            "random_state": 42,
            "extra_trees": True,
            "num_leaves": 30,
            "n_threads": -1,
        }

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            df: Input DataFrame containing features and target
            
        Returns:
            Tuple of training and test DataFrames
        """
        train_df = df[:-self.test_size]
        test_df = df[-self.test_size:]
        
        self.feature_cols = df.columns.to_list()
        self.feature_cols.remove('target')
        
        return train_df, test_df

    def create_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[lgb.Dataset, lgb.Dataset]:
        """
        Create LightGBM datasets for training.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of training and validation datasets
        """
        lgb_train = lgb.Dataset(
            train_df[self.feature_cols], 
            label=train_df['target']
        )
        lgb_val = lgb.Dataset(
            test_df[self.feature_cols], 
            label=test_df['target'], 
            reference=lgb_train
        )
        return lgb_train, lgb_val

    def train(self, 
             train_df: pd.DataFrame, 
             test_df: pd.DataFrame, 
             custom_params: Dict = None, 
             num_boost_round: int = 200) -> None:
        """
        Train the LightGBM model.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            custom_params: Optional custom parameters for the model
            num_boost_round: Number of boosting rounds
        """
        params = custom_params if custom_params else self.default_params
        lgb_train, lgb_val = self.create_datasets(train_df, test_df)
        
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=num_boost_round,
            callbacks=[lgb.log_evaluation(100)]
        )

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(test_df[self.feature_cols])

    def evaluate(self, test_df: pd.DataFrame, predictions: np.ndarray) -> float:
        """
        Evaluate model performance using RMSE.
        
        Args:
            test_df: Test DataFrame
            predictions: Model predictions
            
        Returns:
            RMSE score
        """
        return np.sqrt(mean_squared_error(test_df['target'], predictions))

    def train_and_evaluate(self, df: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of predictions and RMSE score
        """
        # Prepare data
        train_df, test_df = self.prepare_data(df)
        
        # Train model
        self.train(train_df, test_df)
        
        # Make predictions
        predictions = self.predict(test_df)
        
        # Calculate score
        score = self.evaluate(test_df, predictions)
        
        return predictions, score

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    lob_depth: int = 25
    window_size: int = 50
    prediction_horizon: int = 10
    ma_periods: List[int] = (10, 15, 20)
    hma_periods: List[int] = (10, 20, 30)
    rsi_period: int = 14

class FeatureEngineer:
    """Class for creating features from limit order book data"""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: FeatureConfig object containing parameters
        """
        self.config = config
        self.features_df = pd.DataFrame()
        
    def create_spread_features(self, lob: pd.DataFrame) -> None:
        """Create spread-based features."""
        for i in range(self.config.lob_depth):
            self.features_df[f'spread{i}'] = (
                lob[f'asks[{i}].price'] - lob[f'bids[{i}].price']
            )

    def create_price_difference_features(self, lob: pd.DataFrame) -> None:
        """Create price difference features."""
        # Level-wise price differences
        for i in range(1, self.config.lob_depth):
            self.features_df[f'price_difference_asks{i}'] = abs(
                lob[f'asks[{i}].price'] - lob[f'asks[{i-1}].price']
            )
            self.features_df[f'price_difference_bids{i}'] = abs(
                lob[f'bids[{i}].price'] - lob[f'bids[{i-1}].price']
            )
        
        # Extreme levels difference
        self.features_df['price_difference_asks'] = (
            lob[f'asks[{self.config.lob_depth - 1}].price'] - lob['asks[0].price']
        )
        self.features_df['price_difference_bids'] = (
            lob[f'bids[{self.config.lob_depth - 1}].price'] - lob['bids[0].price']
        )

    def get_column_groups(self) -> Dict[str, List[str]]:
        """Get groups of column names for different levels."""
        return {
            'bids_price': [f'bids[{i}].price' for i in range(self.config.lob_depth)],
            'asks_price': [f'asks[{i}].price' for i in range(self.config.lob_depth)],
            'bids_amount': [f'bids[{i}].amount' for i in range(self.config.lob_depth)],
            'asks_amount': [f'asks[{i}].amount' for i in range(self.config.lob_depth)]
        }

    def create_statistical_features(self, lob: pd.DataFrame) -> None:
        """Create statistical features."""
        columns = self.get_column_groups()
        
        # Mean values
        self.features_df['bids_mean_price'] = lob[columns['bids_price']].mean(axis=1)
        self.features_df['asks_mean_price'] = lob[columns['asks_price']].mean(axis=1)
        self.features_df['bids_mean_amount'] = lob[columns['bids_amount']].mean(axis=1)
        self.features_df['asks_mean_amount'] = lob[columns['asks_amount']].mean(axis=1)
        
        # Accumulated differences
        self.features_df['acc_dif_price'] = (
            lob[columns['asks_price']].sum(axis=1) - 
            lob[columns['bids_price']].sum(axis=1)
        )
        self.features_df['acc_dif_amount'] = (
            lob[columns['asks_amount']].sum(axis=1) - 
            lob[columns['bids_amount']].sum(axis=1)
        )

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for the data."""
        
        def wma(row: pd.Series, period: int) -> float:
            weights = np.arange(period) + 1
            return np.sum(row * weights) / np.sum(weights)

        def hma(row: pd.Series, period: int) -> float:
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            wma1 = wma(row[:half_period], half_period)
            wma2 = wma(row[:period], period)
            return wma(pd.Series([2 * wma1 - wma2]), sqrt_period)

        def rsi(row: pd.Series, period: int = 14) -> float:
            diff = row.diff()
            gains = diff.where(diff > 0, 0)
            losses = -diff.where(diff < 0, 0)
            
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs.iloc[-1]))

        window_data = pd.DataFrame(data)
        results = {}
        
        # Calculate moving averages
        for period in [10, 15, 20]:
            results[f'ma{period}'] = window_data.rolling(period).mean().iloc[-1]
        
        # Calculate Hull Moving Averages
        for period in [10, 20, 30]:
            results[f'hma{period}'] = window_data.apply(
                lambda x: hma(x, period)
            )
        
        # Calculate RSI
        results['rsi14'] = window_data.apply(
            lambda x: rsi(x, 14)
        )
        
        return results

    def create_window_features(self, df: pd.DataFrame) -> None:
        """Create features based on rolling windows."""
        window_cols = [str(i) for i in range(self.config.window_size)]
        
        # Basic statistics
        df['high'] = df[window_cols].max(axis=1)
        df['low'] = df[window_cols].min(axis=1)
        df['mean'] = df[window_cols].mean(axis=1)
        
        # Technical indicators
        technical_indicators = self.calculate_technical_indicators(df[window_cols])
        for name, values in technical_indicators.items():
            df[name] = values

    def engineer_features(self, lob: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to create all features.
        
        Args:
            lob: Limit order book DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create basic features
        self.create_spread_features(lob)
        self.create_price_difference_features(lob)
        self.create_statistical_features(lob)
        
        # Create window-based features
        df = pd.DataFrame(np.lib.stride_tricks.sliding_window_view(
            lob['mid_price'].values, 
            self.config.window_size
        ))
        
        # Add all features to main DataFrame
        df[self.features_df.columns] = self.features_df
        
        # Create window-based features
        self.create_window_features(df)
        
        return df