import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

def create_sequences(data, input_length, forecast_length):
    sequences = []
    targets = []
    for i in range(len(data) - input_length - forecast_length):
        seq = data[i:i + input_length]
        target = data[i + input_length:i + input_length + forecast_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def enhanced_preprocessing(data, args):
    # Reference existing prepare_data.py
    """
    startLine: 17
    endLine: 57
    """
    
    # Add new preprocessing steps
    def remove_outliers(features, targets, threshold=3):
        z_scores = np.abs((features - features.mean()) / features.std())
        mask = (z_scores < threshold).all(axis=1)
        return features[mask], targets[mask]
    
    def add_temporal_features(sequences):
        # Add time-based features
        time_features = []
        for seq in sequences:
            # Calculate temporal statistics
            rolling_mean = np.convolve(seq, np.ones(3)/3, mode='valid')
            rolling_std = np.array([np.std(seq[i:i+3]) for i in range(len(seq)-2)])
            
            # Add features
            time_features.append(np.concatenate([
                rolling_mean[..., None],
                rolling_std[..., None]
            ], axis=-1))
            
        return np.array(time_features)
    
    # Apply enhanced preprocessing
    sequences, targets = create_sequences(data, args.input_length, args.forecast_length)
    sequences, targets = remove_outliers(sequences, targets)
    temporal_features = add_temporal_features(sequences)
    
    # Combine original and temporal features
    enhanced_sequences = np.concatenate([sequences, temporal_features], axis=-1)
    
    return enhanced_sequences, targets

def main(args):
    # Load the data
    df = pd.read_json(args.input_file)
    
    # Convert timestamp to datetime, remove duplicates, and sort
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop_duplicates(subset='Timestamp', keep='first')
    df = df.sort_values('Timestamp')
    print(f"Number of unique timestamps: {len(df['Timestamp'].unique())}")

    # Select numerical columns for features
    feature_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    print(f"Numerical feature columns: {feature_columns}")
    features = df[feature_columns].values
    print(f"Selected {len(feature_columns)} numerical features")

    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Check if we have enough data points to create sequences
    if len(scaled_features) < args.input_length + args.forecast_length:
        raise ValueError(f"Not enough data points to create sequences. Have {len(scaled_features)}, need at least {args.input_length + args.forecast_length}.")

    # Create sequences
    sequences, targets = create_sequences(scaled_features, args.input_length, args.forecast_length)
    print(f"Generated {len(sequences)} sequences.")

    # Split into input (X) and output (y)
    X = sequences
    y = targets

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Save the preprocessed data
    np.savez(args.output_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"Data saved to {args.output_file}")

class OptimizedDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
        # Pre-calculate indices
        self.valid_indices = torch.arange(
            self.sequence_length, 
            len(self.data) - self.sequence_length
        )
        
        if torch.cuda.is_available():
            self.data = self.data.pin_memory()
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        sequence = self.data[real_idx-self.sequence_length:real_idx]
        target = self.data[real_idx:real_idx+self.sequence_length]
        return sequence, target
    
    def __len__(self):
        return len(self.valid_indices)

def create_optimized_dataloaders(args, dataset, batch_size=32):
    train_loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    return train_loader

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation script for time series forecasting.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file (npz format).')
    parser.add_argument('--input_length', type=int, required=True, help='Length of the input sequences.')
    parser.add_argument('--forecast_length', type=int, required=True, help='Length of the forecast sequences.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')

    args = parser.parse_args()
    main(args)
