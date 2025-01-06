import torch
import torch.nn as nn
import threading
from queue import Queue
import torch.multiprocessing as mp
from typing import List, Dict
from model.mamba import SSM_HIPPO, ModelArgs
from model.losses import EnhancedSSMHiPPOLoss
from model.regularization import AdaptiveRegularization
from model.prepare_data import (
    load_and_preprocess_data,
    create_sequences,
    normalize_data,
    split_data,
    create_data_loaders
)
from model.train import (
    train_model,
    validate_model,
    EarlyStopping,
    LRScheduler,
    TrainingLogger
)
import numpy as np
import matplotlib.pyplot as plt
from model.mamba_tvm import SSMHippoModule
from model.mamba_tvm_block import MambaBlockTVM
from model.mamba_tvm_utils import get_optimal_target, optimize_for_hardware
from model.mamba_tvm_memory import optimize_memory_layout
import tvm
from tvm import relax
from torch.cuda import amp
import torch.nn.functional as F
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import psutil
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from contextlib import nullcontext
import time
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from model.ensemble_data import TimeSeriesEnsembleData
from model.mamba_power_distributor import PowerDistributor, create_power_managed_executor
from model.mamba_tvm_utils import optimize_ssm_hippo_hardware

class AdvancedSSMHiPPO:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with advanced configurations
        self.model_args = ModelArgs(
            d_model=config['d_model'],
            n_layer=config['n_layer'],
            seq_len=config['seq_len'],
            num_channels=config['num_channels'],
            patch_len=config['patch_len'],
            stride=config['stride'],
            forecast_len=config['forecast_len'],
            d_state=config['d_state'],
            expand=config['expand'],
            dt_rank=config['dt_rank'],
            sigma=config['sigma'],
            reduction_ratio=config['reduction_ratio']
        )
        
        # Initialize model components
        self.model = SSM_HIPPO(self.model_args).to(self.device)
        self.model = AdaptiveRegularization(self.model, 
                                          dropout_rate=config['dropout_rate'],
                                          l1_factor=config['l1_factor'],
                                          l2_factor=config['l2_factor'])
        
        self.criterion = EnhancedSSMHiPPOLoss(
            alpha=config['loss_alpha'],
            beta=config['loss_beta'],
            gamma=config['loss_gamma']
        )
        
        # Initialize optimizer with layer-wise learning rates
        param_groups = []
        for i, layer in enumerate(self.model.model.layers):
            param_groups.append({
                'params': layer.parameters(),
                'lr': config['learning_rate'] * (0.9 ** i)
            })
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Initialize TVM optimization components
        self.use_tvm = config.get('use_tvm', True)
        if self.use_tvm:
            # Get optimal target for current hardware
            self.target = get_optimal_target()
            
            # Initialize TVM module
            self.tvm_module = SSMHippoModule
            self.tvm_module = optimize_for_hardware(self.tvm_module, self.target)
            
            # Replace standard Mamba blocks with TVM-optimized versions
            for i, layer in enumerate(self.model.model.layers):
                self.model.model.layers[i] = MambaBlockTVM(self.model_args)

        # Add mixed precision training support
        self.scaler = amp.GradScaler()
        
        # Add caching for data preprocessing
        self.prepare_data = lru_cache(maxsize=32)(self.prepare_data)
        
        # Optimize memory usage
        self.memory_threshold = 0.9  # 90% memory usage threshold

        # Add distributed training support
        self.distributed = config.get('distributed', False)
        if self.distributed:
            self.local_rank = config.get('local_rank', 0)
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl')
            self.model = DDP(self.model, device_ids=[self.local_rank])

        
        self.amp_context = (
            amp.autocast(device_type='cuda', dtype=torch.float16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        # Add ensemble data augmentation
        self.ensemble_data = TimeSeriesEnsembleData(
            noise_std=config.get('noise_std', 0.01),
            time_warp_factor=config.get('time_warp_factor', 0.2),
            crop_ratio=config.get('crop_ratio', 0.9),
            jitter_std=config.get('jitter_std', 0.03),
            max_segments=config.get('max_segments', 5)
        )

        # Add power management components
        self.power_distributor = PowerDistributor(
            total_memory=config.get('total_memory', None),
            power_budget=config.get('power_budget', None),
            num_threads=config.get('num_threads', None)
        )
        
        # Create power-managed forward pass
        self.power_managed_forward = create_power_managed_executor(
            self.model, 
            device_type=self.device
        )
        
        # Hardware optimization
        if config.get('optimize_hardware', True):
            self.lib, self.memory_pool = optimize_ssm_hippo_hardware(
                self.model,
                input_shape=(config['batch_size'], config['num_channels'], config['seq_len']),
                device_type=self.device
            )

    @staticmethod
    def get_memory_usage():
        return psutil.Process().memory_percent()

    def prepare_data(self, data, args):
        """Prepare data with enhanced preprocessing and ensemble augmentation"""
        sequences, targets = enhanced_preprocessing(data, args)
        
        # Create initial data loaders
        train_loader, val_loader = create_optimized_dataloaders(
            args, sequences, targets, 
            batch_size=self.config['batch_size']
        )
        
        # Apply ensemble data augmentation if enabled
        if self.config.get('use_ensemble_data', True):
            # Create augmented training dataset
            augmented_dataset = self.ensemble_data.create_ensemble_dataset(
                train_loader,
                num_augmentations=self.config.get('num_augmentations', 3)
            )
            
            # Create new training loader with augmented data
            train_loader = torch.utils.data.DataLoader(
                augmented_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
        
        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch, total_epochs):
        """Optimized training with mixed precision and distributed training"""
        self.model.train()
        total_loss = 0
        
        # Use tqdm for progress tracking
        from tqdm import tqdm
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for i, (X_batch, y_batch) in enumerate(progress_bar):
            # Fix: Move prefetching inside try-except to handle StopIteration
            try:
                if i + 1 < len(train_loader):
                    next_batch = next(iter(train_loader))
                    next_X = next_batch[0].to(self.device, non_blocking=True)
                    next_y = next_batch[1].to(self.device, non_blocking=True)
            except StopIteration:
                pass
            
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            
            # Use mixed precision training context
            with self.amp_context:
                if self.use_tvm:
                    X_batch = optimize_memory_layout(X_batch, layout="NCHW")
                
                self.optimizer.zero_grad(set_to_none=True)
                output, reg_loss = self.model(X_batch, (epoch + i/len(train_loader)) / total_epochs)
                loss = self.criterion(output, y_batch)
                total_loss = loss + reg_loss
            
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': total_loss.item()})

        return total_loss.item() / len(train_loader)

    def evaluate(self, val_loader):
        """Evaluate model with multiple metrics"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                output, _ = self.model(X_batch, 1.0)
                loss = self.criterion(output, y_batch)
                
                total_loss += loss.item()
                predictions.append(output.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        metrics = {
            'mse': np.mean((predictions - actuals) ** 2),
            'mae': np.mean(np.abs(predictions - actuals)),
            'rmse': np.sqrt(np.mean((predictions - actuals) ** 2))
        }
        
        return total_loss / len(val_loader), metrics

    def plot_forecast(self, x, y_true, forecast_steps=24):
        """Plot forecasting results with confidence intervals"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
            output, _ = self.model(x, 1.0)
            prediction = output.cpu().numpy()[0]
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(range(len(y_true)-forecast_steps, len(y_true)), 
                prediction[-forecast_steps:], 
                label='Forecast', 
                color='red')
        plt.fill_between(range(len(y_true)-forecast_steps, len(y_true)),
                        prediction[-forecast_steps:] - prediction[-forecast_steps:].std(),
                        prediction[-forecast_steps:] + prediction[-forecast_steps:].std(),
                        color='red', alpha=0.2)
        plt.legend()
        plt.title('SSM-HIPPO Forecast with Confidence Interval')
        plt.show()

    def plot_attention_weights(self, x):
        """Visualize attention weights across layers"""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
            attention_weights = self.model.get_attention_weights(x)
            
            plt.figure(figsize=(15, 5 * len(attention_weights)))
            for i, weights in enumerate(attention_weights):
                plt.subplot(len(attention_weights), 1, i + 1)
                plt.imshow(weights.cpu().numpy(), aspect='auto')
                plt.colorbar()
                plt.title(f'Layer {i+1} Attention Weights')
            plt.tight_layout()
            plt.show()

    def plot_feature_importance(self, x, feature_names=None):
        """Analyze and visualize feature importance"""
        importances = self.model.calculate_feature_importance(x)
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        if feature_names:
            plt.xticks(range(len(importances)), feature_names, rotation=45)
        plt.title('Feature Importance Analysis')
        plt.show()

    def analyze_performance(self, val_loader):
        """Comprehensive performance analysis"""
        metrics = {
            'mse': [],
            'mae': [],
            'rmse': [],
            'mape': [],
            'latency': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                start_time = time.time()
                output, _ = self.model(X_batch.to(self.device), 1.0)
                latency = time.time() - start_time
                
                pred = output.cpu().numpy()
                true = y_batch.numpy()
                
                metrics['mse'].append(np.mean((pred - true) ** 2))
                metrics['mae'].append(np.mean(np.abs(pred - true)))
                metrics['rmse'].append(np.sqrt(np.mean((pred - true) ** 2)))
                metrics['mape'].append(np.mean(np.abs((true - pred) / true)) * 100)
                metrics['latency'].append(latency)
        
        return {k: np.mean(v) for k, v in metrics.items()}

    def optimize_hyperparameters(self, train_data, val_data, param_grid, n_splits=5):
        """Optimize hyperparameters using grid search with k-fold cross validation
        
        Args:
            train_data: Training data
            val_data: Validation data 
            param_grid: Dictionary of parameters to search
            n_splits: Number of cross-validation folds
        
        Returns:
            best_params: Dictionary of best parameters
            best_val_loss: Best validation loss achieved
        """
        best_val_loss = float('inf')
        best_params = None
        
        # Create k-fold cross validator
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for params in ParameterGrid(param_grid):
            cv_losses = []
            
            # Perform k-fold cross validation for each parameter set
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
                # Split data for this fold
                fold_train = train_data[train_idx]
                fold_val = train_data[val_idx]
                
                # Update model configuration
                self.config.update(params)
                self.model = self._initialize_model()
                
                # Prepare data for this fold
                train_loader, val_loader = self.prepare_data(fold_train, fold_val)
                
                # Train model with current parameters
                for epoch in range(3):  # Quick training for each fold
                    self.train_epoch(train_loader, epoch, 3)
                
                # Evaluate performance
                val_loss, _ = self.evaluate(val_loader)
                cv_losses.append(val_loss)
            
            # Calculate mean cross-validation loss
            mean_cv_loss = np.mean(cv_losses)
            
            # Update best parameters if we found better ones
            if mean_cv_loss < best_val_loss:
                best_val_loss = mean_cv_loss
                best_params = params
                print(f"New best parameters found: {best_params}")
                print(f"Cross-validation loss: {best_val_loss:.4f}")
        
        # Final evaluation on held-out validation set
        self.config.update(best_params)
        self.model = self._initialize_model()
        _, final_val_loader = self.prepare_data(train_data, val_data)
        final_val_loss, _ = self.evaluate(final_val_loader)
        
        print(f"\nFinal Results:")
        print(f"Best parameters: {best_params}")
        print(f"Cross-validation loss: {best_val_loss:.4f}")
        print(f"Final validation loss: {final_val_loss:.4f}")
        
        return best_params, best_val_loss

    def augment_data(self, x, y):
        """Apply various data augmentation techniques"""
        augmented_x, augmented_y = [], []
        
        # Time warping
        def time_warp(seq, sigma=0.2):
            t = np.arange(len(seq))
            t_new = t + np.random.normal(0, sigma, size=len(t))
            return np.interp(t, t_new, seq)
        
        # Magnitude warping
        def magnitude_warp(seq, sigma=0.2):
            return seq * (1 + np.random.normal(0, sigma))
        
        # Apply augmentations
        for xi, yi in zip(x, y):
            # Original data
            augmented_x.append(xi)
            augmented_y.append(yi)
            
            # Time warped version
            augmented_x.append(time_warp(xi))
            augmented_y.append(yi)
            
            # Magnitude warped version
            augmented_x.append(magnitude_warp(xi))
            augmented_y.append(yi)
        
        return np.array(augmented_x), np.array(augmented_y)

    def interpret_predictions(self, x, y_true):
        """Generate interpretation of model predictions"""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
            output, attention = self.model(x_tensor, return_attention=True)
            prediction = output.cpu().numpy()[0]
        
        # Calculate feature attributions
        attributions = self.calculate_feature_attributions(x, prediction)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predictions vs actual
        ax1.plot(y_true, label='Actual', color='blue')
        ax1.plot(prediction, label='Predicted', color='red')
        ax1.set_title('Predictions vs Actual')
        ax1.legend()
        
        # Plot feature attributions
        im = ax2.imshow(attributions.T, aspect='auto', cmap='RdBu')
        ax2.set_title('Feature Attributions')
        plt.colorbar(im)
        
        plt.tight_layout()
        plt.show()

    def augment_batch(self, x, y):
        """Augment a single batch of data using ensemble methods"""
        return self.ensemble_data.generate_ensemble_data(
            x.cpu().numpy(), 
            y.cpu().numpy(),
            num_augmentations=self.config.get('num_augmentations', 3)
        )

    def forward(self, x, training_progress=0.0):
        """Enhanced forward pass with power management"""
        # Monitor power usage
        current_power = self.power_distributor.monitor_power_usage()
        
        # Use power-managed forward pass
        with self.memory_pool:
            output = self.power_managed_forward(x, training_progress)
            
        # Adjust power distribution based on usage
        self.power_distributor.adjust_power_distribution(current_power)
        
        return output

class ParallelSSMEnsemble:
    def __init__(self, base_config: Dict, num_models: int = 3):
        self.num_models = num_models
        self.models: List[AdvancedSSMHiPPO] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
        
        # Fix: Initialize multiprocessing with proper start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Create variations of the base config for ensemble diversity
        self.configs = []
        for i in range(num_models):
            config = base_config.copy()
            # Vary model architectures slightly
            config['d_model'] = int(config['d_model'] * (0.8 + 0.4 * np.random.random()))
            config['n_layer'] = max(2, int(config['n_layer'] + np.random.randint(-1, 2)))
            config['dropout_rate'] = config['dropout_rate'] * (0.8 + 0.4 * np.random.random())
            self.configs.append(config)
        
        # Use ThreadPoolExecutor for better thread management
        self.executor = ThreadPoolExecutor(max_workers=num_models)
        
        # Add multiprocessing pool for true parallel processing
        self.pool = mp.Pool(processes=min(num_models, mp.cpu_count()))
    
    def train_model_process(self, config: Dict, data, args):
        """Train a single model in a separate process"""
        # Fix: Add proper error handling and GPU memory management
        try:
            torch.set_num_threads(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model = AdvancedSSMHiPPO(config)
            train_loader, val_loader = model.prepare_data(data, args)
            
            for epoch in range(3):
                train_loss = model.train_epoch(train_loader, epoch, total_epochs=3)
                val_loss, metrics = model.evaluate(val_loader)
            
            return model
        except Exception as e:
            print(f"Error in training process: {str(e)}")
            return None
    
    def train_parallel(self, data, args):
        """Optimized parallel training using multiprocessing"""
        # Use Pool.starmap for parallel processing
        models = self.pool.starmap(
            self.train_model_process,
            [(config, data, args) for config in self.configs]
        )
        self.models.extend(models)
    
    def __del__(self):
        self.pool.close()
        self.pool.join()
    
    def predict(self, x):
        """Generate ensemble predictions"""
        predictions = []
        
        # Get predictions from each model
        with torch.no_grad():
            for model in self.models:
                pred = model.model(
                    torch.FloatTensor(x).unsqueeze(0).to(self.device),
                    training_progress=1.0
                )[0]
                predictions.append(pred.cpu().numpy())
        
        # Combine predictions (mean and std)
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

def advanced_example():
    """Optimized example usage with power management"""
    # Add power management configuration
    config.update({
        'power_budget': 150.0,  # Watts
        'total_memory': None,   # Auto-detect
        'num_threads': None,    # Auto-detect
        'optimize_hardware': True,
        'device_type': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    # Configuration with TVM options
    config = {
        'd_model': 128,
        'n_layer': 4,
        'seq_len': 96,
        'num_channels': 1,
        'patch_len': 16,
        'stride': 8,
        'forecast_len': 24,
        'd_state': 16,
        'expand': 2,
        'dt_rank': 'auto',
        'sigma': 0.5,
        'reduction_ratio': 4,
        'dropout_rate': 0.1,
        'l1_factor': 1e-5,
        'l2_factor': 1e-4,
        'loss_alpha': 0.3,
        'loss_beta': 0.2,
        'loss_gamma': 0.15,
        'learning_rate': 0.001,
        'batch_size': 32,
        'use_tvm': True,  # Enable TVM optimization
        'tvm_opt_level': 3,  # TVM optimization level
        'tvm_target': None,  # Auto-detect optimal target
        'distributed': True,
        'local_rank': 0,  # Set this based on your distributed setup
        'use_ensemble_data': True,
        'num_augmentations': 3,
        'noise_std': 0.01,
        'time_warp_factor': 0.2,
        'crop_ratio': 0.9,
        'jitter_std': 0.03,
        'max_segments': 5,
        'num_workers': 4
    }
    
    # Initialize distributed training
    if config['distributed']:
        torch.distributed.init_process_group(backend='nccl')
    
    # Initialize model with TVM optimization
    model = AdvancedSSMHiPPO(config)
    
    # Load and prepare data using functions from prepare_data.py
    raw_data = load_and_preprocess_data(data_path='path/to/data')
    normalized_data = normalize_data(raw_data)
    sequences = create_sequences(
        normalized_data,
        seq_length=config['seq_len'],
        forecast_length=config['forecast_len']
    )
    train_data, val_data = split_data(sequences, split_ratio=0.8)
    train_loader, val_loader = create_data_loaders(
        train_data,
        val_data,
        batch_size=config['batch_size']
    )

    # Initialize training components from train.py
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    lr_scheduler = LRScheduler(model.optimizer, mode='min', patience=3)
    logger = TrainingLogger()

    # Training loop with components from train.py
    for epoch in range(config['epochs']):
        train_metrics = train_model(
            model=model,
            train_loader=train_loader,
            criterion=model.criterion,
            optimizer=model.optimizer,
            device=model.device,
            scaler=model.scaler,
            epoch=epoch
        )
        
        val_metrics = validate_model(
            model=model,
            val_loader=val_loader,
            criterion=model.criterion,
            device=model.device
        )
        
        # Update learning rate and check early stopping
        lr_scheduler.step(val_metrics['loss'])
        if early_stopping.step(val_metrics['loss']):
            print("Early stopping triggered")
            break
            
        # Log metrics
        logger.log_metrics(epoch, train_metrics, val_metrics)
        
        # Save checkpoint if best model
        if early_stopping.is_best():
            save_checkpoint(model, model.optimizer, epoch, 'best_model.pt')

    # Plot training history
    logger.plot_metrics()

def ensemble_example():
    """Example usage of SSM-HIPPO ensemble with parallel training"""
    try:
        # Use the same config from advanced_example
        config = {
            'd_model': 128,
            'n_layer': 4,
            'seq_len': 96,
            'num_channels': 1,
            'patch_len': 16,
            'stride': 8,
            'forecast_len': 24,
            'd_state': 16,
            'expand': 2,
            'dt_rank': 'auto',
            'sigma': 0.5,
            'reduction_ratio': 4,
            'dropout_rate': 0.1,
            'l1_factor': 1e-5,
            'l2_factor': 1e-4,
            'loss_alpha': 0.3,
            'loss_beta': 0.2,
            'loss_gamma': 0.15,
            'learning_rate': 0.001,
            'batch_size': 32,
            'use_tvm': True,
            'tvm_opt_level': 3,
            'tvm_target': None,
            'distributed': True,
            'local_rank': 0,  # Set this based on your distributed setup
            'use_ensemble_data': True,
            'num_augmentations': 3,
            'noise_std': 0.01,
            'time_warp_factor': 0.2,
            'crop_ratio': 0.9,
            'jitter_std': 0.03,
            'max_segments': 5,
            'num_workers': 4
        }
        
        # Initialize distributed training
        if config['distributed']:
            torch.distributed.init_process_group(backend='nccl')
        
        # Create and train ensemble
        ensemble = ParallelSSMEnsemble(config, num_models=3)
        
        # Create sample data
        data = np.random.randn(1000, config['num_channels'])
        args = type('Args', (), {
            'input_length': config['seq_len'],
            'forecast_length': config['forecast_len']
        })()
        
        # Train ensemble
        print("Training ensemble models in parallel...")
        ensemble.train_parallel(data, args)
        
        # Generate ensemble forecast
        print("\nGenerating ensemble forecast...")
        mean_forecast, std_forecast = ensemble.predict(data[-config['seq_len']:])
        
        # Plot results with confidence intervals
        plt.figure(figsize=(12, 6))
        plt.plot(data[-config['seq_len']:], label='Input', color='blue')
        plt.plot(range(len(data)-config['forecast_len'], len(data)), 
                mean_forecast[-config['forecast_len']:], 
                label='Ensemble Forecast', 
                color='red')
        plt.fill_between(
            range(len(data)-config['forecast_len'], len(data)),
            mean_forecast[-config['forecast_len']:] - 2*std_forecast[-config['forecast_len']:],
            mean_forecast[-config['forecast_len']:] + 2*std_forecast[-config['forecast_len']:],
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        plt.legend()
        plt.title('SSM-HIPPO Ensemble Forecast')
        plt.show()
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Cleanup multiprocessing resources
        ensemble.pool.close()
        ensemble.pool.join()

def save_checkpoint(model, optimizer, epoch, filename):
    """Save model checkpoint with additional training info"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    config = checkpoint['config']
    model = AdvancedSSMHiPPO(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

if __name__ == "__main__":
    # You can choose which example to run
    use_ensemble = True
    if use_ensemble:
        ensemble_example()
    else:
        advanced_example() 
