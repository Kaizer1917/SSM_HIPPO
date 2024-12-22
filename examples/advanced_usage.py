import torch
import torch.nn as nn
import threading
from queue import Queue
import torch.multiprocessing as mp
from typing import List, Dict
from model.mamba import SSM_HIPPO, ModelArgs
from model.losses import EnhancedSSMHiPPOLoss
from model.regularization import AdaptiveRegularization
from model.prepare_data import enhanced_preprocessing, create_optimized_dataloaders
import numpy as np
import matplotlib.pyplot as plt
from model.mamba_tvm import SSMHippoModule
from model.mamba_tvm_block import MambaBlockTVM
from model.mamba_tvm_utils import get_optimal_target, optimize_for_hardware
from model.mamba_tvm_memory import optimize_memory_layout
import tvm
from tvm import relax

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

    def prepare_data(self, data, args):
        """Prepare data with enhanced preprocessing"""
        sequences, targets = enhanced_preprocessing(data, args)
        train_loader, val_loader = create_optimized_dataloaders(
            args, sequences, targets, 
            batch_size=self.config['batch_size']
        )
        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch, total_epochs):
        """Train for one epoch with advanced features and TVM optimization"""
        self.model.train()
        total_loss = 0
        
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Calculate training progress
            progress = (epoch + i/len(train_loader)) / total_epochs
            
            if self.use_tvm:
                # Optimize memory layout for better performance
                X_batch = optimize_memory_layout(X_batch, layout="NCHW")
            
            # Forward pass with regularization
            self.optimizer.zero_grad()
            output, reg_loss = self.model(X_batch, progress)
            
            # Calculate loss with enhanced components
            loss = self.criterion(output, y_batch, training_progress=progress)
            total_loss = loss + reg_loss
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
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

class ParallelSSMEnsemble:
    def __init__(self, base_config: Dict, num_models: int = 3):
        self.num_models = num_models
        self.models: List[AdvancedSSMHiPPO] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create variations of the base config for ensemble diversity
        self.configs = []
        for i in range(num_models):
            config = base_config.copy()
            # Vary model architectures slightly
            config['d_model'] = int(config['d_model'] * (0.8 + 0.4 * np.random.random()))
            config['n_layer'] = max(2, int(config['n_layer'] + np.random.randint(-1, 2)))
            config['dropout_rate'] = config['dropout_rate'] * (0.8 + 0.4 * np.random.random())
            self.configs.append(config)
    
    def train_model_thread(self, config: Dict, data, args, queue: Queue):
        """Train a single model in a separate thread"""
        try:
            model = AdvancedSSMHiPPO(config)
            train_loader, val_loader = model.prepare_data(data, args)
            
            # Training loop
            for epoch in range(3):  # Using same epochs as advanced_example
                train_loss = model.train_epoch(train_loader, epoch, total_epochs=3)
                val_loss, metrics = model.evaluate(val_loader)
                
                print(f"Model {len(self.models)+1}, Epoch {epoch+1}:")
                print(f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                print(f"Metrics: MSE = {metrics['mse']:.4f}, MAE = {metrics['mae']:.4f}")
            
            queue.put(model)
        except Exception as e:
            queue.put(e)
    
    def train_parallel(self, data, args):
        """Train multiple models in parallel"""
        queue = Queue()
        threads = []
        
        # Start training threads
        for config in self.configs:
            thread = threading.Thread(
                target=self.train_model_thread,
                args=(config, data, args, queue)
            )
            thread.start()
            threads.append(thread)
        
        # Collect trained models
        for thread in threads:
            result = queue.get()
            if isinstance(result, Exception):
                raise result
            self.models.append(result)
            thread.join()
    
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
    """Example usage of advanced SSM-HIPPO implementation with TVM optimization"""
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
    }
    
    # Initialize model with TVM optimization
    model = AdvancedSSMHiPPO(config)
    
    # Create sample data
    data = np.random.randn(1000, config['num_channels'])
    args = type('Args', (), {
        'input_length': config['seq_len'],
        'forecast_length': config['forecast_len']
    })()
    
    # Prepare data
    train_loader, val_loader = model.prepare_data(data, args)
    
    # Training loop with performance monitoring
    import time
    for epoch in range(3):
        start_time = time.time()
        train_loss = model.train_epoch(train_loader, epoch, total_epochs=3)
        val_loss, metrics = model.evaluate(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"Metrics: MSE = {metrics['mse']:.4f}, MAE = {metrics['mae']:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
    
    # Plot example forecast
    x = data[-config['seq_len']:]
    y_true = data[-config['seq_len']-config['forecast_len']:]
    model.plot_forecast(x, y_true, config['forecast_len'])

def ensemble_example():
    """Example usage of SSM-HIPPO ensemble with parallel training"""
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
    }
    
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

if __name__ == "__main__":
    # You can choose which example to run
    use_ensemble = True
    if use_ensemble:
        ensemble_example()
    else:
        advanced_example() 
