import torch
import torch.nn as nn
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

if __name__ == "__main__":
    advanced_example() 