import torch
import torch.nn as nn

class AdaptiveRegularization(nn.Module):
    def __init__(self, model, dropout_rate=0.1, l1_factor=1e-5, l2_factor=1e-4):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        
    def forward(self, x, training_progress):
        # Adaptive L1/L2 regularization based on training progress
        l1_reg = 0
        l2_reg = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Increase L1 regularization over time for sparsity
                l1_reg += self.l1_factor * training_progress * torch.norm(param, 1)
                # Decrease L2 regularization over time
                l2_reg += self.l2_factor * (1 - training_progress) * torch.norm(param, 2)
        
        # Apply dropout with adaptive rate
        x = self.dropout(x * (1 - 0.5 * training_progress))
        
        return x, l1_reg + l2_reg
