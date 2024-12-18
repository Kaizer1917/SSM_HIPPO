import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCoherenceLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2):
        super().__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta   # Weight for temporal coherence
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        # Temporal coherence loss (smoothness of predictions)
        temp_coherence = F.mse_loss(
            pred[:, 1:] - pred[:, :-1],
            target[:, 1:] - target[:, :-1]
        )
        
        # Frequency domain loss using FFT
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        freq_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
        
        # Combine losses
        total_loss = (
            self.alpha * mse_loss + 
            self.beta * temp_coherence + 
            (1 - self.alpha - self.beta) * freq_loss
        )
        
        return total_loss
