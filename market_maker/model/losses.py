import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveTemporalCoherenceLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, training_progress=None):
        # Dynamic loss weighting based on training progress
        if training_progress is not None:
            alpha = self.alpha * (1 - training_progress)
            beta = self.beta * training_progress
        else:
            alpha, beta = self.alpha, self.beta
            
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # Temporal coherence with adaptive window size
        window_size = max(2, int(pred.size(1) * (1 - training_progress)))
        temp_coherence = F.mse_loss(
            pred[:, 1:window_size] - pred[:, :window_size-1],
            target[:, 1:window_size] - target[:, :window_size-1]
        )
        
        # Frequency domain loss with adaptive scaling
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        freq_loss = F.mse_loss(
            pred_fft.abs() * training_progress,
            target_fft.abs() * training_progress
        )
        
        return alpha * mse_loss + beta * temp_coherence + (1 - alpha - beta) * freq_loss
