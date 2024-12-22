import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedSSMHiPPOLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.15):
        super().__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta   # Temporal coherence weight
        self.gamma = gamma # Frequency domain weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, training_progress=None):
        batch_size = pred.size(0)
        
        # 1. Dynamic weighting based on training progress
        if training_progress is not None:
            # Gradually increase focus on temporal and frequency components
            alpha = self.alpha * (1 - 0.5 * training_progress)
            beta = self.beta * (1 + training_progress)
            gamma = self.gamma * (1 + 2 * training_progress)
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
            
        # 2. Enhanced MSE with outlier robustness
        mse_loss = self.mse(pred, target)
        huber_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        base_loss = 0.7 * mse_loss + 0.3 * huber_loss
        
        # 3. Multi-scale temporal coherence
        temp_losses = []
        scales = [2, 4, 8]  # Multiple temporal scales
        for scale in scales:
            window_size = max(2, int(pred.size(1) * scale / 10))
            temp_diff_pred = pred[:, scale:] - pred[:, :-scale]
            temp_diff_target = target[:, scale:] - target[:, :-scale]
            temp_loss = F.mse_loss(temp_diff_pred, temp_diff_target)
            temp_losses.append(temp_loss)
        
        # Add gradient-based regularization
        if training_progress is not None:
            grad_scale = torch.exp(-5 * training_progress)  # Decrease over time
            pred.register_hook(lambda grad: grad * grad_scale)
        
        # Use weighted temporal loss
        temp_weights = torch.exp(-torch.arange(len(scales), device=pred.device) / 2)
        temporal_loss = sum(w * l for w, l in zip(temp_weights, temp_losses)) / temp_weights.sum()
        
        # 4. Enhanced frequency domain analysis
        # Apply FFT with Hann window for better frequency resolution
        hann_window = torch.hann_window(pred.size(1), device=pred.device)
        pred_fft = torch.fft.rfft(pred * hann_window, dim=1)
        target_fft = torch.fft.rfft(target * hann_window, dim=1)
        
        # Separate amplitude and phase losses
        amp_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
        phase_loss = F.mse_loss(torch.angle(pred_fft), torch.angle(target_fft))
        freq_loss = 0.7 * amp_loss + 0.3 * phase_loss
        
        # Adaptive frequency loss weight
        freq_weight = 0.3 * (1 + torch.sin(math.pi * training_progress))
        freq_loss = freq_loss * freq_weight
        
        # 5. State space consistency loss
        if pred.size(1) > 1:
            state_consistency = F.mse_loss(
                pred[:, 1:] - pred[:, :-1],
                target[:, 1:] - target[:, :-1]
            )
        else:
            state_consistency = torch.tensor(0.0, device=pred.device)
        
        # 6. Combine losses with dynamic weighting
        total_loss = (
            alpha * base_loss + 
            beta * temporal_loss + 
            gamma * freq_loss + 
            0.1 * state_consistency
        )
        
        # 7. Add regularization term for numerical stability
        stability_term = 0.01 * torch.mean(torch.abs(pred))
        total_loss = total_loss + stability_term * (1 - training_progress)
        
        return total_loss

    def get_loss_components(self, pred, target, training_progress=None):
        """Return individual loss components for monitoring"""
        with torch.no_grad():
            base_loss = self.mse(pred, target)
            temp_loss = F.mse_loss(
                pred[:, 1:] - pred[:, :-1],
                target[:, 1:] - target[:, :-1]
            )
            pred_fft = torch.fft.rfft(pred, dim=1)
            target_fft = torch.fft.rfft(target, dim=1)
            freq_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
            
        return {
            'base_loss': base_loss.item(),
            'temporal_loss': temp_loss.item(),
            'frequency_loss': freq_loss.item()
        }
