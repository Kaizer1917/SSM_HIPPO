import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import resample
import random
from typing import List, Tuple, Optional
from functools import lru_cache

class TimeSeriesEnsembleData:
    """Ensemble methods for time series data augmentation and processing."""
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 time_warp_factor: float = 0.2,
                 crop_ratio: float = 0.9,
                 jitter_std: float = 0.03,
                 max_segments: int = 5):
        """
        Args:
            noise_std: Standard deviation for Gaussian noise
            time_warp_factor: Factor for time warping (0-1)
            crop_ratio: Ratio of sequence to keep in random cropping
            jitter_std: Standard deviation for random jitter
            max_segments: Maximum number of segments for piece-wise scaling
        """
        self.noise_std = noise_std
        self.time_warp_factor = time_warp_factor
        self.crop_ratio = crop_ratio
        self.jitter_std = jitter_std
        self.max_segments = max_segments

    @lru_cache(maxsize=32)
    def _get_warping_matrix(self, length: int) -> np.ndarray:
        """Generate cached time warping matrix."""
        return np.random.normal(1.0, self.time_warp_factor, size=length)

    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the signal."""
        noise = np.random.normal(0, self.noise_std, size=x.shape)
        return x + noise

    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """Apply non-linear time warping."""
        length = x.shape[-1]
        warp_matrix = self._get_warping_matrix(length)
        
        # Create new time points with warping
        old_times = np.arange(length)
        new_times = np.cumsum(warp_matrix)
        new_times = (length - 1) * new_times / new_times[-1]
        
        # Interpolate signal to new time points
        cs = CubicSpline(old_times, x)
        return cs(new_times)

    def random_crop(self, x: np.ndarray) -> np.ndarray:
        """Randomly crop and resize the signal."""
        length = x.shape[-1]
        crop_length = int(length * self.crop_ratio)
        start = random.randint(0, length - crop_length)
        cropped = x[..., start:start + crop_length]
        return resample(cropped, length)

    def magnitude_warp(self, x: np.ndarray) -> np.ndarray:
        """Apply piece-wise magnitude scaling."""
        length = x.shape[-1]
        num_segments = random.randint(2, self.max_segments)
        segments = np.array_split(np.arange(length), num_segments)
        
        warped = x.copy()
        for segment in segments:
            scale = np.random.normal(1.0, self.jitter_std)
            warped[..., segment] *= scale
        return warped

    def frequency_mask(self, x: np.ndarray, num_masks: int = 2) -> np.ndarray:
        """Apply frequency domain masking."""
        x_fft = np.fft.rfft(x)
        freq_length = x_fft.shape[-1]
        
        for _ in range(num_masks):
            mask_length = random.randint(1, freq_length // 4)
            mask_start = random.randint(0, freq_length - mask_length)
            x_fft[..., mask_start:mask_start + mask_length] = 0
            
        return np.fft.irfft(x_fft, n=x.shape[-1])

    def generate_ensemble_data(self, 
                             x: np.ndarray, 
                             y: np.ndarray, 
                             num_augmentations: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble of augmented data.
        
        Args:
            x: Input sequences (batch_size, channels, seq_length)
            y: Target sequences (batch_size, channels, forecast_length)
            num_augmentations: Number of augmented versions to generate
        
        Returns:
            Tuple of augmented inputs and targets
        """
        augmented_x = [x]
        augmented_y = [y]
        
        for _ in range(num_augmentations):
            # Apply random combination of augmentations
            aug_x = x.copy()
            
            # Randomly select and apply augmentations
            if random.random() < 0.7:
                aug_x = self.add_noise(aug_x)
            if random.random() < 0.5:
                aug_x = self.time_warp(aug_x)
            if random.random() < 0.3:
                aug_x = self.random_crop(aug_x)
            if random.random() < 0.4:
                aug_x = self.magnitude_warp(aug_x)
            if random.random() < 0.3:
                aug_x = self.frequency_mask(aug_x)
            
            augmented_x.append(aug_x)
            augmented_y.append(y)
        
        return np.concatenate(augmented_x, axis=0), np.concatenate(augmented_y, axis=0)

    def create_ensemble_dataset(self, 
                              data_loader: torch.utils.data.DataLoader,
                              num_augmentations: int = 3) -> torch.utils.data.Dataset:
        """Create an ensemble dataset with augmented samples.
        
        Args:
            data_loader: Original data loader
            num_augmentations: Number of augmented versions per sample
            
        Returns:
            Enhanced dataset with augmented samples
        """
        all_x = []
        all_y = []
        
        for x_batch, y_batch in data_loader:
            x_aug, y_aug = self.generate_ensemble_data(
                x_batch.numpy(),
                y_batch.numpy(),
                num_augmentations
            )
            all_x.append(x_aug)
            all_y.append(y_aug)
        
        x_ensemble = np.concatenate(all_x, axis=0)
        y_ensemble = np.concatenate(all_y, axis=0)
        
        return torch.utils.data.TensorDataset(
            torch.FloatTensor(x_ensemble),
            torch.FloatTensor(y_ensemble)
        ) 