# src/preprocessing/augmentations.py - CORRECTED VERSION
"""
ECG Data Augmentations

Paper References:
- Kim et al. (2025): Lead-mixup, random cropping
- Van Santvliet et al. (2025): Powerline noise, random shifts

CORRECTED:
- Lead mixup bug fixed (save original before mixing)
- Powerline noise broadcasting fixed
- Random shift uses padding instead of roll
"""

import numpy as np
from typing import Optional


def lead_mixup(signal: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Lead-level mixup augmentation.
    
    Paper: Van Santvliet et al. (2025) Section 2.8
    
    CORRECTED: Save original values before mixing to avoid using modified values.
    
    Args:
        signal: (12, T) ECG signal
        alpha: Beta distribution parameter
    
    Returns:
        mixed: (12, T) augmented signal
    """
    signal = signal.copy()
    pairs = [(6, 7), (10, 11)]  # V1-V2, V5-V6 indices
    
    for i, j in pairs:
        if np.random.rand() < 0.5:
            lam = np.random.beta(alpha, alpha)
            
            # CORRECTED: Save originals first!
            orig_i = signal[i].copy()
            orig_j = signal[j].copy()
            
            # Mix using original values
            signal[i] = lam * orig_i + (1 - lam) * orig_j
            signal[j] = lam * orig_j + (1 - lam) * orig_i
    
    return signal


def powerline_noise(
    signal: np.ndarray, 
    fs: int = 100, 
    freq: float = 50.0,
    amplitude: float = 0.1
) -> np.ndarray:
    """
    Add 50/60Hz powerline interference.
    
    Paper: Van Santvliet et al. (2025) Section 2.8
    
    CORRECTED: Fix broadcasting to work with (12, T) signals.
    
    Args:
        signal: (12, T) ECG signal
        fs: sampling frequency
        freq: powerline frequency (50 or 60 Hz)
        amplitude: noise amplitude relative to signal std
    
    Returns:
        noisy: (12, T) signal with powerline noise
    """
    signal = signal.copy()
    num_leads, T = signal.shape
    
    # Generate time vector
    t = np.arange(T) / fs
    
    # Generate powerline noise (same for all leads)
    noise_1d = amplitude * np.sin(2 * np.pi * freq * t)  # (T,)
    
    # Add random phase shift per lead for realism
    noise = np.zeros((num_leads, T))
    for lead_idx in range(num_leads):
        phase = np.random.rand() * 2 * np.pi
        noise[lead_idx] = amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    signal += noise
    return signal


def random_shift(
    signal: np.ndarray, 
    max_shift: int = 100,
    pad_mode: str = 'edge'
) -> np.ndarray:
    """
    Random temporal shift augmentation.
    
    Paper: Kim et al. (2025) Section 2.3
    
    CORRECTED: Use padding instead of roll to avoid wraparound.
    
    Args:
        signal: (12, T) ECG signal
        max_shift: maximum shift in samples
        pad_mode: padding mode ('edge', 'constant', 'reflect')
    
    Returns:
        shifted: (12, T) shifted signal
    """
    signal = signal.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    
    if shift == 0:
        return signal
    
    elif shift > 0:
        # Shift right (delay) - pad on left
        padded = np.pad(signal, ((0, 0), (shift, 0)), mode=pad_mode)
        return padded[:, :signal.shape[1]]
    
    else:
        # Shift left (advance) - pad on right
        padded = np.pad(signal, ((0, 0), (0, -shift)), mode=pad_mode)
        return padded[:, -signal.shape[1]:]


def amplitude_scaling(
    signal: np.ndarray,
    scale_range: tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Random amplitude scaling.
    
    Args:
        signal: (12, T) ECG signal
        scale_range: (min_scale, max_scale)
    
    Returns:
        scaled: (12, T) scaled signal
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def random_baseline_wander(
    signal: np.ndarray,
    fs: int = 100,
    amplitude: float = 0.2,
    freq_range: tuple[float, float] = (0.1, 0.5)
) -> np.ndarray:
    """
    Add random baseline wander (low-frequency drift).
    
    Args:
        signal: (12, T) ECG signal
        fs: sampling frequency
        amplitude: wander amplitude
        freq_range: frequency range for wander
    
    Returns:
        wandered: (12, T) signal with baseline wander
    """
    signal = signal.copy()
    num_leads, T = signal.shape
    
    t = np.arange(T) / fs
    
    # Add different wander to each lead
    for lead_idx in range(num_leads):
        freq = np.random.uniform(freq_range[0], freq_range[1])
        phase = np.random.rand() * 2 * np.pi
        wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
        signal[lead_idx] += wander
    
    return signal


def apply_augmentations(
    signal: np.ndarray,
    augmentation_config: dict,
    training: bool = True
) -> np.ndarray:
    """
    Apply a set of augmentations with specified probabilities.
    
    Args:
        signal: (12, T) ECG signal
        augmentation_config: dict with augmentation settings
        training: if False, skip all augmentations
    
    Returns:
        augmented: (12, T) augmented signal
    
    Example config:
        {
            'lead_mixup': {'prob': 0.3, 'alpha': 0.2},
            'powerline_noise': {'prob': 0.3, 'amplitude': 0.1, 'freq': 60.0},
            'random_shift': {'prob': 0.5, 'max_shift': 100},
            'amplitude_scaling': {'prob': 0.3, 'scale_range': (0.8, 1.2)},
            'baseline_wander': {'prob': 0.2, 'amplitude': 0.2}
        }
    """
    if not training:
        return signal
    
    augmented = signal.copy()
    
    # Lead mixup
    if 'lead_mixup' in augmentation_config:
        cfg = augmentation_config['lead_mixup']
        if np.random.rand() < cfg.get('prob', 0.0):
            augmented = lead_mixup(augmented, alpha=cfg.get('alpha', 0.2))
    
    # Powerline noise
    if 'powerline_noise' in augmentation_config:
        cfg = augmentation_config['powerline_noise']
        if np.random.rand() < cfg.get('prob', 0.0):
            augmented = powerline_noise(
                augmented,
                amplitude=cfg.get('amplitude', 0.1),
                freq=cfg.get('freq', 50.0)
            )
    
    # Random shift
    if 'random_shift' in augmentation_config:
        cfg = augmentation_config['random_shift']
        if np.random.rand() < cfg.get('prob', 0.0):
            augmented = random_shift(
                augmented,
                max_shift=cfg.get('max_shift', 100)
            )
    
    # Amplitude scaling
    if 'amplitude_scaling' in augmentation_config:
        cfg = augmentation_config['amplitude_scaling']
        if np.random.rand() < cfg.get('prob', 0.0):
            augmented = amplitude_scaling(
                augmented,
                scale_range=cfg.get('scale_range', (0.8, 1.2))
            )
    
    # Baseline wander
    if 'baseline_wander' in augmentation_config:
        cfg = augmentation_config['baseline_wander']
        if np.random.rand() < cfg.get('prob', 0.0):
            augmented = random_baseline_wander(
                augmented,
                amplitude=cfg.get('amplitude', 0.2),
                freq_range=cfg.get('freq_range', (0.1, 0.5))
            )
    
    return augmented


# Default augmentation config for training
DEFAULT_AUGMENTATION_CONFIG = {
    'lead_mixup': {
        'prob': 0.3,
        'alpha': 0.2
    },
    'powerline_noise': {
        'prob': 0.3,
        'amplitude': 0.1,
        'freq': 60.0  # Use 60 Hz for US, 50 Hz for Europe
    },
    'random_shift': {
        'prob': 0.5,
        'max_shift': 100
    },
    'amplitude_scaling': {
        'prob': 0.3,
        'scale_range': (0.8, 1.2)
    },
    'baseline_wander': {
        'prob': 0.2,
        'amplitude': 0.2,
        'freq_range': (0.1, 0.5)
    }
}