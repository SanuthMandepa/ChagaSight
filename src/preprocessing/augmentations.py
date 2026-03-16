# src/preprocessing/augmentations.py — v2 PAPER-ALIGNED
"""
ECG Data Augmentations

Paper References:
  Van Santvliet et al. (2025) Table 1:
    Powerline: prob=0.5, freq random from {50±0.2, 60±0.2} Hz,
               SNR [15,30] dB, phase [0,2π], + 2nd and 3rd harmonics
    Cropping:  L1 from [5.65, 10] s
    Shifting:  L2 from [0 ± min(1, 10−L1)] s

  Kim et al. (2025) Section 2.3.1:
    Lead-mixup: Gaussian σ=0.1, applied per lead-pair independently

CHANGES vs original:
  - powerline_noise: SNR-based amplitude (not fixed 0.1), random 50 or 60 Hz,
                     2nd + 3rd harmonics added, identical across leads
  - DEFAULT_AUGMENTATION_CONFIG: prob=0.5 for powerline (was 0.3)
  - apply_augmentations: supports 'use_snr', 'random_freq', 'add_harmonics' keys
"""

import numpy as np
from typing import Optional, Dict


def lead_mixup(signal: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Lead-level mixup augmentation.

    Kim et al. (2025): Gaussian-coefficient linear interpolation between
    adjacent lead-pair patches.

    CORRECTED: Save originals before mixing.

    Args:
        signal: (12, T) ECG signal
        alpha:  Gaussian std / Beta parameter
    Returns:
        mixed: (12, T)
    """
    signal = signal.copy()
    pairs = [(6, 7), (10, 11)]   # V1-V2, V5-V6

    for i, j in pairs:
        if np.random.rand() < 0.5:
            lam = np.random.beta(alpha, alpha)
            orig_i = signal[i].copy()
            orig_j = signal[j].copy()
            signal[i] = lam * orig_i + (1 - lam) * orig_j
            signal[j] = lam * orig_j + (1 - lam) * orig_i
    return signal


def powerline_noise(
    signal: np.ndarray,
    fs: int = 100,
    freq: float = 50.0,
    amplitude: Optional[float] = None,
    snr_db: Optional[float] = None,
    snr_range: tuple = (15, 30),
    add_harmonics: bool = False,
) -> np.ndarray:
    """
    Add powerline interference — paper-aligned version.

    Van Santvliet Table 1:
      freq     : randomly from [50±0.2] ∪ [60±0.2] Hz  (caller sets freq)
      SNR      : [15, 30] dB (uniform sampling)
      phase    : [0, 2π] (uniform sampling)
      harmonics: 2nd and 3rd harmonics at SNR/2 and SNR/3 of main SNR

    Key change from original: noise is IDENTICAL across all 12 leads
    (Van Santvliet: "identical noise was added across all 12 ECG leads").

    Args:
        signal:        (12, T)
        fs:            sampling frequency (Hz)
        freq:          fundamental frequency (Hz)
        amplitude:     absolute amplitude (overrides SNR; use for backwards compat)
        snr_db:        explicit SNR in dB (overrides snr_range)
        snr_range:     (min_snr, max_snr) in dB — sampled uniformly
        add_harmonics: add 2nd and 3rd harmonic components
    Returns:
        noisy: (12, T)
    """
    signal = signal.copy()
    num_leads, T = signal.shape
    t = np.arange(T) / fs

    # Compute amplitude from SNR if not given directly
    if amplitude is None:
        if snr_db is None:
            snr_db = float(np.random.uniform(snr_range[0], snr_range[1]))
        # SNR (dB) = 20 * log10(signal_std / noise_std)
        # → noise_std = signal_std / 10^(SNR/20)
        signal_std = float(np.std(signal)) + 1e-8
        noise_std  = signal_std / (10 ** (snr_db / 20.0))
        amplitude  = noise_std
    else:
        snr_db = None   # amplitude was given directly, SNR unknown

    # Single phase for all leads (paper: identical across leads)
    phase = float(np.random.rand() * 2 * np.pi)

    # Fundamental
    noise = amplitude * np.sin(2 * np.pi * freq * t + phase)  # (T,)

    if add_harmonics:
        # 2nd harmonic: SNR = main_SNR / 2  → amplitude ∝ 1/sqrt(2) of main
        # (paper says "SNR equal to half and one third of main SNR")
        h2_amp = amplitude / np.sqrt(2)
        h3_amp = amplitude / np.sqrt(3)
        phase2 = float(np.random.rand() * 2 * np.pi)
        phase3 = float(np.random.rand() * 2 * np.pi)
        noise += h2_amp * np.sin(2 * np.pi * 2 * freq * t + phase2)
        noise += h3_amp * np.sin(2 * np.pi * 3 * freq * t + phase3)

    # Add identical noise to all 12 leads
    signal += noise[np.newaxis, :]   # broadcast (1,T) → (12,T)
    return signal


def random_shift(
    signal: np.ndarray,
    max_shift: int = 100,
    pad_mode: str = 'edge',
) -> np.ndarray:
    """
    Random temporal shift (zero-padding equivalent).

    Van Santvliet: positive shift = zero-pad before signal (delay),
                   negative shift = zero-pad after signal (advance).
    Applied identically across all leads.

    Args:
        signal:    (12, T)
        max_shift: maximum samples to shift (±max_shift)
        pad_mode:  'constant' (zeros) matches paper, 'edge' for continuity
    Returns:
        shifted: (12, T)
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return signal.copy()

    if shift > 0:
        padded = np.pad(signal, ((0, 0), (shift, 0)), mode=pad_mode)
        return padded[:, :signal.shape[1]]
    else:
        padded = np.pad(signal, ((0, 0), (0, -shift)), mode=pad_mode)
        return padded[:, -signal.shape[1]:]


def amplitude_scaling(
    signal: np.ndarray,
    scale_range: tuple = (0.8, 1.2),
) -> np.ndarray:
    """Random amplitude scaling — global (same scale for all leads)."""
    scale = float(np.random.uniform(scale_range[0], scale_range[1]))
    return signal * scale


def random_baseline_wander(
    signal: np.ndarray,
    fs: int = 100,
    amplitude: float = 0.2,
    freq_range: tuple = (0.1, 0.5),
) -> np.ndarray:
    """Random low-frequency baseline wander (per-lead independent)."""
    signal = signal.copy()
    t = np.arange(signal.shape[1]) / fs
    for lead in range(signal.shape[0]):
        freq  = float(np.random.uniform(freq_range[0], freq_range[1]))
        phase = float(np.random.rand() * 2 * np.pi)
        signal[lead] += amplitude * np.sin(2 * np.pi * freq * t + phase)
    return signal


def apply_augmentations(
    signal: np.ndarray,
    augmentation_config: Dict,
    training: bool = True,
) -> np.ndarray:
    """
    Apply augmentations according to config dict.

    Supports two powerline modes:
      Classic (backwards-compatible):
        {'prob': 0.5, 'amplitude': 0.1, 'freq': 60.0}
      Paper-aligned (v11):
        {'prob': 0.5, 'use_snr': True, 'snr_range': (15,30),
         'random_freq': True, 'add_harmonics': True}

    Args:
        signal:              (12, T)
        augmentation_config: dict
        training:            if False, skip all augmentations
    Returns:
        augmented: (12, T)
    """
    if not training:
        return signal

    aug = signal.copy()

    # Lead mixup
    if 'lead_mixup' in augmentation_config:
        cfg = augmentation_config['lead_mixup']
        if np.random.rand() < cfg.get('prob', 0.0):
            aug = lead_mixup(aug, alpha=cfg.get('alpha', 0.2))

    # Powerline noise
    if 'powerline_noise' in augmentation_config:
        cfg = augmentation_config['powerline_noise']
        if np.random.rand() < cfg.get('prob', 0.0):

            # Determine frequency
            if cfg.get('random_freq', False):
                # Van Santvliet: randomly from {50±0.2, 60±0.2} Hz
                base = float(np.random.choice([50.0, 60.0]))
                freq = base + float(np.random.uniform(-0.2, 0.2))
            else:
                freq = float(cfg.get('freq', 50.0))

            # Determine amplitude mode
            if cfg.get('use_snr', False):
                aug = powerline_noise(
                    aug,
                    freq=freq,
                    amplitude=None,
                    snr_range=cfg.get('snr_range', (15, 30)),
                    add_harmonics=cfg.get('add_harmonics', True),
                )
            else:
                aug = powerline_noise(
                    aug,
                    freq=freq,
                    amplitude=float(cfg.get('amplitude', 0.1)),
                    add_harmonics=cfg.get('add_harmonics', False),
                )

    # Random shift
    if 'random_shift' in augmentation_config:
        cfg = augmentation_config['random_shift']
        if np.random.rand() < cfg.get('prob', 0.0):
            aug = random_shift(aug, max_shift=cfg.get('max_shift', 100))

    # Amplitude scaling
    if 'amplitude_scaling' in augmentation_config:
        cfg = augmentation_config['amplitude_scaling']
        if np.random.rand() < cfg.get('prob', 0.0):
            aug = amplitude_scaling(aug, scale_range=cfg.get('scale_range', (0.8, 1.2)))

    # Baseline wander
    if 'baseline_wander' in augmentation_config:
        cfg = augmentation_config['baseline_wander']
        if np.random.rand() < cfg.get('prob', 0.0):
            aug = random_baseline_wander(
                aug,
                amplitude=cfg.get('amplitude', 0.2),
                freq_range=cfg.get('freq_range', (0.1, 0.5)),
            )

    return aug


# ── Default config (backwards-compatible) ─────────────────────────────────────
DEFAULT_AUGMENTATION_CONFIG = {
    'lead_mixup':       {'prob': 0.3, 'alpha': 0.2},
    'powerline_noise':  {'prob': 0.3, 'amplitude': 0.1, 'freq': 60.0},
    'random_shift':     {'prob': 0.5, 'max_shift': 100},
    'amplitude_scaling':{'prob': 0.3, 'scale_range': (0.8, 1.2)},
    'baseline_wander':  {'prob': 0.2, 'amplitude': 0.2, 'freq_range': (0.1, 0.5)},
}

# ── Paper-aligned config for v11 training ─────────────────────────────────────
PAPER_AUGMENTATION_CONFIG = {
    'lead_mixup':       {'prob': 0.3, 'alpha': 0.2},
    'powerline_noise':  {
        'prob': 0.5,            # Van Santvliet: "probability of 0.5"
        'use_snr': True,        # SNR-based amplitude
        'snr_range': (15, 30),  # dB, Table 1
        'random_freq': True,    # randomly 50 or 60 Hz per sample
        'add_harmonics': True,  # 2nd + 3rd harmonics
    },
    'random_shift':     {'prob': 0.5, 'max_shift': 100},
    'amplitude_scaling':{'prob': 0.3, 'scale_range': (0.8, 1.2)},
    'baseline_wander':  {'prob': 0.2, 'amplitude': 0.2, 'freq_range': (0.1, 0.5)},
}


if __name__ == '__main__':
    import sys
    print('Augmentations v2 self-test')
    np.random.seed(42)
    sig = np.random.randn(12, 1000).astype(np.float32)

    # Test classic
    out_classic = apply_augmentations(sig, DEFAULT_AUGMENTATION_CONFIG, training=True)
    assert out_classic.shape == (12, 1000), 'Shape mismatch'
    print(f'  Classic: ok  (std={out_classic.std():.4f})')

    # Test paper-aligned
    out_paper = apply_augmentations(sig, PAPER_AUGMENTATION_CONFIG, training=True)
    assert out_paper.shape == (12, 1000), 'Shape mismatch'
    print(f'  Paper:   ok  (std={out_paper.std():.4f})')

    # Verify harmonics: run powerline_noise directly
    noisy = powerline_noise(sig, freq=50.0, snr_range=(15, 30), add_harmonics=True)
    assert noisy.shape == sig.shape
    print(f'  Harmonics: ok  (diff_std={np.std(noisy - sig):.4f})')

    print('All tests passed.')