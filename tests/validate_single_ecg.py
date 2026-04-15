# tests/validate_single_ecg.py
"""
Visual sanity-check for ONE ECG across the full preprocessing pipeline.

Covers every step for all three datasets: PTB-XL, SaMi-Trop, CODE-15%.

FIXES vs original:
  - resample_signal() not resample_ecg()
  - normalize_per_lead() not normalize_dataset()
  - build_2d_image() not ecg_to_contour_image()
  - 'samitrop' not 'sami_trop'
  - Signal convention: (12, T) throughout
  - build_2d_image expects (12, T) float, clipped to [-3,3]
  - Saved 1D files: shape (12, 1000) not (1000, 12)
  - Image files: {id}.npy not {id}_img.npy

Usage:
  python -m tests.validate_single_ecg --dataset ptbxl --id 1
  python -m tests.validate_single_ecg --dataset samitrop --id 3629
  python -m tests.validate_single_ecg --dataset code15 --id 13
  python -m tests.validate_single_ecg --all
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wfdb
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample        import resample_signal, pad_or_trim
from src.preprocessing.normalization   import normalize_per_lead
from src.preprocessing.image_embedding import build_2d_image

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_WFDB   = Path("data/official_wfdb")
PROC_100    = Path("data/processed/1d_signals_100hz")
PROC_IMG    = Path("data/processed/2d_images")
OUT_BASE    = Path("tests/verification_outputs/pipeline")
TARGET_FS_FM     = 100
TARGET_FS_IMAGE  = 500
TARGET_LEN_100   = 1000   # 10 s @ 100 Hz
TARGET_LEN_500   = 5000   # 10 s @ 500 Hz
TARGET_WIDTH     = 2048
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASET_INFO = {
    'ptbxl':    {'native_fs': 500, 'label': 'PTB-XL (500 Hz, all negative)',    'color': '#2E86AB'},
    'samitrop': {'native_fs': 400, 'label': 'SaMi-Trop (400 Hz, all positive)', 'color': '#A23B72'},
    'code15':   {'native_fs': 400, 'label': 'CODE-15% (400 Hz, mixed)',         'color': '#F18F01'},
}

# ── WFDB folder aliases ───────────────────────────────────────────────────────
def _wfdb_folder(ds: str) -> Path:
    for name in [ds, f'{ds}s']:
        p = BASE_WFDB / name
        if p.exists():
            return p
    return BASE_WFDB / ds


def _processed_folder(ds: str, kind: str) -> Path:
    base = PROC_100 if kind == '1d' else PROC_IMG
    return base / ds


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_wfdb(ds: str, record_id: str):
    """Returns (signal (12,T) float32, fs float)."""
    folder = _wfdb_folder(ds)
    if ds == 'ptbxl':
        num = int(record_id)
        sub = f"{num // 1000:05d}"
        for suffix in ['_lr', '_hr', '']:
            for subdir in ['records100', 'records500', '']:
                if subdir:
                    p = folder / subdir / sub / f"{num:05d}{suffix}"
                else:
                    p = folder / sub / f"{num:05d}{suffix}"
                if p.with_suffix('.hea').exists():
                    sig, fields = wfdb.rdsamp(str(p))
                    return sig.T.astype(np.float32), float(fields['fs'])
        raise FileNotFoundError(f"PTB-XL WFDB not found for ID {record_id}")
    else:
        p = folder / record_id
        if not p.with_suffix('.hea').exists():
            raise FileNotFoundError(f"WFDB not found: {p}.hea")
        sig, fields = wfdb.rdsamp(str(p))
        return sig.T.astype(np.float32), float(fields['fs'])


def load_saved_1d(ds: str, record_id: str) -> np.ndarray:
    p = _processed_folder(ds, '1d') / f"{record_id}.npy"
    if not p.exists():
        # PTB-XL uses zero-padded IDs
        if ds == 'ptbxl':
            p = _processed_folder(ds, '1d') / f"{int(record_id):05d}_hr.npy"
    if not p.exists():
        raise FileNotFoundError(f"Saved 1D signal not found: {p}")
    arr = np.load(p)
    # Normalise to (12, T) if saved as (T, 12) by older scripts
    if arr.ndim == 2 and arr.shape[0] != 12 and arr.shape[1] == 12:
        arr = arr.T
    return arr


def load_saved_2d(ds: str, record_id: str) -> np.ndarray:
    p = _processed_folder(ds, '2d') / f"{record_id}.npy"
    if not p.exists():
        if ds == 'ptbxl':
            p = _processed_folder(ds, '2d') / f"{int(record_id):05d}_hr.npy"
    if not p.exists():
        raise FileNotFoundError(f"Saved 2D image not found: {p}")
    return np.load(p)


# ── Pipeline simulation ───────────────────────────────────────────────────────
def run_pipeline(raw_signal: np.ndarray, fs: float, ds: str) -> dict:
    """Reproduce build_all_data.py pipeline. Input/output shape: (12, T)."""
    assert raw_signal.ndim == 2 and raw_signal.shape[0] == 12, \
        f"Expected (12, T), got {raw_signal.shape}"

    # 1. Baseline removal
    filtered = remove_baseline(raw_signal, fs=fs)

    # 2a. 1D FM path: → 100 Hz → 1000 samples (no z-score at this stage)
    sig_100 = resample_signal(filtered, original_fs=fs, target_fs=TARGET_FS_FM)
    sig_100 = pad_or_trim(sig_100, TARGET_LEN_100)

    # 2b. 2D image path: → 500 Hz → 5000 samples → z-score → clip → build image
    sig_500 = resample_signal(filtered, original_fs=fs, target_fs=TARGET_FS_IMAGE)
    sig_500 = pad_or_trim(sig_500, TARGET_LEN_500)
    sig_500_z = normalize_per_lead(sig_500, clip_std=3.0)  # clips to [-3,3]

    # 3. Build 2D image from z-scored 500 Hz signal
    image = build_2d_image(sig_500_z, target_width=TARGET_WIDTH, random_crop=False)

    return {
        'raw':       raw_signal,
        'filtered':  filtered,
        'sig_100':   sig_100,
        'sig_500':   sig_500,
        'sig_500_z': sig_500_z,
        'image':     image,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _save(fig, path): fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def plot_pipeline_overview(steps: dict, ds: str, record_id: str, out_dir: Path):
    """Master plot: all 1D pipeline stages for Lead I."""
    info  = DATASET_INFO.get(ds, {'color': '#555555', 'label': ds})
    color = info['color']

    pipeline_steps = [
        ('01 Raw WFDB',             steps['raw'],      DATASET_INFO.get(ds,{}).get('native_fs', 400)),
        ('02 Baseline removed',     steps['filtered'], DATASET_INFO.get(ds,{}).get('native_fs', 400)),
        ('03 1D FM → 100 Hz',       steps['sig_100'],  TARGET_FS_FM),
        ('04 2D path → 500 Hz',     steps['sig_500'],  TARGET_FS_IMAGE),
        ('05 500 Hz z-scored+clip', steps['sig_500_z'],TARGET_FS_IMAGE),
    ]
    n = len(pipeline_steps)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=False)

    for ax, (label, sig, fs) in zip(axes, pipeline_steps):
        t = np.arange(sig.shape[1]) / fs
        ax.plot(t, sig[0], lw=0.7, color=color)
        ax.set_title(f'{label}  shape={sig.shape}  fs={fs} Hz', fontsize=9)
        ax.set_ylabel('mV'); ax.grid(True, alpha=0.3)
        if 'z-scored' in label:
            ax.axhline( 3, color='r', ls='--', lw=0.8, alpha=0.6, label='clip ±3')
            ax.axhline(-3, color='r', ls='--', lw=0.8, alpha=0.6)
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'{info["label"]}  |  Record: {record_id}  |  Lead I',
                 fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir / '01_pipeline_lead_I.png')


def plot_all_12_leads(sig: np.ndarray, fs: float, title: str,
                      fname: Path, color: str = '#2E86AB'):
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    t = np.arange(sig.shape[1]) / fs
    for i in range(12):
        axes[i].plot(t, sig[i], lw=0.5, color=color)
        axes[i].set_title(LEAD_NAMES[i], fontsize=9)
        axes[i].grid(True, alpha=0.2)
    for ax in axes: ax.set_xlabel('Time (s)')
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    _save(fig, fname)


def plot_2d_image_construction(steps: dict, ds: str, record_id: str, out_dir: Path):
    """The 8×T → 24×T → (3, 24, 2048) construction diagram."""
    img  = steps['image']   # (3, 24, 2048) uint8
    z500 = steps['sig_500_z']  # (12, 5000) float

    fig = plt.figure(figsize=(20, 20))
    gs  = gridspec.GridSpec(5, 1, hspace=0.5)

    # Panel 1: 12-lead signals (12 × 5000) as heatmap
    ax0 = fig.add_subplot(gs[0])
    # Downsample for display
    z_ds = z500[:, ::5]  # (12, 1000) for display
    im0  = ax0.imshow(z_ds, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
                      interpolation='nearest')
    ax0.set_yticks(range(12)); ax0.set_yticklabels(LEAD_NAMES, fontsize=8)
    ax0.set_title('Input: 12-lead z-scored signal  (12, 5000) @ 500 Hz  '
                  '→ clipped to [-3, 3]', fontsize=10)
    ax0.set_xlabel('Sample (downsampled ×5 for display)')
    plt.colorbar(im0, ax=ax0, fraction=0.015, label='z-score')

    # Panel 2: After stacking → (24, 2048), Channel 0
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(img[0].astype(float), aspect='auto', cmap='RdBu_r',
                     vmin=0, vmax=255, interpolation='nearest')
    for r in range(0, 24, 2):
        ax1.axhline(r - 0.5, color='lime', lw=0.3, alpha=0.5)
    ax1.set_yticks([2*i+0.5 for i in range(12)])
    ax1.set_yticklabels(LEAD_NAMES, fontsize=7)
    ax1.set_title('Channel 0 (RA-ref): each lead → 2 rows  →  (24, 2048) uint8\n'
                  'Pixel = (clip(signal, -3, 3) + 3) / 6 × 255', fontsize=10)
    ax1.set_xlabel('Pixel column (width=2048)')
    plt.colorbar(im1, ax=ax1, fraction=0.015, label='uint8 [0-255]')

    # Panel 3,4,5: all 3 channels
    ch_labels = ['Ch0: RA-referenced', 'Ch1: LA-referenced', 'Ch2: LL-referenced']
    for ch_idx in range(3):
        ax = fig.add_subplot(gs[2 + ch_idx])
        im = ax.imshow(img[ch_idx].astype(float), aspect='auto', cmap='viridis',
                       vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(f'{ch_labels[ch_idx]}  (24, 2048) uint8', fontsize=10)
        ax.set_yticks([2*i+0.5 for i in range(12)])
        ax.set_yticklabels(LEAD_NAMES, fontsize=7)
        ax.set_xlabel('Pixel column')
        plt.colorbar(im, ax=ax, fraction=0.015, label='[0-255]')

    plt.suptitle(
        f'{DATASET_INFO.get(ds,{}).get("label","Dataset")}  |  Record {record_id}\n'
        f'2D Image Construction: (12, 5000) signal  →  (3, 24, 2048) uint8\n'
        f'Kim et al. (2025): 12 leads × 2 rows = 24 rows per channel',
        fontweight='bold', fontsize=11, y=1.01
    )
    _save(fig, out_dir / '03_2d_image_construction.png')


def plot_comparison_grid(steps: dict, ds: str, record_id: str, out_dir: Path):
    """Side-by-side comparison of raw vs processed for each lead."""
    color = DATASET_INFO.get(ds, {}).get('color', '#555555')
    fs_raw = DATASET_INFO.get(ds, {}).get('native_fs', 400)
    fig, axes = plt.subplots(12, 3, figsize=(18, 24), sharex=False)
    step_info = [
        ('Raw WFDB',      steps['raw'],    fs_raw),
        ('100 Hz FM',     steps['sig_100'],TARGET_FS_FM),
        ('500 Hz z-score',steps['sig_500_z'],TARGET_FS_IMAGE),
    ]
    for lead in range(12):
        for col, (label, sig, fs) in enumerate(step_info):
            ax = axes[lead, col]
            t  = np.arange(sig.shape[1]) / fs
            ax.plot(t, sig[lead], lw=0.5, color=color)
            if lead == 0:
                ax.set_title(label, fontsize=8)
            if col == 0:
                ax.set_ylabel(LEAD_NAMES[lead], fontsize=8, rotation=0, labelpad=24)
            ax.grid(True, alpha=0.2); ax.set_xticks([])
    plt.suptitle(
        f'{DATASET_INFO.get(ds,{}).get("label","Dataset")}  |  Record {record_id}\n'
        f'All 12 leads across 3 pipeline stages',
        fontweight='bold', fontsize=11
    )
    plt.tight_layout()
    _save(fig, out_dir / '04_all_leads_comparison.png')


# ── Main ─────────────────────────────────────────────────────────────────────
def validate_single_ecg(ds: str, record_id: str):
    ds = ds.lower().replace('-', '').replace('_', '')
    # Normalise aliases
    if ds in ('samitrop', 'samitrops'):
        ds = 'samitrop'

    out_dir = OUT_BASE / ds / f"ecg_{record_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Validating: {ds.upper()} | record {record_id}")
    print(f"Output dir: {out_dir}")
    print('='*60)

    # ── Load raw WFDB ────────────────────────────────────────────
    raw, fs = load_wfdb(ds, record_id)
    print(f"  Loaded WFDB: shape={raw.shape}  fs={fs}")

    # ── Run pipeline ─────────────────────────────────────────────
    steps = run_pipeline(raw, fs, ds)
    print(f"  Pipeline: 100Hz shape={steps['sig_100'].shape}  "
          f"500Hz shape={steps['sig_500_z'].shape}  "
          f"image shape={steps['image'].shape}")

    # ── Compare with saved files (if present) ────────────────────
    try:
        saved_1d = load_saved_1d(ds, record_id)
        print(f"  Saved 1D: {saved_1d.shape}  "
              f"match={'OK' if np.allclose(saved_1d, steps['sig_100'], atol=1e-4) else 'MISMATCH'}")
    except FileNotFoundError as e:
        print(f"  Saved 1D: not found ({e})")

    try:
        saved_2d = load_saved_2d(ds, record_id)
        match = np.allclose(saved_2d.astype(float), steps['image'].astype(float), atol=1)
        print(f"  Saved 2D: {saved_2d.shape}  match={'OK' if match else 'MISMATCH'}")
    except FileNotFoundError as e:
        print(f"  Saved 2D: not found ({e})")

    # ── Generate plots ────────────────────────────────────────────
    print("  Generating plots...")
    plot_pipeline_overview(steps, ds, record_id, out_dir)
    print("    [OK] 01_pipeline_lead_I.png")

    plot_all_12_leads(steps['raw'], fs,
                      f"{ds.upper()} Record {record_id} — Raw 12 Leads  ({raw.shape})",
                      out_dir / '02_raw_12leads.png',
                      color=DATASET_INFO.get(ds,{}).get('color','#555'))
    print("    [OK] 02_raw_12leads.png")

    plot_2d_image_construction(steps, ds, record_id, out_dir)
    print("    [OK] 03_2d_image_construction.png")

    plot_comparison_grid(steps, ds, record_id, out_dir)
    print("    [OK] 04_all_leads_comparison.png")

    # Extra: 100 Hz normalized (what model actually receives)
    sig_100_z = normalize_per_lead(steps['sig_100'], clip_std=3.0)
    plot_all_12_leads(sig_100_z, TARGET_FS_FM,
                      f"{ds.upper()} Record {record_id} — 1D FM Input (12, 1000) @ 100 Hz z-scored",
                      out_dir / '05_fm_input_12leads.png',
                      color=DATASET_INFO.get(ds,{}).get('color','#555'))
    print("    [OK] 05_fm_input_12leads.png")

    print(f"\n  [DONE] Complete. {len(list(out_dir.glob('*.png')))} plots in {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate ECG preprocessing pipeline.')
    parser.add_argument('--dataset', choices=['ptbxl', 'samitrop', 'code15'])
    parser.add_argument('--id', type=str)
    parser.add_argument('--all', action='store_true',
                        help='Run one sample from each dataset')
    args = parser.parse_args()

    if args.all:
        for ds, rid in [('ptbxl', '1'), ('samitrop', '3629'), ('code15', '13')]:
            try:
                validate_single_ecg(ds, rid)
            except FileNotFoundError as e:
                print(f"  SKIP {ds}: {e}")
    elif args.dataset and args.id:
        validate_single_ecg(args.dataset, args.id)
    else:
        parser.print_help()
