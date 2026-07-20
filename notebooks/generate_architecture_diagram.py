"""
ChagaSight Architecture Diagram v2 — Thesis-Quality
Professional research paper style with colored section boxes.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 7,
})

fig = plt.figure(figsize=(36, 20), dpi=200)
ax = fig.add_axes([0.01, 0.01, 0.98, 0.96])
ax.set_xlim(0, 36)
ax.set_ylim(0, 20)
ax.set_aspect('equal')
ax.axis('off')

# ─── Colors ────────────────────────────────────────────────────────────────────
C = {
    'data_bg': '#E8F0FE', 'data_bd': '#4285F4',
    'prep_bg': '#E8F5E9', 'prep_bd': '#34A853',
    'aug_bg':  '#FFF3E0', 'aug_bd':  '#E65100',
    '2d_bg':   '#E3F2FD', '2d_bd':   '#1565C0',
    '1d_bg':   '#F3E5F5', '1d_bd':   '#7B1FA2',
    'repa_bg': '#E8F5E9', 'repa_bd': '#2E7D32',
    'cls_bg':  '#FFEBEE', 'cls_bd':  '#C62828',
    'train_bg':'#FFF8E1', 'train_bd':'#F57F17',
    'demo_bg': '#FCE4EC', 'demo_bd': '#AD1457',
    'dk_blue': '#0D47A1', 'white':   '#FFFFFF',
    'gray':    '#424242', 'lt_blue': '#D2E3FC',
}

def rbox(x, y, w, h, fc, ec, lw=1.5, a=0.7, z=1, r=0.05):
    b = FancyBboxPatch((x,y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=a, zorder=z)
    ax.add_patch(b); return b

def blk(x, y, w, h, txt, fc='#1A73E8', tc='white', fs=6.5, ec=None, lw=1, a=0.9, z=3, fw='normal', r=0.04):
    if ec is None: ec = fc
    b = FancyBboxPatch((x,y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=a, zorder=z)
    ax.add_patch(b)
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=fs,
            color=tc, fontweight=fw, zorder=z+1, linespacing=1.2)
    return b

def arr(x1, y1, x2, y2, c='#424242', lw=1.2, ms=12, z=5):
    a = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='->', mutation_scale=ms,
                         color=c, linewidth=lw, zorder=z)
    ax.add_patch(a); return a

def slbl(x, y, txt, c, fs=10):
    ax.text(x, y, txt, ha='left', va='bottom', fontsize=fs, color=c, fontweight='bold', zorder=10)

def dlbl(x, y, txt, fs=5.5, c='#616161'):
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, color=c, fontstyle='italic', zorder=6)

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(18, 19.6, 'ChagaSight: Dual-Pathway Hybrid Architecture for Chagas Disease Detection',
        ha='center', va='center', fontsize=15, fontweight='bold', color=C['dk_blue'], zorder=10)

# ═══════════════════════════════════════════════════════════════════════════════
# A. DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.2, 10.2, 8.3, 9.0, C['data_bg'], C['data_bd'], lw=2, a=0.25, z=0)
slbl(0.35, 19.0, 'A. Data Pipeline', C['data_bd'])

# Datasets
rbox(0.4, 16.0, 3.2, 2.8, C['white'], C['data_bd'], lw=1.2, a=0.8, z=1)
ax.text(2.0, 18.65, 'ECG Databases', ha='center', va='center', fontsize=8, fontweight='bold', color=C['data_bd'], zorder=5)

blk(0.6, 17.7, 2.8, 0.65, 'PTB-XL\n(~21,837 records)', fc='#1A73E8', fs=6)
blk(0.6, 16.9, 2.8, 0.65, 'SaMi-Trop', fc='#1A73E8', fs=6)
blk(0.6, 16.1, 2.8, 0.65, 'CODE-15%\n(soft labels: 0.2 / 0.8)', fc='#5C6BC0', fs=5.5)

# Split
arr(3.7, 17.3, 4.1, 17.3)
rbox(4.1, 16.0, 2.2, 2.3, C['white'], C['data_bd'], lw=1.2, a=0.8, z=1)
ax.text(5.2, 18.1, 'Final Split', ha='center', va='center', fontsize=7.5, fontweight='bold', color=C['data_bd'], zorder=5)
blk(4.25, 17.4, 1.9, 0.5, 'Train', fc='#43A047', fs=6.5)
blk(4.25, 16.75, 1.9, 0.5, 'Validation', fc='#FB8C00', fs=6.5)
blk(4.25, 16.1, 1.9, 0.5, 'Test (held-out)', fc='#E53935', fs=6.5)

# Weighted sampler
arr(6.35, 17.65, 6.7, 17.65)
blk(6.7, 16.8, 1.6, 1.2, 'Weighted\nRandom\nSampler\n(5x pos weight)', fc='#00897B', fs=5.5)

# ─── B. Preprocessing ────────────────────────────────────────────────────────
rbox(0.4, 10.5, 8.0, 5.2, C['prep_bg'], C['prep_bd'], lw=1.5, a=0.25, z=0)
slbl(0.55, 15.5, 'B. Data Preprocessing', C['prep_bd'], fs=9)

blk(0.6, 14.0, 1.5, 0.9, 'Raw 12-Lead\nECG Signal', fc='#546E7A', fs=6)
arr(2.15, 14.45, 2.5, 14.45)
blk(2.5, 14.0, 1.4, 0.9, 'Resample\n500Hz (2D)\n100Hz (1D)', fc='#34A853', fs=5.5)
arr(3.95, 14.45, 4.2, 14.45)
blk(4.2, 14.0, 1.5, 0.9, 'Bandpass Filter\n0.5-40 Hz\n(Butterworth\n4th order)', fc='#34A853', fs=5.2)
arr(5.75, 14.45, 6.0, 14.45)
blk(6.0, 14.0, 1.4, 0.9, 'Z-Score\nNormalize\n(per-lead)\nclip +/-3s', fc='#34A853', fs=5.2)

# Split to 2 paths
ax.plot([6.7, 6.7, 4.5, 4.5], [14.0, 13.4, 13.4, 13.0], color=C['prep_bd'], lw=1.2, zorder=4)
ax.plot([6.7, 6.7], [13.4, 13.0], color=C['prep_bd'], lw=1.2, zorder=4)

ax.text(3.0, 13.2, '2D Path', ha='center', va='center', fontsize=6.5, fontweight='bold', color=C['2d_bd'], zorder=5)
ax.text(6.7, 13.2, '1D Path', ha='center', va='center', fontsize=6.5, fontweight='bold', color=C['1d_bd'], zorder=5)

blk(0.6, 10.8, 3.8, 1.8, 
    'WCT Image Embedding (Kim et al. 2025)\n\n'
    'Ch.0: RA-referenced view (12 leads)\n'
    'Ch.1: LA-referenced view (12 leads)\n'
    'Ch.2: LL-referenced view (12 leads)\n'
    'clip +/-3s  ->  uint8 [0, 255]',
    fc='#1565C0', fs=5.2)
dlbl(2.5, 10.6, 'Output: (B, 3, 24, 2048)', fs=5.2)
arr(4.5, 13.0, 2.5, 12.6, c=C['2d_bd'])

blk(5.0, 10.8, 2.8, 1.8,
    '1D Signal Output\n\n'
    '12 leads x 1000 samples\n'
    '100 Hz, 10 seconds\n'
    'float32',
    fc='#7B1FA2', fs=5.5)
dlbl(6.4, 10.6, 'Output: (B, 12, 1000)', fs=5.2)
arr(6.7, 13.0, 6.4, 12.6, c=C['1d_bd'])

# ═══════════════════════════════════════════════════════════════════════════════
# C. DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
rbox(8.8, 10.2, 3.8, 5.3, C['aug_bg'], C['aug_bd'], lw=1.5, a=0.25, z=0)
slbl(8.95, 15.3, 'C. Data Augmentation', C['aug_bd'], fs=9)
ax.text(10.7, 14.95, '(1D Signal Only - Training)', ha='center', va='center', fontsize=5.5, color='#BF360C', fontstyle='italic', zorder=5)

augs = [
    ('Lead Mixup',        'p=0.3, a=0.2, Beta interp.'),
    ('Powerline Noise',   'p=0.5, SNR 15-30 dB, 50/60Hz'),
    ('Random Shift',      'p=0.5, +/-100 samples'),
    ('Amplitude Scaling', 'p=0.3, scale [0.8, 1.2]'),
    ('Baseline Wander',   'p=0.2, 0.1-0.5 Hz'),
]
ay = 14.2
for name, desc in augs:
    blk(8.95, ay, 1.5, 0.5, name, fc='#E65100', fs=5)
    ax.text(11.6, ay+0.25, desc, ha='center', va='center', fontsize=4.8, color=C['gray'], zorder=5)
    ay -= 0.65

# Connect 1D output to augmentation
arr(7.8, 11.7, 8.8, 11.7, c=C['1d_bd'])

# ═══════════════════════════════════════════════════════════════════════════════
# D. 2D-ViT BRANCH  
# ═══════════════════════════════════════════════════════════════════════════════
rbox(13.0, 12.0, 14.0, 7.5, C['2d_bg'], C['2d_bd'], lw=2, a=0.2, z=0)
slbl(13.15, 19.25, 'D. 2D-ViT Branch (Contour Image Pathway)', C['2d_bd'], fs=9)

# Input
blk(13.2, 16.2, 1.5, 1.3, 'Input:\n2D ECG Image\n(B, 3, 24, 2048)', fc='#546E7A', fs=5.5)

# MAE pretrained
blk(13.2, 15.2, 1.5, 0.7, 'MAE Pretrained\n(optional)', fc='#78909C', fs=5)
arr(13.95, 15.9, 13.95, 16.2, c='#78909C', lw=0.8)

# Connect from data pipeline
ax.plot([2.5, 2.5, 12.5, 12.5, 13.2], [10.6, 10.0, 10.0, 16.85, 16.85], color=C['2d_bd'], lw=1.3, zorder=4)
ax.annotate('', xy=(13.2, 16.85), xytext=(12.5, 16.85), arrowprops=dict(arrowstyle='->', color=C['2d_bd'], lw=1.3))

# PatchEmbed2D
arr(14.75, 16.85, 15.2, 16.85, c=C['2d_bd'])
blk(15.2, 16.0, 2.0, 1.7,
    'PatchEmbed2D\n\nConv2d(3 -> 768)\nkernel = (8, 64)\nstride = (8, 64)\n\n3 x 32 = 96 patches',
    fc='#1565C0', fs=5.2)
dlbl(16.2, 15.8, '(B, 96, 768)', fs=5)

# Pos Embed
arr(17.25, 16.85, 17.6, 16.85, c=C['2d_bd'])
blk(17.6, 16.2, 1.4, 1.3,
    '+ Pos Embed\n(1, 96, 768)\ntrunc_normal\nstd=0.02\n+ Dropout(0.1)',
    fc='#42A5F5', fs=5)

# Transformer Encoder
arr(19.05, 16.85, 19.4, 16.85, c=C['2d_bd'])
rbox(19.4, 15.2, 3.0, 3.6, '#BBDEFB', C['2d_bd'], lw=1.5, a=0.5, z=1)
ax.text(20.9, 18.55, 'Transformer Encoder', ha='center', va='center', fontsize=7.5, fontweight='bold', color=C['2d_bd'], zorder=5)

blk(19.55, 17.5, 2.7, 0.7,
    'Multi-Head Self-Attention\n(heads = 12, head_dim = 64)',
    fc='#0D47A1', fs=5.5)
blk(19.55, 16.6, 2.7, 0.65,
    'Add & LayerNorm (Pre-LN)',
    fc='#1565C0', fs=5.5)
blk(19.55, 15.65, 2.7, 0.7,
    'MLP (768 -> 3072 -> 768)\nGELU, Dropout(0.1)',
    fc='#0D47A1', fs=5.5)

# x12 badge
ax.text(22.7, 16.9, 'x12\nlayers', ha='center', va='center', fontsize=8, fontweight='bold',
        color=C['2d_bd'], zorder=6,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=C['white'], edgecolor=C['2d_bd'], lw=1.5))

# LayerNorm
arr(22.45, 16.2, 23.2, 16.2, c=C['2d_bd'])
blk(23.2, 15.8, 1.2, 0.8, 'LayerNorm\n(768)', fc='#42A5F5', fs=6)

# AoL
arr(24.45, 16.2, 24.8, 16.2, c=C['2d_bd'])
rbox(24.8, 15.0, 2.0, 2.5, '#E3F2FD', C['2d_bd'], lw=1.2, a=0.5, z=1)
ax.text(25.8, 17.3, 'AoL', ha='center', va='center', fontsize=8, fontweight='bold', color=C['2d_bd'], zorder=5)
ax.text(25.8, 16.9, '(Aggregation of', ha='center', va='center', fontsize=5.5, color=C['2d_bd'], zorder=5)
ax.text(25.8, 16.55, 'Layers)', ha='center', va='center', fontsize=5.5, color=C['2d_bd'], zorder=5)
blk(24.95, 15.2, 1.7, 1.0,
    'Mean-pool each\nlayer across patches\nstack 12 layers\naverage',
    fc='#1565C0', fs=4.8)
dlbl(25.8, 14.9, 'f_img (B, 768)', fs=5.5, c=C['2d_bd'])

# Config box
rbox(15.0, 12.3, 9.5, 0.6, C['lt_blue'], C['2d_bd'], lw=1, a=0.6, z=1)
ax.text(19.75, 12.6, '2D-ViT Config:  DEPTH=12 | HEADS=12 | EMBED_DIM=768 | FFN_DIM=3072 | DROPOUT=0.1 | PATCH=(8,64) | Pre-LN',
        ha='center', va='center', fontsize=5.5, color=C['dk_blue'], fontweight='bold', zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# E. 1D-ViT FM BRANCH
# ═══════════════════════════════════════════════════════════════════════════════
rbox(13.0, 3.5, 14.0, 8.2, C['1d_bg'], C['1d_bd'], lw=2, a=0.15, z=0)
slbl(13.15, 11.45, 'E. 1D-ViT FM Branch (Signal + Demographics)', C['1d_bd'], fs=9)

# Input
blk(13.2, 8.5, 1.5, 1.3, 'Input:\n12-Lead ECG\nSignal\n(B, 12, 1000)', fc='#546E7A', fs=5.5)

# Connect from augmentation
ax.plot([10.7, 10.7, 12.5, 12.5, 13.2], [10.2, 9.5, 9.5, 9.15, 9.15], color=C['1d_bd'], lw=1.3, zorder=4)
ax.annotate('', xy=(13.2, 9.15), xytext=(12.5, 9.15), arrowprops=dict(arrowstyle='->', color=C['1d_bd'], lw=1.3))

# ST-MEM pretrained
blk(13.2, 7.5, 1.5, 0.7, 'ST-MEM Pretrained\n(optional)', fc='#78909C', fs=5)
arr(13.95, 8.2, 13.95, 8.5, c='#78909C', lw=0.8)

# PatchEmbed1D
arr(14.75, 9.15, 15.2, 9.15, c=C['1d_bd'])
blk(15.2, 8.1, 2.0, 2.1,
    'PatchEmbed1D\n\nConv1d(1 -> 768)\nkernel = 50\nstride = 50\nper lead\n\n12 x 20 = 240 patches\n+ Lead Embed (1,12,1,768)',
    fc='#7B1FA2', fs=4.8)
dlbl(16.2, 7.9, '(B, 240, 768)', fs=5)

# Pos Embed
arr(17.25, 9.15, 17.6, 9.15, c=C['1d_bd'])
blk(17.6, 8.5, 1.4, 1.3,
    '+ Pos Embed\n(1, 240, 768)\ntrunc_normal\nstd=0.02\n+ Dropout(0.1)',
    fc='#AB47BC', fs=5)

# Transformer Encoder
arr(19.05, 9.15, 19.4, 9.15, c=C['1d_bd'])
rbox(19.4, 7.5, 3.0, 3.6, '#F3E5F5', C['1d_bd'], lw=1.5, a=0.5, z=1)
ax.text(20.9, 10.85, 'Transformer Encoder', ha='center', va='center', fontsize=7.5, fontweight='bold', color=C['1d_bd'], zorder=5)

blk(19.55, 9.8, 2.7, 0.7,
    'Multi-Head Self-Attention\n(heads = 12, head_dim = 64)',
    fc='#6A1B9A', fs=5.5)
blk(19.55, 8.9, 2.7, 0.65,
    'Add & LayerNorm (Pre-LN)',
    fc='#7B1FA2', fs=5.5)
blk(19.55, 7.95, 2.7, 0.7,
    'MLP (768 -> 3072 -> 768)\nGELU, Dropout(0.1)',
    fc='#6A1B9A', fs=5.5)

# x12 badge
ax.text(22.7, 9.2, 'x12\nlayers', ha='center', va='center', fontsize=8, fontweight='bold',
        color=C['1d_bd'], zorder=6,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=C['white'], edgecolor=C['1d_bd'], lw=1.5))

# LayerNorm
arr(22.45, 8.5, 23.2, 8.5, c=C['1d_bd'])
blk(23.2, 8.1, 1.2, 0.8, 'LayerNorm\n(768)', fc='#AB47BC', fs=6)

# AoL
arr(24.45, 8.5, 24.8, 8.5, c=C['1d_bd'])
rbox(24.8, 7.3, 2.0, 2.5, '#F3E5F5', C['1d_bd'], lw=1.2, a=0.5, z=1)
ax.text(25.8, 9.6, 'AoL', ha='center', va='center', fontsize=8, fontweight='bold', color=C['1d_bd'], zorder=5)
ax.text(25.8, 9.2, '(Aggregation of', ha='center', va='center', fontsize=5.5, color=C['1d_bd'], zorder=5)
ax.text(25.8, 8.85, 'Layers)', ha='center', va='center', fontsize=5.5, color=C['1d_bd'], zorder=5)
blk(24.95, 7.5, 1.7, 1.0,
    'Mean-pool each\nlayer across patches\nstack 12 layers\naverage',
    fc='#7B1FA2', fs=4.8)

# ─── Demographics / FiLM ────────────────────────────────────────────────────
rbox(13.2, 4.0, 6.0, 3.2, C['demo_bg'], C['demo_bd'], lw=1.5, a=0.35, z=0)
slbl(13.35, 7.0, 'Demographics Encoder (FiLM)', C['demo_bd'], fs=7.5)

blk(13.4, 5.7, 1.2, 0.9, 'Input:\nAge, Sex\n(B, 2)', fc='#880E4F', fs=5.5)
arr(14.65, 6.15, 15.0, 6.15, c=C['demo_bd'])
blk(15.0, 5.3, 2.2, 1.7,
    'MLP\nLinear(2 -> 256)\nReLU\nLinear(256 -> 256)\nReLU\nLinear(256 -> 1536)',
    fc='#AD1457', fs=5)
arr(17.25, 6.15, 17.6, 6.15, c=C['demo_bd'])
blk(17.6, 5.1, 1.4, 2.1,
    'Split\n\ngamma (B,768)\nbeta (B,768)\n\ninit: gamma=1\nbeta=0',
    fc='#C2185B', fs=5)

# FiLM formula
ax.text(15.5, 4.3, 'FiLM: f_sig = gamma * f_sig + beta',
        ha='center', va='center', fontsize=6.5, color=C['demo_bd'], fontweight='bold', fontstyle='italic', zorder=5,
        bbox=dict(boxstyle='round,pad=0.2', facecolor=C['white'], edgecolor=C['demo_bd'], lw=0.8))

# FiLM modulation block
blk(20.0, 5.0, 2.0, 1.2, 'FiLM\nModality\nWeighting\ngamma * f + beta', fc='#AD1457', fs=5)
arr(19.05, 6.15, 20.0, 5.8, c=C['demo_bd'])

# AoL to FiLM
ax.plot([25.8, 25.8, 22.5, 22.5, 21.0, 21.0], [7.3, 5.6, 5.6, 5.6, 5.6, 5.6], 
        color=C['1d_bd'], lw=1.2, zorder=4)
arr(22.0, 5.6, 21.0, 5.6, c=C['1d_bd'])
ax.text(24.0, 5.4, 'f_sig (B, 768)', ha='center', va='center', fontsize=5, color=C['1d_bd'], fontweight='bold', zorder=5)

dlbl(21.0, 4.5, 'f_sig modulated (B, 768)', fs=5.5, c=C['1d_bd'])

# 1D Config box
rbox(15.0, 3.7, 9.5, 0.55, '#F3E5F5', C['1d_bd'], lw=1, a=0.5, z=1)
ax.text(19.75, 3.95, '1D-ViT FM Config:  DEPTH=12 | HEADS=12 | EMBED_DIM=768 | FFN_DIM=3072 | DROPOUT=0.1 | PATCH=50 | LEADS=12',
        ha='center', va='center', fontsize=5.5, color='#4A148C', fontweight='bold', zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# F. REPA ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
rbox(27.5, 11.5, 2.7, 4.0, C['repa_bg'], C['repa_bd'], lw=2, a=0.3, z=0)
slbl(27.65, 15.3, 'F. REPA Alignment', C['repa_bd'], fs=8)

blk(27.7, 14.0, 2.3, 0.7, 'DepthwiseConv1d\n(768->768, k=1, g=768)', fc='#2E7D32', fs=5)
arr(28.85, 14.0, 28.85, 13.65, c=C['repa_bd'])
blk(27.7, 12.9, 2.3, 0.7, 'SiLU Activation', fc='#388E3C', fs=6)
arr(28.85, 12.9, 28.85, 12.55, c=C['repa_bd'])
blk(27.7, 11.8, 2.3, 0.7, 'Linear (768 -> 768)', fc='#2E7D32', fs=6)
dlbl(28.85, 11.55, 'aligned_2d (B, 768)', fs=5, c=C['repa_bd'])

# Connect 2D to REPA
ax.plot([25.8, 25.8, 28.85, 28.85], [14.9, 14.9, 14.9, 14.7], color=C['2d_bd'], lw=1.5, zorder=4)
ax.annotate('', xy=(28.85, 14.7), xytext=(28.85, 14.9), arrowprops=dict(arrowstyle='->', color=C['2d_bd'], lw=1.3))
ax.text(27.3, 15.1, 'f_img (B, 768)', ha='center', va='center', fontsize=5.5, color=C['2d_bd'], fontweight='bold', zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# G. FUSION & CLASSIFICATION HEAD
# ═══════════════════════════════════════════════════════════════════════════════
rbox(27.5, 1.5, 8.2, 9.5, C['cls_bg'], C['cls_bd'], lw=2, a=0.15, z=0)
slbl(27.65, 10.8, 'G. Fusion & Classification Head', C['cls_bd'], fs=9)

# Concat
blk(28.5, 9.4, 2.3, 0.9, 'Concatenation\n[aligned_2d, f_sig]', fc='#00838F', fs=6)
dlbl(29.65, 9.2, '(B, 1536)', fs=5.5)

# Arrows to concat
arr(28.85, 11.5, 28.85, 10.75, c=C['repa_bd'], lw=1.5)
arr(28.85, 10.75, 29.65, 10.3, c=C['repa_bd'], lw=1.3)

# f_sig to concat
ax.plot([21.0, 21.0, 23.0, 23.0, 29.65, 29.65], [5.0, 4.5, 4.5, 10.0, 10.0, 10.3],
        color=C['1d_bd'], lw=1.3, zorder=4)
ax.annotate('', xy=(29.65, 10.3), xytext=(29.65, 10.0), arrowprops=dict(arrowstyle='->', color=C['1d_bd'], lw=1.3))
ax.text(26.5, 10.15, 'f_sig (B, 768)', ha='center', va='center', fontsize=5.5, color=C['1d_bd'], fontweight='bold', zorder=5)

# Classifier blocks
cx = 29.0
blk(cx, 8.2, 2.5, 0.65, 'Linear (1536 -> 512)', fc='#C62828', fs=6.5)
arr(cx+1.25, 8.2, cx+1.25, 7.95, c=C['cls_bd'])
blk(cx, 7.3, 2.5, 0.6, 'ReLU + Dropout(0.3)', fc='#E53935', fs=6)
arr(cx+1.25, 7.3, cx+1.25, 7.05, c=C['cls_bd'])
blk(cx, 6.4, 2.5, 0.6, 'Linear (512 -> 256)', fc='#C62828', fs=6.5)
arr(cx+1.25, 6.4, cx+1.25, 6.15, c=C['cls_bd'])
blk(cx, 5.5, 2.5, 0.6, 'ReLU + Dropout(0.3)', fc='#E53935', fs=6)
arr(cx+1.25, 5.5, cx+1.25, 5.25, c=C['cls_bd'])
blk(cx, 4.6, 2.5, 0.6, 'Linear (256 -> 1)', fc='#C62828', fs=6.5)
arr(cx+1.25, 4.6, cx+1.25, 4.35, c=C['cls_bd'])
blk(cx, 3.7, 2.5, 0.55, 'Logit (B, 1)', fc='#B71C1C', fs=7)
arr(cx+1.25, 3.7, cx+1.25, 3.45, c=C['cls_bd'])
blk(cx, 2.85, 2.5, 0.55, 'Sigmoid', fc='#880E4F', fs=7)

arr(cx+1.25, 9.4, cx+1.25, 8.85, c=C['cls_bd'], lw=1.5)

# Prediction
blk(32.2, 4.8, 3.2, 3.5,
    'Prediction\n\nP(Chagas)\n\nPositive\nNegative',
    fc='#1B5E20', fs=7)
arr(31.55, 3.1, 32.2, 4.0, c='#1B5E20', lw=1.5)

# Input/Output Summary
rbox(32.2, 8.8, 3.3, 2.0, C['lt_blue'], C['dk_blue'], lw=1.5, a=0.7, z=2)
ax.text(33.85, 10.6, 'Overall Input / Output', ha='center', va='center', fontsize=7, fontweight='bold', color=C['dk_blue'], zorder=5)
io_lines = [
    'Inputs:',
    '  Images: (B, 3, 24, 2048)',
    '  Signal: (B, 12, 1000)',
    '  Age: (B,)  Sex: (B,)',
    'Output:',
    '  P(Chagas) in [0, 1]',
]
iy = 10.2
for line in io_lines:
    ax.text(32.4, iy, line, ha='left', va='center', fontsize=5.5, color=C['dk_blue'], zorder=5)
    iy -= 0.25

# Total params
blk(32.0, 2.0, 3.5, 0.6, 'Total Parameters: ~173M', fc=C['dk_blue'], fs=7, fw='bold')

# ═══════════════════════════════════════════════════════════════════════════════
# H. TRAINING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════
rbox(0.2, 0.3, 12.4, 9.6, C['train_bg'], C['train_bd'], lw=2, a=0.2, z=0)
slbl(0.35, 9.7, 'H. Training Strategy', C['train_bd'], fs=10)

# Phase 1
rbox(0.4, 5.8, 5.8, 3.6, '#FFF9C4', '#F9A825', lw=1.5, a=0.5, z=1)
ax.text(3.3, 9.2, 'Phase 1: FM Frozen', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#E65100', zorder=5)

p1 = ['Frozen: all 1D-ViT FM params (~85M)',
      'Iterations: 2,000',
      'Optimizer: AdamW (lr = 2e-4)',
      'Grad Accumulation: 8 -> eff.batch = 64',
      'LR: Linear warmup (200 steps) -> constant',
      'Weight decay: 1e-4',
      'AMP: mixed precision (float16)']
py = 8.7
for item in p1:
    ax.text(0.65, py, '- ' + item, ha='left', va='center', fontsize=5.5, color=C['gray'], zorder=5)
    py -= 0.4

# Phase 2
rbox(6.5, 5.8, 5.9, 3.6, '#C8E6C9', '#2E7D32', lw=1.5, a=0.5, z=1)
ax.text(9.45, 9.2, 'Phase 2: Full Unfreezing', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1B5E20', zorder=5)

p2 = ['All parameters unfrozen (~173M)',
      'Epoch-based: max 50 epochs',
      'Early stopping: patience=10, delta=1e-4',
      'Differential LR:',
      '   ViT (2D+1D): lr = 2e-5 (low)',
      '   REPA + Classifier: lr = 2e-4 (high)',
      'Grad Accumulation: 4 -> eff.batch = 32']
py = 8.7
for item in p2:
    ax.text(6.75, py, '- ' + item, ha='left', va='center', fontsize=5.5, color=C['gray'], zorder=5)
    py -= 0.4

arr(6.25, 7.6, 6.5, 7.6, c=C['train_bd'], lw=2, ms=15)

# Loss function
rbox(0.4, 0.6, 5.8, 4.9, '#FFCCBC', '#BF360C', lw=1.5, a=0.4, z=1)
ax.text(3.3, 5.3, 'Loss Function', ha='center', va='center', fontsize=9, fontweight='bold', color='#BF360C', zorder=5)

blk(0.6, 3.8, 5.4, 1.2,
    'AsymmetricBCELoss (Van Santvliet et al. 2025)\nL = -w_pos * (1-p)^g+ * y * log(p) - p^g- * (1-y) * log(1-p)\ng+ = 0, g- = 2.0, pos_weight = 10.0',
    fc='#BF360C', fs=5.2)

blk(0.6, 2.4, 5.4, 1.1,
    'Cosine Similarity Alignment Loss (Kim et al. 2025)\nL_align = 1 - cos(aligned_2d, f_sig.detach())\nFM features DETACHED (not trained through alignment)',
    fc='#D84315', fs=5.2)

blk(0.6, 0.9, 5.4, 1.2,
    'Combined Loss\nL_total = L_bce + lambda * L_align   (lambda = 0.5)\nAMP (float16) | Gradient clipping: max_norm = 1.0\nBest checkpoint: highest AUROC on validation',
    fc='#E64A19', fs=5.2)

# Evaluation metrics
rbox(6.5, 0.6, 5.9, 4.9, '#E1F5FE', '#0277BD', lw=1.5, a=0.4, z=1)
ax.text(9.45, 5.3, 'Evaluation & Metrics', ha='center', va='center', fontsize=9, fontweight='bold', color='#0277BD', zorder=5)

ev = ['Primary metric: AUROC (best checkpoint)',
      'Official metric: TPR@5% (PhysioNet 2025)',
      '',
      'Threshold Strategies:',
      '  1. Youden: argmax(TPR - FPR)',
      '  2. Min Recall >= 0.99 -> max precision',
      '  3. Min Precision >= 0.30 -> max recall',
      '  4. Max F1 (scan all thresholds)',
      '  5. Fixed T = 0.5',
      '',
      'Per strategy: TP, FP, TN, FN, Accuracy,',
      '  Precision, Recall, Specificity, NPV, F1, F2']
ey = 4.9
for item in ev:
    ax.text(6.75, ey, item, ha='left', va='center', fontsize=5.5, color=C['gray'], zorder=5)
    ey -= 0.35

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C['data_bd'], 'Data Pipeline'), (C['prep_bd'], 'Preprocessing'),
    (C['aug_bd'], 'Augmentation'), (C['2d_bd'], '2D-ViT (Image)'),
    (C['1d_bd'], '1D-ViT FM (Signal)'), (C['demo_bd'], 'Demographics'),
    (C['repa_bd'], 'REPA Alignment'), (C['cls_bd'], 'Classification'),
    (C['train_bd'], 'Training'),
]
lx = 13.5
for color, label in legend_items:
    rbox(lx, 0.05, 0.3, 0.25, color, color, lw=1, a=0.8, z=8)
    ax.text(lx+0.4, 0.17, label, ha='left', va='center', fontsize=5.5, color=C['gray'], zorder=8)
    lx += 2.3

# ─── Save ──────────────────────────────────────────────────────────────────────
out = r'D:\IIT\L6\FYP\ChagaSight\checkpoints_new\full_p2_50epochs\plots\final_v2\chagasight_architecture_diagram.png'
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {out}')
plt.close()
