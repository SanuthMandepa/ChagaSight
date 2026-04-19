"""
visualise_wct_embedding.py
Generates a thesis figure illustrating the WCT re-referencing and 2D image
construction pipeline used in ChagaSight's visual pathway.

Three panels:
  1. Conceptual schematic: Einthoven's triangle + WCT formula + three channels
  2. The three re-referenced channel images (RA / LA / LL views)
  3. Final stacked (3, 24, 2048) tensor shape annotation

Output: checkpoints/thesis_figures/fig_c7_wct_embedding.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.preprocessing.image_embedding import _compute_wct_proper, _stack_leads_to_height24, _to_uint8_from_clipped

OUT_PATH = "checkpoints/thesis_figures/fig_c7_wct_embedding.png"

# ── Synthetic ECG generator ──────────────────────────────────────────────────
def _ecg_beat(t, offset=0.0, amp=1.0):
    """Single PQRST beat centred at offset."""
    p  =  0.12 * np.exp(-((t - offset - 0.20) ** 2) / (2 * 0.010 ** 2))
    q  = -0.08 * np.exp(-((t - offset - 0.34) ** 2) / (2 * 0.004 ** 2))
    r  =  1.00 * amp * np.exp(-((t - offset - 0.37) ** 2) / (2 * 0.006 ** 2))
    s  = -0.18 * np.exp(-((t - offset - 0.40) ** 2) / (2 * 0.005 ** 2))
    tw =  0.22 * amp * np.exp(-((t - offset - 0.55) ** 2) / (2 * 0.030 ** 2))
    return p + q + r + s + tw

def make_synthetic_12lead(W=2048, fs=500, n_beats=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(W) / fs

    # Limb potentials (RA fixed at 0)
    beat_offsets = [i * (W / fs / n_beats) for i in range(n_beats)]
    RA = np.zeros(W, dtype=np.float32)
    LA_raw = sum(_ecg_beat(t, o, amp=0.9) for o in beat_offsets)
    LL_raw = sum(_ecg_beat(t, o, amp=0.7) for o in beat_offsets)
    LA = LA_raw.astype(np.float32)
    LL = LL_raw.astype(np.float32)
    WCT = (RA + LA + LL) / 3.0

    # Build standard 12-lead from potentials
    I   = LA - RA
    II  = LL - RA
    III = LL - LA
    aVR = RA - WCT
    aVL = LA - WCT
    aVF = LL - WCT
    # Precordial: V leads + small spatial variation
    V = [(LA + LL) * (0.3 + 0.12 * k) - RA * 0.1 + rng.normal(0, 0.02, W).astype(np.float32) for k in range(6)]

    sig = np.stack([I, II, III, aVR, aVL, aVF, *V], dtype=np.float32)  # (12, W)
    # z-score normalise + clip
    for i in range(12):
        mu, sd = sig[i].mean(), sig[i].std() + 1e-6
        sig[i] = np.clip((sig[i] - mu) / sd, -3, 3)
    return sig, t

# ── Build data ───────────────────────────────────────────────────────────────
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
W = 2048
sig, t = make_synthetic_12lead(W=W)

ch0, ch1, ch2 = _compute_wct_proper(sig)
img0 = _to_uint8_from_clipped(_stack_leads_to_height24(ch0))  # (24, W)
img1 = _to_uint8_from_clipped(_stack_leads_to_height24(ch1))
img2 = _to_uint8_from_clipped(_stack_leads_to_height24(ch2))

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor="white")
fig.suptitle(
    "2D Spatial ECG Image Construction via Wilson's Central Terminal Re-Referencing\n"
    "Three spatial perspectives of the cardiac electrical field stacked into a (3, 24, 2048) tensor",
    fontsize=12, fontweight="bold", y=0.99
)

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.05, right=0.97,
    top=0.93, bottom=0.04,
    hspace=0.55, wspace=0.30
)

ax_triangle = fig.add_subplot(gs[0, 0])   # Einthoven triangle schematic
ax_formula  = fig.add_subplot(gs[0, 1])   # WCT maths
ax_pipeline = fig.add_subplot(gs[0, 2])   # Pipeline steps

ax_ch0 = fig.add_subplot(gs[1, :])        # Channel images side-by-side (handled manually)
ax_ch1 = fig.add_subplot(gs[2, :])        # Tensor shape annotation

# We'll replace ax_ch0 and ax_ch1 with manual axes
for ax in [ax_ch0, ax_ch1]:
    ax.remove()

# ─── Panel 1: Einthoven's Triangle ──────────────────────────────────────────
ax_triangle.set_xlim(-1.4, 1.4)
ax_triangle.set_ylim(-1.5, 1.3)
ax_triangle.axis("off")
ax_triangle.set_title("Einthoven's Triangle\n& Limb Potentials", fontsize=9.5, fontweight="bold")

TCOLOR = {"RA": "#E74C3C", "LA": "#2980B9", "LL": "#27AE60", "WCT": "#8E44AD"}

# Triangle vertices
vRA = np.array([-1.0,  0.8])
vLA = np.array([ 1.0,  0.8])
vLL = np.array([ 0.0, -1.1])
vWCT = (vRA + vLA + vLL) / 3.0

tri = plt.Polygon([vRA, vLA, vLL], fill=False, edgecolor="#AAAAAA", linewidth=1.5, linestyle="--")
ax_triangle.add_patch(tri)

# Nodes
for label, v, col in [("RA", vRA, TCOLOR["RA"]), ("LA", vLA, TCOLOR["LA"]),
                       ("LL", vLL, TCOLOR["LL"]), ("WCT", vWCT, TCOLOR["WCT"])]:
    r = 0.16 if label != "WCT" else 0.13
    c = plt.Circle(v, r, color=col, zorder=5)
    ax_triangle.add_patch(c)
    off = {"RA": (-0.28, 0.0), "LA": (0.22, 0.0), "LL": (0.0, -0.28), "WCT": (0.22, 0.0)}[label]
    ax_triangle.text(v[0] + off[0], v[1] + off[1], label, ha="center", va="center",
                     fontsize=9, fontweight="bold", color=col)

# Lead arrows
def arrow(ax, p1, p2, color, label, label_off=(0, 0)):
    mid = (p1 + p2) / 2
    ax.annotate("", xy=p2, xytext=p1,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    ax.text(mid[0] + label_off[0], mid[1] + label_off[1], label,
            ha="center", va="center", fontsize=7.5, color=color,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

arrow(ax_triangle, vRA, vLA, "#555", "Lead I",  (0,  0.18))
arrow(ax_triangle, vRA, vLL, "#555", "Lead II", (-0.30, -0.20))
arrow(ax_triangle, vLA, vLL, "#555", "Lead III",(0.30, -0.20))
# WCT spokes
for v, col in [(vRA, TCOLOR["RA"]), (vLA, TCOLOR["LA"]), (vLL, TCOLOR["LL"])]:
    ax_triangle.plot([vWCT[0], v[0]], [vWCT[1], v[1]], color=col, lw=0.8, linestyle=":", alpha=0.7)

ax_triangle.text(0, -1.42, "WCT = (RA + LA + LL) / 3", ha="center", fontsize=7.5,
                 style="italic", color=TCOLOR["WCT"],
                 bbox=dict(boxstyle="round", fc="#F8F0FF", ec=TCOLOR["WCT"], lw=0.8))

# ─── Panel 2: WCT Maths ──────────────────────────────────────────────────────
ax_formula.axis("off")
ax_formula.set_title("Limb Potential Recovery\n& Three Reference Views", fontsize=9.5, fontweight="bold")

lines = [
    (0.05, 0.92, "Set RA = 0  (ground reference)", "#555555", 9),
    (0.05, 0.82, "LA = Lead I        (LA - RA, RA=0)", TCOLOR["LA"], 8.5),
    (0.05, 0.74, "LL = Lead II       (LL - RA, RA=0)", TCOLOR["LL"], 8.5),
    (0.05, 0.64, "WCT = (RA + LA + LL) / 3", TCOLOR["WCT"], 8.5),
    (0.05, 0.52, "Channel 0  (RA-referenced)", TCOLOR["RA"], 9),
    (0.05, 0.44, "  All 12 leads expressed", "#333", 8),
    (0.05, 0.37, "  relative to RA electrode", "#333", 8),
    (0.05, 0.27, "Channel 1  (LA-referenced)", TCOLOR["LA"], 9),
    (0.05, 0.19, "  All 12 leads expressed", "#333", 8),
    (0.05, 0.12, "  relative to LA electrode", "#333", 8),
    (0.05, 0.02, "Channel 2  (LL-referenced)", TCOLOR["LL"], 9),
]

for x, y, txt, col, fs in lines:
    ax_formula.text(x, y, txt, transform=ax_formula.transAxes,
                    fontsize=fs, color=col, va="top",
                    fontfamily="monospace" if "=" in txt else "sans-serif")

for y, col in [(0.55, TCOLOR["RA"]), (0.30, TCOLOR["LA"]), (0.05, TCOLOR["LL"])]:
    ax_formula.plot([0.02, 0.98], [y, y], color=col, lw=0.6, alpha=0.3,
                    transform=ax_formula.transAxes)

# ─── Panel 3: Pipeline Steps ─────────────────────────────────────────────────
ax_pipeline.axis("off")
ax_pipeline.set_title("Image Construction Pipeline\n(per record)", fontsize=9.5, fontweight="bold")

steps = [
    ("1", "500 Hz signal\n(12, T)",          "#D5E8D4", "#82B366"),
    ("2", "Centre-crop\n(12, 2048)",          "#DAE8FC", "#6C8EBF"),
    ("3", "WCT re-reference\n3x (12, 2048)", "#E1D5E7", "#9673A6"),
    ("4", "Lead duplication\n3x (24, 2048)", "#FFE6CC", "#D6B656"),
    ("5", "uint8 quantise\n[0, 255]",         "#F8CECC", "#B85450"),
    ("6", "Stack channels\n(3, 24, 2048)",    "#2C3E50", "#FFFFFF"),
]

for i, (num, label, bg, fg) in enumerate(steps):
    y = 0.92 - i * 0.155
    rect = mpatches.FancyBboxPatch((0.05, y - 0.06), 0.90, 0.11,
                                    boxstyle="round,pad=0.01",
                                    facecolor=bg, edgecolor=fg, linewidth=1.2,
                                    transform=ax_pipeline.transAxes)
    ax_pipeline.add_patch(rect)
    ax_pipeline.text(0.12, y, num, transform=ax_pipeline.transAxes,
                     fontsize=10, fontweight="bold", va="center", color=fg)
    ax_pipeline.text(0.28, y, label, transform=ax_pipeline.transAxes,
                     fontsize=7.8, va="center", color="#1A1A1A" if bg != "#2C3E50" else "white")
    if i < len(steps) - 1:
        ax_pipeline.annotate("", xy=(0.50, y - 0.06), xytext=(0.50, y - 0.025),
                              xycoords="axes fraction", textcoords="axes fraction",
                              arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0))

# ─── Three channel images ─────────────────────────────────────────────────────
ch_configs = [
    (img0, "Channel 0: RA-Referenced View", TCOLOR["RA"],
     "Limb leads centred on RA electrode.\nPrecordial leads shifted from WCT to RA."),
    (img1, "Channel 1: LA-Referenced View", TCOLOR["LA"],
     "Limb leads centred on LA electrode.\nPrecordial leads shifted from WCT to LA."),
    (img2, "Channel 2: LL-Referenced View", TCOLOR["LL"],
     "Limb leads centred on LL electrode.\nPrecordial leads shifted from WCT to LL."),
]

ch_left  = [0.05, 0.37, 0.69]
ch_width = 0.29
row_top  = 0.42
row_h    = 0.22

for k, ((img, title, col, desc), left) in enumerate(zip(ch_configs, ch_left)):
    ax = fig.add_axes([left, row_top, ch_width, row_h])
    ax.imshow(img, aspect="auto", cmap="RdBu_r", vmin=0, vmax=255, interpolation="nearest")

    # Lead tick labels on y axis (every other lead)
    ax.set_yticks(np.arange(0, 24, 2) + 0.5)
    ax.set_yticklabels(LEAD_NAMES, fontsize=6.5)
    ax.set_xlabel("Time samples (2048 @ 500 Hz = 4.1 s)", fontsize=7)
    ax.set_title(f"{title}\n{desc}", fontsize=8, color=col, fontweight="bold", pad=4)

    # Coloured border
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(2.0)

# ─── Tensor shape annotation bar ─────────────────────────────────────────────
ax_ann = fig.add_axes([0.05, 0.04, 0.90, 0.10])
ax_ann.axis("off")
ax_ann.set_xlim(0, 10)
ax_ann.set_ylim(0, 1)

ax_ann.text(5, 0.82, "Final tensor shape:  (3 channels, 24 rows, 2048 columns)  dtype=uint8",
            ha="center", va="center", fontsize=10, fontweight="bold", color="#2C3E50")

dims = [
    (1.5,  "3\nchannels",  "#E1D5E7", TCOLOR["WCT"]),
    (4.0,  "24\nrows\n(12 leads x2)", "#DAE8FC", "#4472C4"),
    (7.5,  "2048\ncolumns\n(4.1 s @ 500 Hz)", "#D5E8D4", "#82B366"),
]
for cx, label, bg, ec in dims:
    box = mpatches.FancyBboxPatch((cx - 1.1, 0.02), 2.2, 0.60,
                                   boxstyle="round,pad=0.05",
                                   facecolor=bg, edgecolor=ec, linewidth=1.5)
    ax_ann.add_patch(box)
    ax_ann.text(cx, 0.32, label, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#1A1A1A")

ax_ann.text(2.8, 0.32, "×", ha="center", va="center", fontsize=14, color="#555")
ax_ann.text(5.7, 0.32, "×", ha="center", va="center", fontsize=14, color="#555")

# ── Save ─────────────────────────────────────────────────────────────────────
out = Path(OUT_PATH)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
