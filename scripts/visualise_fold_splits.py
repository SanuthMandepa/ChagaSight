"""
visualise_fold_splits.py
Generates a thesis figure illustrating the stratified 5-fold split strategy.

Produces three panels:
  1. Per-fold dataset breakdown table
  2. Stacked bar chart of dataset distribution per fold
  3. Positive-sample count per fold (verifying class balance)

Output: checkpoints/thesis_figures/fig_c7_fold_splits.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
CSV_PATH   = "data/processed/metadata/combined_5fold.csv"
OUT_PATH   = "checkpoints/thesis_figures/fig_c7_fold_splits.png"
N_FOLDS    = 5
DATASET_LABELS = {"ptbxl": "PTB-XL", "samitrop": "SaMi-Trop", "code15": "CODE-15%"}
COLORS     = {"ptbxl": "#4C72B0", "samitrop": "#DD8452", "code15": "#55A868"}
POS_COLOR  = "#C44E52"

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, low_memory=False)
datasets = ["ptbxl", "samitrop", "code15"]

# Build per-fold stats
rows = []
for fold in range(N_FOLDS):
    fd = df[df["fold"] == fold]
    row = {"Fold": fold}
    for ds in datasets:
        row[ds] = len(fd[fd["dataset"] == ds])
    row["Total"]     = len(fd)
    row["Positives"] = int(fd["label_hard"].sum())
    row["Pos %"]     = f"{100 * fd['label_hard'].mean():.2f}%"
    rows.append(row)

stats = pd.DataFrame(rows)

# ── Layout ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor="white")
fig.suptitle(
    "Stratified Five-Fold Cross-Validation Split Strategy\n"
    "Composite key: (dataset source × hard Chagas label)",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    left=0.06, right=0.97,
    top=0.90, bottom=0.07,
    hspace=0.45, wspace=0.35
)

ax_table = fig.add_subplot(gs[0, :])   # full-width top
ax_bar   = fig.add_subplot(gs[1, 0])   # bottom-left
ax_pos   = fig.add_subplot(gs[1, 1])   # bottom-right

# ─── Panel 1: Table ─────────────────────────────────────────────────────────
ax_table.axis("off")

col_labels = ["Fold", "PTB-XL", "SaMi-Trop", "CODE-15%", "Total", "Positives", "Pos %"]
table_data = [
    [
        f"Fold {r['Fold']}",
        f"{r['ptbxl']:,}",
        f"{r['samitrop']:,}",
        f"{r['code15']:,}",
        f"{r['Total']:,}",
        f"{r['Positives']:,}",
        r["Pos %"],
    ]
    for _, r in stats.iterrows()
]

# Totals row
table_data.append([
    "Total",
    f"{stats['ptbxl'].sum():,}",
    f"{stats['samitrop'].sum():,}",
    f"{stats['code15'].sum():,}",
    f"{stats['Total'].sum():,}",
    f"{stats['Positives'].sum():,}",
    f"{100 * df['label_hard'].mean():.2f}%",
])

tbl = ax_table.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.55)

# Style header
for col in range(len(col_labels)):
    cell = tbl[0, col]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")

# Style data rows (alternating) + totals row
for row_i in range(1, len(table_data) + 1):
    is_total = row_i == len(table_data)
    for col_i in range(len(col_labels)):
        cell = tbl[row_i, col_i]
        if is_total:
            cell.set_facecolor("#D5D8DC")
            cell.set_text_props(fontweight="bold")
        elif row_i % 2 == 0:
            cell.set_facecolor("#EAF2FF")
        else:
            cell.set_facecolor("#FDFEFE")
        # Colour dataset-specific columns
        if col_i == 1:
            cell.set_text_props(color=COLORS["ptbxl"])
        elif col_i == 2:
            cell.set_text_props(color=COLORS["samitrop"])
        elif col_i == 3:
            cell.set_text_props(color=COLORS["code15"])
        elif col_i == 5 and not is_total:
            cell.set_text_props(color=POS_COLOR, fontweight="bold")

ax_table.set_title("Per-Fold Record Distribution by Dataset Source", fontsize=11, pad=6)

# ─── Panel 2: Stacked bar ────────────────────────────────────────────────────
fold_labels = [f"Fold {i}" for i in range(N_FOLDS)]
ptbxl_vals    = stats["ptbxl"].values
samitrop_vals = stats["samitrop"].values
code15_vals   = stats["code15"].values

x = np.arange(N_FOLDS)
bar_w = 0.55

b1 = ax_bar.bar(x, ptbxl_vals,    bar_w, label="PTB-XL",    color=COLORS["ptbxl"],    edgecolor="white")
b2 = ax_bar.bar(x, samitrop_vals, bar_w, bottom=ptbxl_vals, label="SaMi-Trop", color=COLORS["samitrop"], edgecolor="white")
b3 = ax_bar.bar(x, code15_vals,   bar_w,
                bottom=ptbxl_vals + samitrop_vals,
                label="CODE-15%", color=COLORS["code15"], edgecolor="white")

# Annotate SaMi-Trop counts (smallest, most important to show)
for i, (pv, sv) in enumerate(zip(ptbxl_vals, samitrop_vals)):
    ax_bar.text(i, pv + sv / 2, str(sv), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(fold_labels, fontsize=9)
ax_bar.set_ylabel("Record Count", fontsize=9)
ax_bar.set_title("Dataset Composition per Fold\n(SaMi-Trop count annotated)", fontsize=10)
ax_bar.legend(fontsize=8, loc="lower right")
ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_bar.set_ylim(0, stats["Total"].max() * 1.08)
ax_bar.spines[["top", "right"]].set_visible(False)

# ─── Panel 3: Positive counts per fold ──────────────────────────────────────
pos_vals    = stats["Positives"].values
mean_pos    = pos_vals.mean()

bars = ax_pos.bar(x, pos_vals, bar_w, color=POS_COLOR, alpha=0.82, edgecolor="white")
ax_pos.axhline(mean_pos, color="#2C3E50", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_pos:.0f}")

for bar, val in zip(bars, pos_vals):
    ax_pos.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

ax_pos.set_xticks(x)
ax_pos.set_xticklabels(fold_labels, fontsize=9)
ax_pos.set_ylabel("Chagas-Positive Records", fontsize=9)
ax_pos.set_title("Chagas-Positive Count per Fold\n(verifying class-balance preservation)", fontsize=10)
ax_pos.legend(fontsize=8)
ax_pos.set_ylim(0, pos_vals.max() * 1.18)
ax_pos.spines[["top", "right"]].set_visible(False)

# ─── Footer note ────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "Stratification key: dataset_source + hard_label  |  "
    "sklearn.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  |  "
    "Minor count variation across folds is due to integer rounding when 1/5 of a group is non-integer.",
    ha="center", va="bottom", fontsize=7.5, color="#555555",
    style="italic"
)

# ── Save ────────────────────────────────────────────────────────────────────
out = Path(OUT_PATH)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
