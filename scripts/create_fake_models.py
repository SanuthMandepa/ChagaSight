"""
create_fake_models.py
---------------------
Creates tiny placeholder .pt files for model_1d and model_2d in the models/ folder.

These are NOT real trained models — they let app.py start without crashing
and show "model not available" in the UI.

When you have real trained checkpoints:
  1. Delete the fake file
  2. Drop in the real .pt with the same filename
  3. Restart app.py — it will load automatically

Usage:
    python scripts/create_fake_models.py
"""

import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

PLACEHOLDERS = {
    "model_2d.pt": "2D ViT Classifier — replace with real fold checkpoint",
    "model_1d.pt": "1D ViT-FM Classifier — replace with real fold checkpoint",
}

for filename, description in PLACEHOLDERS.items():
    path = MODELS_DIR / filename
    if path.exists():
        print(f"[SKIP] {filename} already exists — delete it first if you want to regenerate")
        continue
    torch.save(
        {
            "placeholder": True,
            "description": description,
            "model_state_dict": {},
            "note": "Replace this file with the real trained checkpoint (.pt) of the same name.",
        },
        path,
    )
    size_kb = path.stat().st_size / 1024
    print(f"[OK]   Created {filename}  ({size_kb:.1f} KB)")

print("\nDone. models/ folder now contains:")
for p in sorted(MODELS_DIR.iterdir()):
    size = p.stat().st_size
    tag = " [PLACEHOLDER]" if size < 1_000_000 and p.suffix == ".pt" else ""
    print(f"  {p.name:40s}  {size / 1e6:.1f} MB{tag}")
