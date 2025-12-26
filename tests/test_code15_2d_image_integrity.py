import sys
from pathlib import Path
import pandas as pd

# -------------------------------
# Configuration
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE15_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "2d_images" / "code15"
CODE15_META_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "code15_metadata.csv"

print("🔍 CODE-15 2D IMAGE INTEGRITY CHECK")
print("Image dir :", CODE15_IMG_DIR)
print("Metadata  :", CODE15_META_CSV)

# -------------------------------
# Load metadata
# -------------------------------
if not CODE15_META_CSV.exists():
    raise FileNotFoundError(f"Metadata CSV not found: {CODE15_META_CSV}")

df = pd.read_csv(CODE15_META_CSV)

if "img_path" not in df.columns:
    raise ValueError("Missing 'img_path' column in metadata CSV")

# Normalize paths (Windows-safe)
expected_files = set(
    Path(p).name for p in df["img_path"].astype(str)
)

# -------------------------------
# Load filesystem files
# -------------------------------
if not CODE15_IMG_DIR.exists():
    raise FileNotFoundError(f"Image directory not found: {CODE15_IMG_DIR}")

actual_files = set(
    p.name for p in CODE15_IMG_DIR.glob("*.npy")
)

# -------------------------------
# Compare
# -------------------------------
missing_images = expected_files - actual_files
orphan_images = actual_files - expected_files

print(f"\n📊 Summary")
print(f"Metadata entries : {len(expected_files)}")
print(f"Image files      : {len(actual_files)}")

# -------------------------------
# Report errors
# -------------------------------
if missing_images:
    print(f"\n❌ Missing images ({len(missing_images)}):")
    for name in list(missing_images)[:10]:
        print("  -", name)
    if len(missing_images) > 10:
        print(f"  ... and {len(missing_images) - 10} more")
    
    raise RuntimeError("Integrity check FAILED: missing 2D images")

if orphan_images:
    print(f"\n⚠️ Orphan images ({len(orphan_images)}):")
    for name in list(orphan_images)[:10]:
        print("  -", name)
    if len(orphan_images) > 10:
        print(f"  ... and {len(orphan_images) - 10} more")

print("\n✅ CODE-15 2D IMAGE INTEGRITY CHECK PASSED")
