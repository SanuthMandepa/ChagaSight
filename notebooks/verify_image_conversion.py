"""
Verify ECG → 2D structured image conversion.
This version SAVES all plots as PNG (works on Windows, no GUI needed).

Checks:
1. 1D processed ECG (lead I)
2. 3-channel structured image (3×24×2048)
3. Raw WFDB PTB-XL signal (optional)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import wfdb

OUTPUT_DIR = "notebooks/verification_outputs"


# ------------------------------------------------------------
# Ensure output directory exists
# ------------------------------------------------------------
def ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Save processed 1D signal plot
# ------------------------------------------------------------
def save_signal_plot(signal):
    plt.figure(figsize=(12, 3))
    plt.plot(signal[:, 0])
    plt.title("1D Processed ECG – Lead I")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/signal.png")
    plt.close()
    print("✔ Saved 1D signal → notebooks/verification_outputs/signal.png")


# ------------------------------------------------------------
# Save each of the 3 image channels
# ------------------------------------------------------------
def save_image_plot(image):
    for i in range(3):
        plt.figure(figsize=(12, 3))
        plt.imshow(image[i], aspect="auto", cmap="gray")
        plt.title(f"Image Channel {i+1}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/image_ch{i+1}.png")
        plt.close()
        print(f"✔ Saved image channel {i+1} → notebooks/verification_outputs/image_ch{i+1}.png")


# ------------------------------------------------------------
# Save raw PTB-XL WFDB signal
# ------------------------------------------------------------
def save_raw_plot(raw_path):
    try:
        # IMPORTANT: rdsamp RETURNS TWO VALUES
        record, meta = wfdb.rdsamp(raw_path)

        raw_sig = record.p_signal

        plt.figure(figsize=(12, 3))
        plt.plot(raw_sig[:, 0])
        plt.title("Raw WFDB ECG – Lead I")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/raw_signal.png")
        plt.close()

        print("✔ Saved RAW WFDB signal → notebooks/verification_outputs/raw_signal.png")

    except Exception as e:
        print("⚠️ Could not load RAW WFDB file:", e)


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
def main(signal_path, image_path, raw_path):
    ensure_outdir()

    # -------- Load 1D Processed Signal --------
    print("\n=== Loading 1D Processed Signal ===")
    signal = np.load(signal_path)
    print("Shape:", signal.shape)
    print("Range:", signal.min(), "→", signal.max())
    save_signal_plot(signal)

    # -------- Load 2D Image --------
    print("\n=== Loading 2D Structured Image ===")
    image = np.load(image_path)
    print("Shape:", image.shape)
    print("Range:", image.min(), "→", image.max())
    save_image_plot(image)

    # -------- Load RAW PTB-XL WFDB --------
    if raw_path:
        print("\n=== Comparing with RAW WFDB ===")
        save_raw_plot(raw_path)

    print("\n🎉 Verification complete! Check output folder:")
    print("📁 notebooks/verification_outputs")


# ------------------------------------------------------------
# CLI ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--signal", required=True, help="Path to 1D processed ECG .npy file")
    parser.add_argument("--image", required=True, help="Path to 2D ECG image .npy file")
    parser.add_argument("--raw", help="Optional: path to PTB-XL WFDB base file (NO .dat/.hea)")

    args = parser.parse_args()

    main(args.signal, args.image, args.raw)
