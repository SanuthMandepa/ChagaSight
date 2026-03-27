"""
Upload the 3 trained model files to HuggingFace Hub.

Usage:
    python scripts/upload_models_to_hf.py

Requires:
    pip install huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env var
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

REPO_ID = "sanuthmandepa/chagasight-models"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FILES_TO_UPLOAD = [
    "FINAL_ENSEMBLE_MODEL.pt",
    "model_1d.pt",
    "model_2d.pt",
]

def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi()

    for fname in FILES_TO_UPLOAD:
        local_path = MODELS_DIR / fname
        if not local_path.exists():
            print(f"[SKIP] {fname} not found at {local_path}")
            continue

        size_mb = local_path.stat().st_size / 1e6
        print(f"[UP]  Uploading {fname} ({size_mb:.0f} MB) → {REPO_ID} ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"[OK]  {fname} uploaded.")

    print("\nDone. All models uploaded to HuggingFace Hub.")
    print(f"View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
