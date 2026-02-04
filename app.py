# app.py — ChagaSight backend (Flask)

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import wfdb
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_PATH = ROOT / "models" / "production_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT))

# --------------------------------------------------
# Your preprocessing pipeline
# --------------------------------------------------
from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_ecg
from src.preprocessing.normalization import zscore_per_lead
from src.preprocessing.image_embedding import ecg_to_contour_image

# --------------------------------------------------
# Model — MUST match training
# --------------------------------------------------
class ViTClassifier(nn.Module):
    def __init__(self, embed_dim=384, heads=6, mlp_ratio=4, dropout=0.1, depth=8):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=(8, 16), stride=(8, 16), bias=False
        )

        num_patches = (24 // 8) * (2048 // 16)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                embed_dim,
                heads,
                int(embed_dim * mlp_ratio),
                dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Matches checkpoint keys head.1.weight / head.1.bias
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x[:, 0])
        return self.head(x).squeeze(-1)

# --------------------------------------------------
# Load model
# --------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = ViTClassifier(
    embed_dim=checkpoint.get("config", {}).get("embed_dim", 384),
    heads=checkpoint.get("config", {}).get("heads", 6),
    depth=checkpoint.get("config", {}).get("depth", 8),
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --------------------------------------------------
# Flask app + CORS (FIXES "Failed to fetch")
# --------------------------------------------------
app = Flask(__name__)

# Allow your Vite dev server origins
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}},
)

ALLOWED_EXTENSIONS = {"hea", "dat", "mat"}

def _allowed(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def _save_uploaded_files(files) -> List[Path]:
    paths: List[Path] = []
    for f in files:
        if f.filename and _allowed(f.filename):
            p = UPLOAD_DIR / secure_filename(f.filename)
            f.save(str(p))
            paths.append(p)
    return paths

def _find_record(saved: List[Path]) -> Tuple[Path, str]:
    hea = [p for p in saved if p.suffix.lower() == ".hea"]
    if not hea:
        raise ValueError("Missing .hea file. Upload .hea + .dat/.mat with same base name.")

    name = hea[0].stem
    base = UPLOAD_DIR / name

    if not ((base.with_suffix(".dat")).exists() or (base.with_suffix(".mat")).exists()):
        raise ValueError("Missing matching .dat or .mat file (same base name as .hea).")

    return base, name

def _cleanup(paths: List[Path]) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE), "model_path": str(MODEL_PATH)})

@app.route("/api/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "Upload WFDB files using form-data key = files"}), 400

    saved = _save_uploaded_files(request.files.getlist("files"))

    try:
        record_base, record_name = _find_record(saved)

        # WFDB load (older wfdb versions: no physical= arg)
        signal, fields = wfdb.rdsamp(str(record_base))

        if signal.ndim != 2:
            raise ValueError(f"Invalid ECG shape from WFDB reader: {signal.shape}")

        if signal.shape[1] != 12:
            raise ValueError(f"Expected 12-lead ECG, got shape {signal.shape}")

        fs = float(fields.get("fs", 500.0))

        # Preprocess
        signal = remove_baseline(signal, method="bandpass", fs=fs)
        signal, _ = resample_ecg(signal, fs, 500)
        signal = zscore_per_lead(signal)

        # 2D embedding
        img = ecg_to_contour_image(signal, target_width=2048)
        if img.shape != (3, 24, 2048):
            raise ValueError(f"Invalid image shape {img.shape}. Expected (3,24,2048).")

        x = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        threshold = float(checkpoint.get("threshold", 0.5))
        pred = int(prob >= threshold)

        return jsonify({
            "record": record_name,
            "probability": float(prob),
            "threshold": float(threshold),
            "prediction": int(pred),
            "interpretation": "Positive for Chagas Disease" if pred else "Negative for Chagas Disease",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    finally:
        _cleanup(saved)

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    # Use 0.0.0.0 so both localhost and 127.0.0.1 behave consistently
    app.run(host="0.0.0.0", port=5050, debug=True)
