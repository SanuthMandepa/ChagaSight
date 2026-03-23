# app.py — ChagaSight backend (Flask)
# Serves three model modes: 2D-only, 1D-only, Hybrid ensemble

import os
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT))

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
from src.preprocessing.baseline_removal import remove_baseline
from src.preprocessing.resample import resample_signal, pad_or_trim
from src.preprocessing.normalization import normalize_per_lead
from src.preprocessing.image_embedding import build_2d_image

# --------------------------------------------------
# Model backbones
# --------------------------------------------------
from src.models.vit_2d import ViT2D
from src.models.vit_1d_fm import ViT1D_FM
from src.models.hybrid_model import HybridChagasModel


class ViT2DClassifier(nn.Module):
    """2D ViT backbone + 2-layer MLP head (matches per_2d_fold.ipynb)."""
    def __init__(self):
        super().__init__()
        self.backbone = ViT2D(
            img_size=(24, 2048),
            patch_size=(8, 64),
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            use_aol=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class ViT1DClassifier(nn.Module):
    """1D ViT-FM backbone + 2-layer MLP head (matches per_1d_fold.ipynb)."""
    def __init__(self):
        super().__init__()
        self.backbone = ViT1D_FM(
            num_leads=12,
            seq_len=1000,
            patch_size=50,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            use_aol=True,
            use_demographics=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, sigs: torch.Tensor, ages: torch.Tensor, sexes: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(sigs, ages, sexes)).squeeze(-1)


# --------------------------------------------------
# Load all models at startup
# --------------------------------------------------
def _load(model: nn.Module, path: Path) -> nn.Module:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _is_placeholder(path: Path) -> bool:
    """Returns True if the file is a fake placeholder (not a real trained model)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return bool(ckpt.get("placeholder", False))
    except Exception:
        return False


model_2d = None
_path_2d = ROOT / "models" / "model_2d.pt"
if _path_2d.exists():
    if _is_placeholder(_path_2d):
        print(f"[2D]  placeholder file found — awaiting real checkpoint at {_path_2d}")
    else:
        try:
            model_2d = _load(ViT2DClassifier().to(DEVICE), _path_2d)
            print(f"[2D]  loaded {_path_2d.name}")
        except Exception as e:
            print(f"[2D]  failed to load {_path_2d.name}: {e}")
else:
    print(f"[2D]  checkpoint not found: {_path_2d}")

model_1d = None
_path_1d = ROOT / "models" / "model_1d.pt"
if _path_1d.exists():
    if _is_placeholder(_path_1d):
        print(f"[1D]  placeholder file found — awaiting real checkpoint at {_path_1d}")
    else:
        try:
            model_1d = _load(ViT1DClassifier().to(DEVICE), _path_1d)
            print(f"[1D]  loaded {_path_1d.name}")
        except Exception as e:
            print(f"[1D]  failed to load {_path_1d.name}: {e}")
else:
    print(f"[1D]  checkpoint not found: {_path_1d}")

hybrid_models: List[nn.Module] = []
hybrid_threshold = 0.2841  # fallback
_path_ensemble = ROOT / "models" / "FINAL_ENSEMBLE_MODEL.pt"
if _path_ensemble.exists():
    try:
        _ens_ckpt = torch.load(_path_ensemble, map_location=DEVICE, weights_only=False)
        _cfg = _ens_ckpt.get("model_config", {})
        hybrid_threshold = float(_ens_ckpt.get("threshold", hybrid_threshold))
        for _fm in _ens_ckpt["fold_models"]:
            _m = HybridChagasModel(**_cfg).to(DEVICE)
            _m.load_state_dict(_fm["model_state_dict"])
            _m.eval()
            hybrid_models.append(_m)
            print(f"[HYB] loaded fold {_fm['fold']} from FINAL_ENSEMBLE_MODEL.pt")
        print(f"[HYB] threshold from checkpoint: {hybrid_threshold:.6f}")
    except Exception as e:
        print(f"[HYB] failed to load FINAL_ENSEMBLE_MODEL.pt: {e}")
else:
    print(f"[HYB] ensemble checkpoint not found: {_path_ensemble}")

print(f"\nDevice: {DEVICE}")
print(f"Models ready — 2D: {'yes' if model_2d else 'no'} | "
      f"1D: {'yes' if model_1d else 'no'} | "
      f"Hybrid folds: {len(hybrid_models)}\n")

# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__)
_cors_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
CORS(app, resources={r"/api/*": {"origins": _cors_origins}})

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
        raise ValueError("Missing .hea file. Upload .hea + .dat/.mat with the same base name.")
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
# Preprocessing helpers
# --------------------------------------------------
def _preprocess_wfdb(record_base: Path, fs: float):
    """Returns (img_tensor, sig_tensor) ready for model inference."""
    signal, _ = wfdb.rdsamp(str(record_base))  # (T, 12)
    signal = signal.T.astype("float32")         # → (12, T) leads-first

    if signal.shape[0] != 12:
        raise ValueError(f"Expected 12-lead ECG, got shape {signal.shape}")

    signal = remove_baseline(signal, method="bandpass", fs=fs)

    # 2D path — 500 Hz contour image
    sig_500 = resample_signal(signal, fs, 500.0)
    sig_500 = normalize_per_lead(sig_500)
    img = build_2d_image(sig_500, target_width=2048)
    if img.shape != (3, 24, 2048):
        raise ValueError(f"Unexpected contour image shape: {img.shape}")
    img_t = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)  # (1,3,24,2048)

    # 1D path — 100 Hz, exactly 1000 samples
    sig_100 = resample_signal(signal, fs, 100.0)
    sig_100 = normalize_per_lead(sig_100)
    sig_100 = pad_or_trim(sig_100, 1000)
    sig_t = torch.from_numpy(sig_100).float().unsqueeze(0).to(DEVICE)  # (1,12,1000)

    return img_t, sig_t


def _parse_demographics(req):
    """Returns (age_centuries_tensor, sex_binary_tensor)."""
    age_years = float(req.form.get("age", 50))
    sex_str = req.form.get("sex", "unknown").strip().lower()
    ages = torch.tensor([age_years / 100.0], dtype=torch.float32, device=DEVICE)
    sexes = torch.tensor([1.0 if sex_str == "female" else 0.0],
                         dtype=torch.float32, device=DEVICE)
    return ages, sexes


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "2d": model_2d is not None,
            "1d": model_1d is not None,
            "hybrid": {"folds_loaded": len(hybrid_models), "threshold": hybrid_threshold},
        },
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "Upload WFDB files using form-data key = files"}), 400

    saved = _save_uploaded_files(request.files.getlist("files"))

    try:
        record_base, record_name = _find_record(saved)

        _, fields = wfdb.rdsamp(str(record_base))
        fs = float(fields.get("fs", 500.0))

        img_t, sig_t = _preprocess_wfdb(record_base, fs)
        ages, sexes = _parse_demographics(request)

        model_type = request.form.get("model_type", "hybrid").strip().lower()

        if model_type == "2d":
            if model_2d is None:
                return jsonify({"error": "2D model checkpoint not loaded on this server."}), 503
            with torch.no_grad():
                prob = torch.sigmoid(model_2d(img_t)).item()
            threshold = 0.5
            folds_used = 1

        elif model_type == "1d":
            if model_1d is None:
                return jsonify({"error": "1D model checkpoint not loaded on this server."}), 503
            with torch.no_grad():
                prob = torch.sigmoid(model_1d(sig_t, ages, sexes)).item()
            threshold = 0.5
            folds_used = 1

        else:  # hybrid (default)
            if not hybrid_models:
                return jsonify({"error": "No hybrid model checkpoints loaded on this server."}), 503
            with torch.no_grad():
                probs = [
                    torch.sigmoid(m(img_t, sig_t, ages, sexes)["logits"]).item()
                    for m in hybrid_models
                ]
            prob = float(sum(probs) / len(probs))
            threshold = hybrid_threshold
            folds_used = len(hybrid_models)

        pred = int(prob >= threshold)

        return jsonify({
            "record": record_name,
            "model_type": model_type if model_type in ("2d", "1d") else "hybrid",
            "probability": round(float(prob), 6),
            "threshold": threshold,
            "prediction": pred,
            "folds_used": folds_used,
            "interpretation": (
                "Positive for Chagas Disease" if pred else "Negative for Chagas Disease"
            ),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    finally:
        _cleanup(saved)


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
