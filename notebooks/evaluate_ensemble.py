#!/usr/bin/env python
"""
ChagaSight -- Final Ensemble Evaluation Script
===============================================
Converted from evaluation_FINAL_FAST.ipynb.

Run from the ChagaSight project root OR from notebooks/:
    python evaluate_ensemble.py

Key differences vs notebook:
  * project_root derived from __file__, not Path.cwd().parent
  * Entire script in if __name__ == "__main__" (required for
    num_workers > 0 on Windows -- avoids multiprocessing spawn crashes)
  * SubsetSampler resume: zero disk reads for already-done batches
  * plt.show() removed -- saves PNGs directly, no GUI window
"""

import sys, warnings, time, platform
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")   # non-interactive -- saves to disk, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── project_root: correct regardless of where you run from ──────────────────
# If script is in  ChagaSight/notebooks/evaluate_ensemble.py  -> .parent.parent
# If script is in  ChagaSight/evaluate_ensemble.py            -> .parent
_HERE = Path(__file__).resolve().parent
project_root = _HERE.parent if _HERE.name == "notebooks" else _HERE


def main():
    # ── sys.path ──────────────────────────────────────────────────────────────
    for p in [str(project_root),
              str(project_root / "external" / "official_2025")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # ── Official PhysioNet metric ─────────────────────────────────────────────
    OFFICIAL = False
    try:
        from helper_code import compute_challenge_score, compute_auc as _compute_auc
        OFFICIAL = True
        print("Official PhysioNet metric: ENABLED")
    except ImportError:
        _compute_auc = None
        print("Official metric not found -- using sklearn ROC approximation")

    from src.models.hybrid_model import HybridChagasModel
    from src.training.dataset import create_dataloaders

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM   : {gpu_gb:.1f} GB")
    print(f"Python : {sys.version.split()[0]}  |  PyTorch: {torch.__version__}")
    print(f"OS     : {platform.system()}  |  Root: {project_root}")

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    INFERENCE_BATCH = 32      # 32 for 6 GB GPU; try 64 for >=8 GB
    NUM_WORKERS     = 2 if platform.system() == "Windows" else 4
    SAVE_EVERY_N    = 50      # crash-recovery checkpoint every N batches
    N_PERMS_FINAL   = 10000   # official TPR@5% permutations (final)
    N_PERMS_FOLD    = 5000    # permutations for per-fold / per-dataset
    N_BOOTSTRAP     = 1000    # bootstrap resamples for 95% CI
    SEED            = 12345

    CHECKPOINT_DIR  = project_root / "checkpoints"
    DATA_DIR        = project_root / "data" / "processed"
    METADATA_CSV    = DATA_DIR / "metadata" / "combined_5fold.csv"
    IMAGES_DIR      = DATA_DIR / "2d_images"
    SIGNALS_DIR     = DATA_DIR / "1d_signals_100hz"
    FIGURES_DIR     = CHECKPOINT_DIR / "thesis_figures"
    EVAL_CKPT_DIR   = CHECKPOINT_DIR / "evaluation_checkpoints"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_CKPT_DIR.mkdir(exist_ok=True)

    BENCHMARKS = [
        ("Random baseline",                       0.050),
        ("No-pretrain baseline (expected)",        0.300),
        ("Kim et al. 2025 (2D-only approach)",     0.369),
        ("PhysioNet challenge target",             0.420),
        ("Van Santvliet 2025 top team (val set)",  0.445),
        ("Van Santvliet 2025 CV mean",             0.490),
    ]

    # Auto-detect available fold checkpoints
    fold_ckpts, available_folds = [], []
    print("\nFold checkpoint status:")
    for fold in range(5):
        p = CHECKPOINT_DIR / f"fold{fold}_best.pt"
        if p.exists():
            fold_ckpts.append(p)
            available_folds.append(fold)
            print(f"  [OK]      fold{fold}_best.pt   {p.stat().st_size/1e6:.0f} MB")
        else:
            print(f"  [MISSING] fold{fold}_best.pt")

    if not available_folds:
        raise FileNotFoundError("No fold checkpoints found. Train at least one fold first.")
    if len(available_folds) < 5:
        print(f"\nWARNING: Only {len(available_folds)}/5 folds. Ensemble will be sub-optimal.")
    else:
        print("\nAll 5 folds found.")

    print(f"\nInference batch : {INFERENCE_BATCH}  |  Workers: {NUM_WORKERS}"
          f"  |  AMP: {device == 'cuda'}")

    # =========================================================================
    # LOAD MODELS
    # =========================================================================
    MODEL_CFG = dict(
        img_size=(24, 2048), patch_size_2d=(8, 64),
        num_leads=12, seq_len_1d=1000, patch_size_1d=50,
        embed_dim=768, depth=12, num_heads=12,
        use_aol=True, use_demographics=True,
    )

    print(f"\nLoading {len(available_folds)} model(s)...\n")
    models, fold_val_scores = [], []
    for fold_idx, ckpt_path in zip(available_folds, fold_ckpts):
        m = HybridChagasModel(**MODEL_CFG)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        models.append(m)
        vs = ckpt.get("val_score", None)
        fold_val_scores.append(vs)
        print(f"  Fold {fold_idx}: val_score={'n/a' if vs is None else f'{vs:.4f}'}  "
              f"phase={ckpt.get('phase','?')}  epoch={ckpt.get('epoch','?')}")

    total_params = sum(p.numel() for p in models[0].parameters())
    print(f"\nModel : HybridChagasModel  |  {total_params:,} params/fold")
    print(f"VRAM  : {total_params*4*len(models)/1024**3:.1f} GB est for {len(models)} fp32 models")
    valid = [s for s in fold_val_scores if s is not None]
    if valid:
        print(f"Fold val scores : mean={np.mean(valid):.4f}  std={np.std(valid):.4f}")

    # =========================================================================
    # ENSEMBLE INFERENCE  (crash-safe, resumable, fast Subset skip)
    # =========================================================================
    print("\nRunning ensemble inference on all 5 validation sets...")
    print("Each fold val set evaluated with ALL available models (unbiased).\n")

    if device == "cuda":
        torch.cuda.empty_cache()

    all_probs, all_labels, all_ids, all_datasets, all_folds_list = [], [], [], [], []
    total_start = time.time()

    for fold_idx, _ in zip(available_folds, fold_ckpts):
        fold_done    = EVAL_CKPT_DIR / f"fold{fold_idx}_complete.npz"
        fold_partial = EVAL_CKPT_DIR / f"fold{fold_idx}_partial.npz"

        # ── Load from cache if already finished ───────────────────────────────
        if fold_done.exists():
            d = np.load(fold_done, allow_pickle=True)
            all_probs.extend(d["probs"].tolist())
            all_labels.extend(d["labels"].tolist())
            all_ids.extend(d["ids"].tolist())
            all_datasets.extend(d["datasets"].tolist())
            all_folds_list.extend([fold_idx] * len(d["labels"]))
            print(f"Fold {fold_idx}: loaded from cache  ({len(d['labels']):,} samples)")
            continue

        print(f"Fold {fold_idx}: running inference ...")
        fold_start = time.time()

        _, val_loader_full = create_dataloaders(
            metadata_csv=str(METADATA_CSV),
            images_dir=str(IMAGES_DIR),
            signals_dir=str(SIGNALS_DIR),
            fold=fold_idx,
            batch_size=INFERENCE_BATCH,
            num_workers=NUM_WORKERS,
            use_weighted_sampling=False,
            augment_train=False,
        )
        val_dataset = val_loader_full.dataset
        n_total     = len(val_dataset)
        n_total_batches = -(-n_total // INFERENCE_BATCH)   # ceil

        # ── Resume from partial checkpoint ────────────────────────────────────
        fp, fl, fi, fd = [], [], [], []
        start_sample = 0
        start_batch  = 0

        if fold_partial.exists():
            d = np.load(fold_partial, allow_pickle=True)
            fp           = d["probs"].tolist()
            fl           = d["labels"].tolist()
            fi           = d["ids"].tolist()
            fd           = d["datasets"].tolist()
            last_batch   = int(d["last_batch"])
            start_sample = (last_batch + 1) * INFERENCE_BATCH
            start_batch  = last_batch + 1
            print(f"  Resuming from sample {start_sample:,}  "
                  f"(batch {start_batch}, {len(fp):,} results already saved)")

        # ── Subset skip: ZERO disk reads for already-done samples ─────────────
        if start_sample > 0:
            print(f"  Skipping {start_sample:,} samples via Subset (zero disk reads).")
            val_loader = DataLoader(
                Subset(val_dataset, list(range(start_sample, n_total))),
                batch_size=INFERENCE_BATCH,
                num_workers=NUM_WORKERS,
                shuffle=False,
                pin_memory=(device == "cuda"),
                drop_last=False,
            )
        else:
            val_loader = val_loader_full

        use_amp = (device == "cuda")
        pbar = tqdm(val_loader, desc=f"Fold {fold_idx}",
                    initial=start_batch, total=n_total_batches, leave=True)

        with torch.no_grad():
            for bi_rel, batch in enumerate(pbar):
                bi = start_batch + bi_rel

                imgs  = batch["image"].to(device, non_blocking=True)
                sigs  = batch["signal"].to(device, non_blocking=True)
                ages  = batch["age"].to(device, non_blocking=True)
                sexes = batch["sex"].to(device, non_blocking=True)
                hlab  = batch["hard_label"].numpy()   # binary {0,1}

                preds = []
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    for m in models:
                        out = m(imgs, sigs, ages, sexes)
                        preds.append(torch.sigmoid(out["logits"]).float().flatten())
                # Stack on GPU → single batched transfer → mean on CPU
                ens = torch.stack(preds).mean(dim=0).cpu().numpy()

                fp.extend(ens.tolist())
                fl.extend(hlab.tolist())
                fi.extend(batch["id"])
                fd.extend(batch["dataset"])

                if (bi + 1) % SAVE_EVERY_N == 0:
                    np.savez(fold_partial,
                             probs=np.array(fp,  dtype=np.float32),
                             labels=np.array(fl, dtype=np.float32),
                             ids=fi, datasets=fd, last_batch=bi)

        # ── Save completed fold ───────────────────────────────────────────────
        fold_probs  = np.array(fp,  dtype=np.float32)
        fold_labels = np.array(fl,  dtype=np.float32)
        np.savez(fold_done, probs=fold_probs, labels=fold_labels, ids=fi, datasets=fd)
        if fold_partial.exists():
            fold_partial.unlink()

        all_probs.extend(fold_probs.tolist())
        all_labels.extend(fold_labels.tolist())
        all_ids.extend(fi)
        all_datasets.extend(fd)
        all_folds_list.extend([fold_idx] * len(fold_labels))

        elapsed = time.time() - fold_start
        print(f"  Done: {len(fold_labels):,} samples | "
              f"{int(fold_labels.sum())} pos | {elapsed/60:.1f} min")
        if device == "cuda":
            torch.cuda.empty_cache()

            # ── GPU cooldown: prevent thermal throttling between folds ────────
            # Laptop GPUs (45W TDP) throttle clocks after sustained load,
            # causing 2-3× slowdown on subsequent folds.
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    gpu_temp = int(parts[0])
                    gpu_clock = parts[1] if len(parts) > 1 else "?"
                    print(f"  GPU temp: {gpu_temp}°C  |  SM clock: {gpu_clock} MHz")
                    if gpu_temp > 75:
                        cool_secs = 90  # hot GPU needs longer cooldown
                    elif gpu_temp > 65:
                        cool_secs = 60
                    else:
                        cool_secs = 30  # already cool, short pause
                else:
                    cool_secs = 60
                    gpu_temp = None
            except Exception:
                cool_secs = 60
                gpu_temp = None

            remaining_folds = len(available_folds) - (available_folds.index(fold_idx) + 1)
            if remaining_folds > 0:
                print(f"  Cooling pause: {cool_secs}s before next fold "
                      f"({remaining_folds} fold(s) remaining)...")
                time.sleep(cool_secs)
                torch.cuda.empty_cache()  # second cleanup after cooldown

    all_probs     = np.array(all_probs,      dtype=np.float64)
    all_labels    = np.array(all_labels,     dtype=np.float64)
    all_folds_arr = np.array(all_folds_list, dtype=np.int32)

    assert not np.any(np.isnan(all_probs)),  "NaN in predictions"
    assert not np.any(np.isnan(all_labels)), "NaN in labels"
    assert set(np.unique(all_labels)) <= {0.0, 1.0}, "Labels not binary {0,1}"

    total_elapsed = time.time() - total_start
    print(f"\nInference complete: {len(all_labels):,} samples | "
          f"{int(all_labels.sum())} pos ({100*all_labels.mean():.2f}%) | "
          f"{total_elapsed/60:.1f} min total")

    # =========================================================================
    # PRIMARY METRICS
    # =========================================================================
    np.random.seed(SEED)

    if OFFICIAL:
        print(f"\nComputing official TPR@5% ({N_PERMS_FINAL:,} permutations)...")
        tpr_5pct = float(compute_challenge_score(
            all_labels, all_probs,
            fraction_capacity=0.05, num_permutations=N_PERMS_FINAL, seed=SEED,
        ))
        auroc_v, auprc_v = _compute_auc(all_labels, all_probs)
        auroc = float(auroc_v); auprc = float(auprc_v)
        method_tag = f"OFFICIAL ({N_PERMS_FINAL:,} perms)"
    else:
        fpr_, tpr_, _ = roc_curve(all_labels, all_probs)
        idx5     = np.where(fpr_ <= 0.05)[0]
        tpr_5pct = float(tpr_[idx5[-1]]) if len(idx5) > 0 else 0.0
        auroc    = float(roc_auc_score(all_labels, all_probs))
        auprc    = float(average_precision_score(all_labels, all_probs))
        method_tag = "sklearn approx (~4 pp above official)"

    print("=" * 65)
    print("  FINAL ENSEMBLE RESULTS")
    print("=" * 65)
    print(f"  TPR@5%:  {tpr_5pct:.4f}   [{method_tag}]")
    print(f"  AUROC:   {auroc:.4f}  |  AUPRC:  {auprc:.4f}")
    print(f"  Folds:   {available_folds}")
    print("=" * 65)

    print("\nBenchmark comparison:")
    for name, val in BENCHMARKS:
        diff  = tpr_5pct - val
        arrow = "^" if diff >= 0 else "v"
        print(f"  {arrow} {abs(diff):.4f}  vs  {name:<44} ({val:.3f})")

    N_total  = len(all_labels)
    n_pos    = int(all_labels.sum())
    capacity = int(0.05 * N_total)
    found    = int(round(tpr_5pct * n_pos))
    rand_f   = max(1, int(round(0.05 * n_pos)))
    nns      = round(capacity / found, 1) if found > 0 else float("inf")

    print(f"\nClinical interpretation:")
    print(f"  Total patients          : {N_total:,}")
    print(f"  Chagas-positive         : {n_pos:,}  ({100*n_pos/N_total:.2f}%)")
    print(f"  Screening capacity (5%) : {capacity:,}")
    print(f"  Cases found by model    : {found} / {n_pos}  ({100*tpr_5pct:.1f}%)")
    print(f"  Improvement over random : {found/rand_f:.1f}x  |  NNS: {nns}")

    # =========================================================================
    # THRESHOLD ANALYSIS
    # =========================================================================
    fpr_arr, tpr_arr, roc_thr = roc_curve(all_labels, all_probs)
    prec_arr, rec_arr, pr_thr  = precision_recall_curve(all_labels, all_probs)

    thr_youden = float(roc_thr[np.argmax(tpr_arr - fpr_arr)])
    f1_arr = 2*prec_arr[:-1]*rec_arr[:-1] / (prec_arr[:-1]+rec_arr[:-1]+1e-9)
    thr_f1 = float(pr_thr[np.argmax(f1_arr)])

    results_thr = {}
    for name, thr in [("default_0.5", 0.5), ("youden_j", thr_youden), ("optimal_f1", thr_f1)]:
        pred = (all_probs >= thr).astype(int)
        tn, fp_c, fn, tp = confusion_matrix(all_labels, pred).ravel()
        spec = tn/(tn+fp_c) if (tn+fp_c) > 0 else 0.0
        npv  = tn/(tn+fn)   if (tn+fn)   > 0 else 0.0
        results_thr[name] = dict(
            threshold   = round(thr, 4),
            TP=int(tp), TN=int(tn), FP=int(fp_c), FN=int(fn),
            sensitivity = round(float(recall_score(all_labels, pred)), 4),
            specificity = round(spec, 4),
            precision   = round(float(precision_score(all_labels, pred, zero_division=0)), 4),
            npv         = round(npv, 4),
            f1          = round(float(f1_score(all_labels, pred, zero_division=0)), 4),
            mcc         = round(float(matthews_corrcoef(all_labels, pred)), 4),
            accuracy    = round(float(accuracy_score(all_labels, pred)), 4),
        )

    df_thr = pd.DataFrame(results_thr).T
    cols   = ["threshold","sensitivity","specificity","precision",
              "npv","f1","mcc","accuracy"]
    print("\nThreshold analysis:")
    print(df_thr[cols].to_string())

    primary     = results_thr["youden_j"]
    pred_binary = (all_probs >= primary["threshold"]).astype(int)
    print(f"\nPrimary (Youden J, thr={primary['threshold']}):")
    for k in ["sensitivity","specificity","precision","npv","f1","mcc","accuracy"]:
        print(f"  {k:<14}: {primary[k]:.4f}")
    print(f"  Confusion  TP={primary['TP']}  TN={primary['TN']:,}  "
          f"FP={primary['FP']:,}  FN={primary['FN']}")

    # =========================================================================
    # BOOTSTRAP 95% CI
    # =========================================================================
    print(f"\nBootstrap CI ({N_BOOTSTRAP} resamples)...")
    np.random.seed(SEED)
    n = len(all_labels)
    bt_tpr, bt_auroc, bt_auprc = [], [], []
    for _ in tqdm(range(N_BOOTSTRAP), desc="Bootstrap", leave=False):
        idx = np.random.choice(n, n, replace=True)
        lbl, prb = all_labels[idx], all_probs[idx]
        if lbl.sum() < 2 or (lbl == 0).sum() < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(lbl, prb)
        i5b = np.where(fpr_b <= 0.05)[0]
        bt_tpr.append(float(tpr_b[i5b[-1]]) if len(i5b) > 0 else 0.0)
        bt_auroc.append(float(roc_auc_score(lbl, prb)))
        bt_auprc.append(float(average_precision_score(lbl, prb)))

    def ci95(arr):
        a = np.array(arr)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    tpr_lo, tpr_hi     = ci95(bt_tpr)
    auroc_lo, auroc_hi = ci95(bt_auroc)
    auprc_lo, auprc_hi = ci95(bt_auprc)
    print(f"  TPR@5%: {tpr_5pct:.4f}  [{tpr_lo:.4f} - {tpr_hi:.4f}]")
    print(f"  AUROC : {auroc:.4f}  [{auroc_lo:.4f} - {auroc_hi:.4f}]")
    print(f"  AUPRC : {auprc:.4f}  [{auprc_lo:.4f} - {auprc_hi:.4f}]")

    # =========================================================================
    # PER-DATASET ANALYSIS
    # =========================================================================
    print("\nPer-dataset analysis:")
    all_datasets_arr = np.array(all_datasets)
    ds_rows = []
    for ds_name in ["ptbxl", "samitrop", "code15"]:
        mask = all_datasets_arr == ds_name
        if not mask.any():
            continue
        dl, dp = all_labels[mask], all_probs[mask]
        row = dict(dataset=ds_name.upper(), n_total=int(mask.sum()), n_pos=int(dl.sum()))
        if len(np.unique(dl)) < 2:
            note = "all positive" if dl.mean() == 1.0 else "all negative"
            row.update(tpr_5pct="n/a", auroc="n/a", auprc="n/a")
            print(f"  {ds_name.upper()}: {row['n_total']:,} -- {note}, metrics undefined")
        else:
            if OFFICIAL:
                ds_tpr = float(compute_challenge_score(
                    dl.astype(np.float64), dp.astype(np.float64),
                    fraction_capacity=0.05, num_permutations=N_PERMS_FOLD, seed=SEED,
                ))
                ds_auroc, ds_auprc = _compute_auc(dl, dp)
            else:
                fpr_d, tpr_d, _ = roc_curve(dl, dp)
                i5d      = np.where(fpr_d <= 0.05)[0]
                ds_tpr   = float(tpr_d[i5d[-1]]) if len(i5d) > 0 else 0.0
                ds_auroc = float(roc_auc_score(dl, dp))
                ds_auprc = float(average_precision_score(dl, dp))
            row.update(tpr_5pct=round(ds_tpr, 4),
                       auroc=round(float(ds_auroc), 4),
                       auprc=round(float(ds_auprc), 4))
            print(f"  {ds_name.upper()}: {row['n_total']:,} | "
                  f"TPR@5%={ds_tpr:.4f}  AUROC={float(ds_auroc):.4f}  "
                  f"AUPRC={float(ds_auprc):.4f}")
        ds_rows.append(row)

    df_ds = pd.DataFrame(ds_rows)
    df_ds.to_csv(CHECKPOINT_DIR / "per_dataset_metrics.csv", index=False)

    # =========================================================================
    # PER-FOLD TABLE
    # =========================================================================
    fold_rows = []
    for fold_idx in available_folds:
        mask = all_folds_arr == fold_idx
        fl, fp2 = all_labels[mask], all_probs[mask]
        row = dict(fold=fold_idx, n_total=int(mask.sum()), n_pos=int(fl.sum()))
        if len(np.unique(fl)) < 2:
            row.update(tpr_5pct="n/a", auroc="n/a", auprc="n/a")
        else:
            if OFFICIAL:
                ft = float(compute_challenge_score(
                    fl.astype(np.float64), fp2.astype(np.float64),
                    fraction_capacity=0.05, num_permutations=N_PERMS_FOLD, seed=SEED,
                ))
                fa, fp3 = _compute_auc(fl, fp2)
                fa, fp3 = float(fa), float(fp3)
            else:
                fpr_f, tpr_f, _ = roc_curve(fl, fp2)
                i5f = np.where(fpr_f <= 0.05)[0]
                ft  = float(tpr_f[i5f[-1]]) if len(i5f) > 0 else 0.0
                fa  = float(roc_auc_score(fl, fp2))
                fp3 = float(average_precision_score(fl, fp2))
            row.update(tpr_5pct=round(ft, 4), auroc=round(fa, 4), auprc=round(fp3, 4))
        fold_rows.append(row)

    fold_rows.append(dict(
        fold="Ensemble", n_total=len(all_labels), n_pos=int(all_labels.sum()),
        tpr_5pct=round(tpr_5pct, 4), auroc=round(auroc, 4), auprc=round(auprc, 4),
    ))
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(CHECKPOINT_DIR / "per_fold_metrics.csv", index=False)
    print("\nPer-fold results:")
    print(df_folds.to_string(index=False))

    numeric_tpr = [r["tpr_5pct"] for r in fold_rows[:-1] if isinstance(r["tpr_5pct"], float)]
    if len(numeric_tpr) == 5:
        print(f"\nFold mean +/- std : {np.mean(numeric_tpr):.4f} +/- {np.std(numeric_tpr):.4f}")
        print(f"Ensemble gain     : +{tpr_5pct - np.mean(numeric_tpr):.4f}")

    # =========================================================================
    # THESIS FIGURES  (6 x 300-dpi PNGs, saved to checkpoints/thesis_figures/)
    # =========================================================================
    print("\nGenerating thesis figures...")
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    def save_fig(name):
        path = FIGURES_DIR / name
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: thesis_figures/{name}")

    i5 = np.argmin(np.abs(fpr_arr - 0.05))

    # Fig 4.1 -- ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, lw=2.5, color="#2E86AB",
            label=f"ChagaSight ensemble  AUC={auroc:.3f}  [{auroc_lo:.3f}-{auroc_hi:.3f}]")
    ax.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.5,label="Random classifier")
    ax.plot(fpr_arr[i5], tpr_arr[i5], "ro", ms=10, zorder=5,
            label=f"5% FPR  TPR={tpr_arr[i5]:.3f}")
    ax.axvline(0.05, color="grey", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 4.1 -- ROC Curve"); ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    plt.tight_layout(); save_fig("fig4_1_roc_curve.png")

    # Fig 4.2 -- Precision-Recall
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_arr, prec_arr, lw=2.5, color="#A23B72",
            label=f"ChagaSight ensemble  AP={auprc:.3f}  [{auprc_lo:.3f}-{auprc_hi:.3f}]")
    baseline_prec = all_labels.mean()
    ax.axhline(baseline_prec, color="k", ls="--", lw=1.2, alpha=0.5,
               label=f"Random ({baseline_prec:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Figure 4.2 -- Precision-Recall Curve"); ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    plt.tight_layout(); save_fig("fig4_2_pr_curve.png")

    # Fig 4.3 -- Confusion Matrix
    cm = confusion_matrix(all_labels, pred_binary)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Neg","Pred Pos"],
                yticklabels=["True Neg","True Pos"],
                annot_kws={"size":14,"weight":"bold"}, ax=ax)
    total_cm = cm.sum()
    for ri in range(2):
        for ci in range(2):
            ax.text(ci+0.5, ri+0.72, f"({100*cm[ri,ci]/total_cm:.1f}%)",
                    ha="center", va="center", fontsize=10, color="dimgrey")
    ax.set_title(f"Figure 4.3 -- Confusion Matrix  (thr={primary['threshold']:.4f}, Youden J)")
    plt.tight_layout(); save_fig("fig4_3_confusion_matrix.png")

    # Fig 4.4 -- Probability Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_probs[all_labels==0], bins=60, alpha=0.6, density=True,
            color="steelblue", label=f"Negative  n={int((all_labels==0).sum()):,}")
    ax.hist(all_probs[all_labels==1], bins=60, alpha=0.6, density=True,
            color="crimson",  label=f"Positive  n={int(all_labels.sum()):,}")
    ax.axvline(primary["threshold"], color="k", ls="--", lw=1.5,
               label=f"Threshold {primary['threshold']:.3f}")
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.set_title("Figure 4.4 -- Predicted Probability Distribution by True Class")
    ax.legend(); plt.tight_layout(); save_fig("fig4_4_prob_histogram.png")

    # Fig 4.5 -- Calibration
    n_bins, bin_edges = 10, np.linspace(0, 1, 11)
    bc, bt = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        msk = (all_probs >= lo) & (all_probs < hi if i < n_bins-1 else all_probs <= hi)
        if msk.sum() > 0:
            bc.append((lo+hi)/2)
            bt.append(all_labels[msk].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.5,label="Perfect calibration")
    ax.plot(bc, bt, "o-", lw=2, ms=7, color="#F18F01", label="Ensemble")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Figure 4.5 -- Calibration Curve"); ax.legend()
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    plt.tight_layout(); save_fig("fig4_5_calibration.png")

    # Fig 4.6 -- Per-fold bar chart
    numeric_rows = [(r["fold"], r["tpr_5pct"])
                    for r in fold_rows if isinstance(r["tpr_5pct"], float)]
    if numeric_rows:
        labels_bar = [str(f) for f, _ in numeric_rows]
        vals_bar   = [v for _, v in numeric_rows]
        colors_bar = ["#2E86AB"]*(len(labels_bar)-1) + ["#E84855"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels_bar, vals_bar, color=colors_bar,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals_bar):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(0.420, color="orange", ls="--", lw=1.5,
                   label="Challenge target (0.420)")
        ax.set_xlabel("Fold / Ensemble"); ax.set_ylabel("TPR @ 5% FPR")
        ax.set_title("Figure 4.6 -- Per-Fold and Ensemble TPR@5%"); ax.legend()
        plt.tight_layout(); save_fig("fig4_6_per_fold_bar.png")

    print("All figures saved.")

    # =========================================================================
    # SAVE RESULTS + PACKAGE FINAL_ENSEMBLE_MODEL.pt
    # =========================================================================
    summary = {
        "TPR @ 5% FPR (primary)":  f"{tpr_5pct:.4f}  [{tpr_lo:.4f}-{tpr_hi:.4f}]",
        "AUROC":                    f"{auroc:.4f}  [{auroc_lo:.4f}-{auroc_hi:.4f}]",
        "AUPRC":                    f"{auprc:.4f}  [{auprc_lo:.4f}-{auprc_hi:.4f}]",
        "Sensitivity (recall)":     f"{primary['sensitivity']:.4f}",
        "Specificity":              f"{primary['specificity']:.4f}",
        "Precision (PPV)":          f"{primary['precision']:.4f}",
        "NPV":                      f"{primary['npv']:.4f}",
        "F1 Score":                 f"{primary['f1']:.4f}",
        "MCC":                      f"{primary['mcc']:.4f}",
        "Accuracy":                 f"{primary['accuracy']:.4f}",
        "Optimal threshold":        f"{primary['threshold']:.4f}  (Youden J)",
        "TP / TN / FP / FN":       (f"{primary['TP']} / {primary['TN']:,} / "
                                    f"{primary['FP']:,} / {primary['FN']}"),
        "Number Needed to Screen":  str(nns),
        "Total samples":            f"{len(all_labels):,}",
        "Positive samples":         f"{int(all_labels.sum()):,}  ({100*all_labels.mean():.2f}%)",
        "Ensemble models":          f"{len(models)} ({len(available_folds)}-fold CV)",
        "Params per model":         f"{total_params:,}",
        "Bootstrap CI resamples":   f"{N_BOOTSTRAP}",
        "Inference batch size":     f"{INFERENCE_BATCH}",
        "Primary metric method":    "OFFICIAL PhysioNet" if OFFICIAL else "sklearn approx",
        "Folds used":               str(available_folds),
        "--- Comparison ---":       "",
        "vs Kim et al. 2025":       f"{tpr_5pct-0.369:+.4f}  (Kim: 0.369)",
        "vs Van Santvliet val":     f"{tpr_5pct-0.445:+.4f}  (VS val: 0.445)",
        "vs Van Santvliet CV":      f"{tpr_5pct-0.490:+.4f}  (VS CV: 0.490)",
    }
    df_summary = pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])
    df_summary.index.name = "Metric"
    print("\n" + df_summary.to_string())

    pd.DataFrame({
        "id":                    all_ids,
        "fold":                  all_folds_arr.tolist(),
        "dataset":               all_datasets,
        "true_label":            all_labels,
        "predicted_probability": all_probs,
        "predicted_class":       pred_binary,
    }).to_csv(CHECKPOINT_DIR / "ensemble_predictions.csv", index=False)

    df_summary.to_csv(CHECKPOINT_DIR / "ensemble_summary.csv")
    df_thr.to_csv(CHECKPOINT_DIR / "threshold_comparison.csv")

    print("\nCSVs saved:")
    for f in ["ensemble_summary.csv", "threshold_comparison.csv",
              "per_dataset_metrics.csv", "per_fold_metrics.csv",
              "ensemble_predictions.csv"]:
        p = CHECKPOINT_DIR / f
        if p.exists():
            print(f"  {f}  ({p.stat().st_size/1e3:.0f} KB)")

    # Package FINAL_ENSEMBLE_MODEL.pt
    print("\nPackaging FINAL_ENSEMBLE_MODEL.pt ...")
    pkg = {
        "model_config":     MODEL_CFG,
        "ensemble_metrics": {
            "tpr_5pct":   tpr_5pct,   "auroc":    auroc,   "auprc":    auprc,
            "tpr_ci":     (tpr_lo,    tpr_hi),
            "auroc_ci":   (auroc_lo,  auroc_hi),
            "auprc_ci":   (auprc_lo,  auprc_hi),
            "threshold":  primary["threshold"],
            "n_total":    len(all_labels),
            "n_positive": int(all_labels.sum()),
            "official":   OFFICIAL,
        },
        "fold_val_scores": fold_val_scores,
        "available_folds": available_folds,
        "fold_models":     [],
    }
    for fold_idx, ckpt_path in zip(available_folds, fold_ckpts):
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pkg["fold_models"].append({
            "fold":             fold_idx,
            "model_state_dict": c["model_state_dict"],
            "val_score":        c.get("val_score", None),
            "phase":            c.get("phase", "unknown"),
        })

    out_path = CHECKPOINT_DIR / "FINAL_ENSEMBLE_MODEL.pt"
    torch.save(pkg, out_path)
    mb = out_path.stat().st_size / 1e6
    n_figs = len(list(FIGURES_DIR.glob("*.png")))
    print(f"Saved: FINAL_ENSEMBLE_MODEL.pt  ({mb:.0f} MB)")
    print(f"Figures: thesis_figures/  ({n_figs} PNG @ 300 dpi)")
    print("\n  Evaluation complete!")


# Windows multiprocessing safety -- MUST wrap main() in this guard
# so that spawned DataLoader workers don't re-run the whole script.
if __name__ == "__main__":
    main()
