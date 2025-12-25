# notebooks/train_vit_final.py
# ChagaSight: FINAL WORKING ViT Training + Inference Script
# Run from project root with: python notebooks/train_vit_final.py

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==============================
# Configuration & Setup
# ==============================
start_total = time.time()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Paths - CORRECT when running from project root
DATA_BASE = Path("data/processed")  # Direct path from project root
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "vit_chagas_best.pth"
LAST_MODEL_PATH = MODEL_DIR / "vit_chagas_last.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 2
SUBSET_FRAC = 0.1

# ==============================
# Step 1: Load Metadata & Split
# ==============================
print("\nStep 1: Loading metadata and creating splits...")
try:
    datasets = ['ptbxl', 'sami_trop', 'code15']
    df_list = []
    for ds in datasets:
        csv_path = DATA_BASE / "metadata" / f"{ds}_metadata.csv"
        if csv_path.exists():
            print(f"  Loading {csv_path.name}...")
            df = pd.read_csv(csv_path)
            df['dataset'] = ds
            df_list.append(df)
        else:
            raise FileNotFoundError(f"Metadata not found: {csv_path}")

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Total records loaded: {len(df_all)}")

    df_all['label_bin'] = np.where(df_all['label'] < 0.3, 0,
                                   np.where(df_all['label'] > 0.7, 1, 0.5))

    df_all = df_all.sample(frac=SUBSET_FRAC, random_state=42).reset_index(drop=True)

    df_all['stratify_group'] = df_all['label_bin'].astype(str) + "_" + df_all['dataset']
    train_df, temp_df = train_test_split(df_all, test_size=0.2, stratify=df_all['stratify_group'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['stratify_group'], random_state=42)

    for df in [df_all, train_df, val_df, test_df]:
        df.drop(columns=['stratify_group', 'label_bin'], errors='ignore', inplace=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Positive ratio (label > 0.7): {(df_all['label'] > 0.7).mean():.4f}")

except Exception as e:
    print(f"ERROR in data loading: {e}")
    exit(1)

# ==============================
# Step 2: Dataset & DataLoader
# ==============================
print("\nStep 2: Creating datasets and loaders...")
try:
    class ECGImageDataset(Dataset):
        def __init__(self, df, augment=True):
            self.df = df.reset_index(drop=True)
            self.augment = augment
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]) if augment else transforms.Compose([])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            try:
                img = np.load(row['img_path']).astype(np.float32)
            except Exception as e:
                raise FileNotFoundError(f"Image not found: {row['img_path']} | Error: {e}")
            
            if img.shape == (2048, 3, 24):
                img = img.transpose(1, 2, 0)
            elif img.shape != (3, 24, 2048):
                raise ValueError(f"Unexpected shape {img.shape}")
            
            img = torch.from_numpy(img)
            img = self.aug_transform(img)
            
            label = torch.tensor(row['label'], dtype=torch.float32)
            return img, label

    train_ds = ECGImageDataset(train_df, augment=True)
    val_ds   = ECGImageDataset(val_df, augment=False)
    test_ds  = ECGImageDataset(test_df, augment=False)

    weights = train_df['label'].apply(lambda x: 10.0 if x > 0.7 else 1.0).values
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    img_batch, lbl_batch = next(iter(train_loader))
    print(f"Batch shape: {img_batch.shape} | Labels sample: {lbl_batch.tolist()[:10]}")

except Exception as e:
    print(f"ERROR in dataset/dataloader: {e}")
    exit(1)

# ==============================
# Step 3: ViT Model
# ==============================
print("\nStep 3: Defining ViT model...")
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (24 // patch_size) * (2048 // patch_size)
    
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

class ViTClassifier(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, depth=12, heads=12, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, dropout=dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.head(x).squeeze(-1)

model = ViTClassifier().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ==============================
# Step 4: Training
# ==============================
print("\nStep 4: Starting training...")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_auroc = 0.0
train_losses = []
val_aurocs = []

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits)
            val_preds.extend(probs.cpu().numpy())
            val_trues.extend(labels.numpy())

    val_preds = np.array(val_preds)
    val_trues = np.array(val_trues)
    val_trues_bin = (val_trues > 0.5).astype(int)
    val_auroc = roc_auc_score(val_trues_bin, val_preds)
    val_auprc = average_precision_score(val_trues_bin, val_preds)
    val_aurocs.append(val_auroc)

    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  → New best model saved! Val AUROC: {val_auroc:.4f}")

    scheduler.step()
    print(f"Epoch {epoch+1} | Loss: {train_losses[-1]:.4f} | Val AUROC: {val_auroc:.4f} | Time: {time.time() - epoch_start:.1f}s")

# ==============================
# Step 5: Final Test Evaluation
# ==============================
print("\nStep 5: Final test evaluation...")
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

test_preds, test_trues = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits)
        test_preds.extend(probs.cpu().numpy())
        test_trues.extend(labels.numpy())

test_preds = np.array(test_preds)
test_trues = np.array(test_trues)
test_trues_bin = (test_trues > 0.5).astype(int)

test_auroc = roc_auc_score(test_trues_bin, test_preds)
test_auprc = average_precision_score(test_trues_bin, test_preds)

sorted_idx = np.argsort(-test_preds)
top_k = max(1, int(0.05 * len(test_preds)))
top_trues = test_trues[sorted_idx[:top_k]]
challenge_score = np.mean(top_trues > 0.7)

print("\n=== FINAL RESULTS ===")
print(f"Test AUROC: {test_auroc:.4f}")
print(f"Test AUPRC: {test_auprc:.4f}")
print(f"Challenge Score (top 5% confident positives): {challenge_score:.4f}")
print(f"Best model saved at: {BEST_MODEL_PATH}")

total_time = time.time() - start_total
print(f"\nTotal script time: {total_time/60:.1f} minutes")

# ==============================
# Inference Function - Ready for Viva Demo
# ==============================
print("\n" + "="*60)
print("MODEL READY FOR INFERENCE!")
print("="*60)

def predict_chagas_probability(img_npy_path):
    """
    Predict Chagas probability for a single 2D contour image (.npy)
    Returns probability (0.0 to 1.0)
    """
    model.eval()
    with torch.no_grad():
        img = np.load(img_npy_path).astype(np.float32)
        if img.shape == (2048, 3, 24):
            img = img.transpose(1, 2, 0)
        elif img.shape != (3, 24, 2048):
            raise ValueError(f"Invalid shape: {img.shape}")
        img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        logit = model(img)
        prob = torch.sigmoid(logit).item()
        return prob

print("Example usage:")
print("prob = predict_chagas_probability('data/processed/2d_images/ptbxl/00001_hr_img.npy')")
print("print(f'Chagas probability: {prob:.4f} → {'POSITIVE' if prob > 0.5 else 'NEGATIVE'}')")

print("\nTraining and model preparation complete!")
print("You are fully ready for your viva demonstration tomorrow!")