# src/models/hybrid_model.py
"""
Hybrid ChagaSight Model - Dual-Pathway System

Combines:
- 2D-ViT (contour images)
- 1D-ViT FM (raw signals + demographics)
- REPA alignment
- MLP classifier

Papers: Kim et al. (2025) + Van Santvliet et al. (2025)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

from .vit_2d import ViT2D
from .vit_1d_fm import ViT1D_FM
from .repa_alignment import REPAAlignment


class HybridChagasModel(nn.Module):
    """
    Complete dual-pathway model for Chagas detection.
    
    Architecture:
    1. 2D-ViT: Processes contour images → (B, 768)
    2. 1D-ViT FM: Processes signals + demographics → (B, 768)
    3. REPA: Aligns 2D features to FM space → (B, 768)
    4. Classifier: [Aligned_2D | FM] → (B, 1536) → prediction
    """
    
    def __init__(
        self,
        # 2D pathway
        img_size: Tuple[int, int] = (24, 2048),
        patch_size_2d: Tuple[int, int] = (8, 64),
        # 1D pathway
        num_leads: int = 12,
        seq_len_1d: int = 1000,
        patch_size_1d: int = 50,
        # Shared
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # Features
        use_aol: bool = True,
        use_demographics: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 2D pathway
        self.vit_2d = ViT2D(
            img_size=img_size,
            patch_size=patch_size_2d,
            in_channels=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_aol=use_aol
        )
        
        # 1D pathway
        self.vit_1d_fm = ViT1D_FM(
            num_leads=num_leads,
            seq_len=seq_len_1d,
            patch_size=patch_size_1d,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_aol=use_aol,
            use_demographics=use_demographics
        )
        
        # Alignment
        self.repa = REPAAlignment(embed_dim=embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),  # 1536 → 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary classification
        )
    
    def forward(
        self,
        images: torch.Tensor,
        signals: torch.Tensor,
        ages: torch.Tensor,
        sexes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-pathway system.
        
        Args:
            images: (B, 3, 24, 2048) contour images
            signals: (B, 12, 1000) ECG signals
            ages: (B,) age in centuries
            sexes: (B,) sex as binary
        
        Returns:
            dict with:
                - logits: (B, 1) raw predictions
                - fm_features: (B, 768) from FM
                - aligned_2d_features: (B, 768) aligned 2D
        """
        # 2D pathway
        features_2d = self.vit_2d(images)  # (B, 768)
        
        # 1D pathway
        features_fm = self.vit_1d_fm(signals, ages, sexes)  # (B, 768)
        
        # Align 2D to FM space
        aligned_2d = self.repa(features_2d)  # (B, 768)
        
        # Concatenate for classification
        combined = torch.cat([aligned_2d, features_fm], dim=1)  # (B, 1536)
        
        # Classify
        logits = self.classifier(combined)  # (B, 1)
        
        return {
            'logits': logits.squeeze(1),  # (B,)
            'fm_features': features_fm,  # For alignment loss
            'aligned_2d_features': aligned_2d  # For alignment loss
        }


# Test the model
if __name__ == "__main__":
    print("Testing HybridChagasModel...")
    
    model = HybridChagasModel()
    
    # Create dummy inputs
    B = 4
    images = torch.randint(0, 256, (B, 3, 24, 2048), dtype=torch.uint8)
    signals = torch.randn(B, 12, 1000)
    ages = torch.rand(B) * 1.2  # [0, 1.2] centuries
    sexes = torch.randint(0, 2, (B,)).float()
    
    # Forward pass
    outputs = model(images, signals, ages, sexes)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  FM features shape: {outputs['fm_features'].shape}")
    print(f"  Aligned 2D features shape: {outputs['aligned_2d_features'].shape}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total:,}")