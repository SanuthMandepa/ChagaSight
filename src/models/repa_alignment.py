# src/models/repa_alignment.py
"""
REPA Alignment Module

Paper: Kim et al. (2025)

CORRECTED:
- Simple projection (NOT cross-attention)
- Depthwise Conv → SiLU → MLP
- FM features detached during alignment loss
"""

import torch
import torch.nn as nn


class REPAAlignment(nn.Module):
    """
    CORRECTED Simple REPA alignment.
    
    Paper: Kim et al. (2025) - Simple projection works best
    
    Architecture:
    - Depthwise Conv1d (groups=768)
    - SiLU activation
    - Linear projection
    
    NOT using cross-attention (overcomplicated).
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        
        self.projection = nn.Sequential(
            # Depthwise convolution
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, groups=embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, features_2d):
        """
        Args:
            features_2d: (B, 768) from 2D-ViT
        
        Returns:
            aligned: (B, 768) aligned features
        """
        # Add channel dimension for Conv1d
        x = features_2d.unsqueeze(2)  # (B, 768, 1)
        
        # Depthwise conv
        x = self.projection[0](x)  # (B, 768, 1)
        x = x.squeeze(2)  # (B, 768)
        
        # SiLU
        x = self.projection[1](x)  # (B, 768)
        
        # Linear
        x = self.projection[2](x)  # (B, 768)
        
        return x