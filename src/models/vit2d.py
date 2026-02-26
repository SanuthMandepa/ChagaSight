# src/models/vit_2d.py
"""
2D Vision Transformer for ECG Contour Images

Paper: Kim et al. (2025) + Van Santvliet et al. (2025)

CORRECTED:
- Patch size: (8, 64) divides (24, 2048) evenly
- 24÷8=3, 2048÷64=32 → 96 patches total
- AoL (Aggregation of Layers) from all 12 layers
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed2D(nn.Module):
    """
    CORRECTED 2D Patch Embedding.
    
    Patch size: (8, 64)
    Image size: (24, 2048)
    Result: 3×32 = 96 patches
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (24, 2048),
        patch_size: Tuple[int, int] = (8, 64),
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Verify patches divide evenly
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        
        self.num_patches_h = img_size[0] // patch_size[0]  # 3
        self.num_patches_w = img_size[1] // patch_size[1]  # 32
        self.num_patches = self.num_patches_h * self.num_patches_w  # 96
        
        # Conv2D projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 24, 2048) uint8 images [0, 255]
        
        Returns:
            patches: (B, 96, 768)
        """
        # Normalize to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Project patches
        x = self.proj(x)  # (B, 768, 3, 32)
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (B, 768, 96)
        
        # Transpose to (B, 96, 768)
        x = x.transpose(1, 2)
        
        return x


class ViT2D(nn.Module):
    """
    2D Vision Transformer with AoL.
    
    Paper: Van Santvliet et al. (2025) - AoL gives +11% improvement
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (24, 2048),
        patch_size: Tuple[int, int] = (8, 64),
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_aol: bool = True
    ):
        super().__init__()
        
        self.use_aol = use_aol
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            img_size, patch_size, in_channels, embed_dim
        )
        
        self.num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer encoder
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # For AoL
        self.layer_outputs = []
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 24, 2048) images
        
        Returns:
            features: (B, 768)
        """
        self.layer_outputs = []
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 96, 768)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
            
            if self.use_aol:
                self.layer_outputs.append(x)
        
        # Final norm
        x = self.norm(x)
        
        if self.use_aol:
            # AoL: Aggregate from all layers
            layer_features = []
            for layer_output in self.layer_outputs:
                pooled = layer_output.mean(dim=1)  # (B, 768)
                layer_features.append(pooled)
            
            features = torch.stack(layer_features, dim=0).mean(dim=0)  # (B, 768)
        else:
            features = x.mean(dim=1)  # (B, 768)
        
        return features
    
    def load_mae_pretrained(self, checkpoint_path, strict=False):
        """Load MAE pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter encoder keys
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if 'decoder' not in k and 'mask_token' not in k:
                key = k.replace('encoder.', '')
                encoder_state_dict[key] = v
        
        msg = self.load_state_dict(encoder_state_dict, strict=strict)
        
        print(f"✓ Loaded MAE pretrained weights")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        
        return msg