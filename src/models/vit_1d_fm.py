# src/models/vit_1d_fm.py - FULLY FIXED VERSION
"""
1D Vision Transformer Foundation Model

Paper: Van Santvliet et al. (2025)

CRITICAL FIXES:
1. Per-lead processing (previous fix)
2. Use .contiguous() before .view() to avoid stride errors
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed1D(nn.Module):
    """
    FULLY FIXED 1D Patch Embedding for ECG signals.
    
    Input: (12, 1000) @ 100Hz
    Patch size: 50 samples  
    Result: 12 leads × 20 patches = 240 patches
    
    FIXES:
    - Per-lead processing (not all leads at once)
    - .contiguous() before .view() to avoid stride errors
    """
    
    def __init__(
        self,
        num_leads: int = 12,
        seq_len: int = 1000,
        patch_size: int = 50,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.num_leads = num_leads
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        assert seq_len % patch_size == 0
        
        self.num_patches_per_lead = seq_len // patch_size  # 20
        self.num_patches = num_leads * self.num_patches_per_lead  # 240
        
        # Per-lead Conv1D projection
        self.proj = nn.Conv1d(
            1,  # Process one lead at a time
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Lead embeddings
        self.lead_embed = nn.Parameter(
            torch.zeros(1, num_leads, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.lead_embed, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, 12, 1000) signals
        
        Returns:
            patches: (B, 240, 768)
        """
        B, L, T = x.shape
        
        # Process each lead separately
        x = x.view(B * L, 1, T)  # (B*12, 1, 1000)
        
        # Project patches
        x = self.proj(x)  # (B*12, 768, 20)
        
        # Reshape back
        x = x.view(B, L, self.embed_dim, self.num_patches_per_lead)  # (B, 12, 768, 20)
        x = x.permute(0, 1, 3, 2)  # (B, 12, 20, 768)
        
        # Add lead embeddings
        x = x + self.lead_embed  # (B, 12, 20, 768) + (1, 12, 1, 768)
        
        # FIXED: Make contiguous before view
        x = x.contiguous()  # Ensure tensor is contiguous
        x = x.view(B, self.num_patches, self.embed_dim)  # (B, 240, 768)
        
        return x


class DemographicsEncoder(nn.Module):
    """
    Demographics encoder: (age, sex) → (γ, β) modulation.
    """
    
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        
        # Initialize to identity
        self.mlp[-1].weight.data.zero_()
        self.mlp[-1].bias.data.zero_()
        self.mlp[-1].bias.data[:embed_dim] = 1.0  # γ = 1
    
    def forward(self, age, sex):
        """
        Args:
            age: (B,) age in centuries
            sex: (B,) sex as binary
        
        Returns:
            gamma: (B, 768)
            beta: (B, 768)
        """
        demo = torch.stack([age, sex], dim=1)  # (B, 2)
        params = self.mlp(demo)  # (B, 1536)
        gamma, beta = params.chunk(2, dim=1)
        return gamma, beta


class ViT1D_FM(nn.Module):
    """
    1D Vision Transformer Foundation Model.
    """
    
    def __init__(
        self,
        num_leads: int = 12,
        seq_len: int = 1000,
        patch_size: int = 50,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_aol: bool = True,
        use_demographics: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_aol = use_aol
        self.use_demographics = use_demographics
        
        # Patch embedding
        self.patch_embed = PatchEmbed1D(
            num_leads, seq_len, patch_size, embed_dim
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
        
        # Demographics encoder
        if use_demographics:
            self.demo_encoder = DemographicsEncoder(embed_dim)
        
        # For AoL
        self.layer_outputs = []
    
    def forward(self, x, age=None, sex=None):
        """
        Args:
            x: (B, 12, 1000) signals
            age: (B,) age in centuries (optional)
            sex: (B,) sex as binary (optional)
        
        Returns:
            features: (B, 768)
        """
        self.layer_outputs = []
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 240, 768)
        
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
        
        # Aggregate features
        if self.use_aol:
            layer_features = []
            for layer_output in self.layer_outputs:
                pooled = layer_output.mean(dim=1)
                layer_features.append(pooled)
            
            features = torch.stack(layer_features, dim=0).mean(dim=0)
        else:
            features = x.mean(dim=1)
        
        # Demographics modulation
        if self.use_demographics and age is not None and sex is not None:
            gamma, beta = self.demo_encoder(age, sex)
            features = gamma * features + beta
        
        return features
    
    def load_stmem_pretrained(self, checkpoint_path, strict=False):
        """Load ST-MEM pretrained weights."""
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
            if 'decoder' not in k and 'mask' not in k:
                key = k.replace('encoder.', '')
                encoder_state_dict[key] = v
        
        msg = self.load_state_dict(encoder_state_dict, strict=strict)
        
        print(f"✓ Loaded ST-MEM pretrained weights")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        
        return msg


# Test
if __name__ == "__main__":
    print("Testing FULLY FIXED ViT1D_FM...")
    
    model = ViT1D_FM()
    
    B = 16
    signals = torch.randn(B, 12, 1000)
    ages = torch.rand(B) * 1.2
    sexes = torch.randint(0, 2, (B,)).float()
    
    features = model(signals, ages, sexes)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {signals.shape}")
    print(f"  Output: {features.shape}")
    assert features.shape == torch.Size([B, 768]), f"Expected ({B}, 768), got {features.shape}"
    print(f"  ✓ Shape correct!")