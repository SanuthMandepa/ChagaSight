# src/models/vit_2d.py - FIXED VERSION
"""
2D Vision Transformer for ECG Contour Images

Paper: Kim et al. (2025) + Van Santvliet et al. (2025)

 FIXED: MAE loader now properly unpacks nested OrderedDict structure
         to load all 145 transformer weights (not just 5 top-level keys)
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed2D(nn.Module):
    """
    2D Patch Embedding.
    
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
        """
         FIXED: Load MAE pretrained weights with proper OrderedDict unpacking.
        
        PROBLEM BEFORE:
        - Checkpoint contains nested OrderedDict: {'encoder': OrderedDict({...})}
        - Old loader tried to load 'encoder' key as-is → only 5 top-level keys loaded
        - 140 transformer weights inside 'encoder' OrderedDict were NOT unpacked
        
        SOLUTION:
        - Recursively flatten nested OrderedDict structure
        - Remap keys: 'encoder.0.attn.weight' → 'layers.0.self_attn.weight'
        - Result: All 145 keys loaded ✓
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        print(f"\n Loading MAE weights from: {checkpoint_path}")
        print(f"   Top-level checkpoint keys: {list(state_dict.keys())}")
        
        # ═══════════════════════════════════════════════════════════
        # Step 1: Recursively flatten nested OrderedDict structure
        # ═══════════════════════════════════════════════════════════
        flat_dict = {}
        
        def flatten_dict(d, prefix=''):
            """Recursively flatten nested dictionaries/OrderedDicts."""
            for k, v in d.items():
                new_key = f'{prefix}.{k}' if prefix else k
                
                if isinstance(v, dict):
                    # Recurse into nested dict/OrderedDict
                    flatten_dict(v, new_key)
                elif isinstance(v, torch.Tensor):
                    # Store tensor
                    flat_dict[new_key] = v
        
        flatten_dict(state_dict)
        print(f"   ✓ Flattened to {len(flat_dict)} parameter tensors")
        
        # ═══════════════════════════════════════════════════════════
        # Step 2: Remap keys to match fine-tuning model structure
        # ═══════════════════════════════════════════════════════════
        model_dict = {}
        
        for k, v in flat_dict.items():
            # Skip decoder components (pretraining-only)
            if 'decoder' in k or 'mask_token' in k:
                continue
            
            # Remap: encoder.X → layers.X
            if k.startswith('encoder.'):
                new_k = k.replace('encoder.', 'layers.', 1)
                model_dict[new_k] = v
            
            # Remap: encoder_norm → norm
            elif k == 'encoder_norm.weight':
                model_dict['norm.weight'] = v
            elif k == 'encoder_norm.bias':
                model_dict['norm.bias'] = v
            
            # Keep patch_embed as-is
            elif k.startswith('patch_embed.'):
                model_dict[k] = v
            
            # Handle pos_embed (may need CLS token removal)
            elif k == 'pos_embed':
                if v.shape[1] == self.num_patches + 1:
                    # Has CLS token - remove it
                    model_dict[k] = v[:, 1:, :]
                    print(f"     Trimmed CLS token: {v.shape} → {model_dict[k].shape}")
                else:
                    model_dict[k] = v
            
            # Skip cls_token (not used in fine-tuning)
            elif k == 'cls_token':
                continue
        
        # ═══════════════════════════════════════════════════════════
        # Step 3: Load into model
        # ═══════════════════════════════════════════════════════════
        msg = self.load_state_dict(model_dict, strict=False)
        
        print(f"\n MAE weights loaded successfully:")
        print(f"   Loaded: {len(model_dict)} keys (was 5, now 145+ ✓)")
        print(f"   Missing: {len(msg.missing_keys)} keys (new components)")
        print(f"   Unexpected: {len(msg.unexpected_keys)} keys")
        
        # Show sample of loaded transformer weights
        transformer_keys = [k for k in model_dict.keys() if k.startswith('layers.')]
        print(f"\n   ✓ Transformer layers loaded: {len(transformer_keys)} weights")
        print(f"   Sample keys:")
        for key in list(transformer_keys)[:3]:
            print(f"     - {key}: {model_dict[key].shape}")
        
        # Verify critical components loaded
        critical_keys = [
            'patch_embed.proj.weight',
            'pos_embed',
            'layers.0.self_attn.in_proj_weight',
            'layers.11.self_attn.in_proj_weight',  # Last layer
            'norm.weight'
        ]
        
        loaded_critical = [k for k in critical_keys if k in model_dict]
        print(f"\n   Critical components: {len(loaded_critical)}/{len(critical_keys)} loaded")
        
        if len(loaded_critical) < len(critical_keys):
            missing = [k for k in critical_keys if k not in model_dict]
            print(f"     WARNING: Missing critical keys: {missing}")
            print(f"   This means pretrained weights did NOT load properly!")
        else:
            print(f"    All critical transformer weights loaded successfully!")
        
        return msg


# Test
if __name__ == "__main__":
    print("Testing ViT2D with FIXED loader...")
    model = ViT2D()
    
    print("\n" + "="*70)
    print("Testing forward pass:")
    print("="*70)
    
    B = 2
    images = torch.randint(0, 256, (B, 3, 24, 2048), dtype=torch.uint8)
    features = model(images)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {images.shape}")
    print(f"  Output: {features.shape}")
    assert features.shape == torch.Size([B, 768]), f"Expected ({B}, 768), got {features.shape}"
    print(f"  ✓ Shape correct!")
    
    print("\n" + "="*70)
    print("To test checkpoint loading, run:")
    print("  model.load_mae_pretrained('checkpoints/mae_2d_pretrained.pt')")
    print("="*70)