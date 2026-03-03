# check_ckpts_DEEP.py
import torch
from pathlib import Path

def deep_inspect_checkpoint(path):
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"❌ {ckpt_path} not found")
        return
    
    print(f"\n{'='*70}")
    print(f"🔍 DEEP INSPECTION: {ckpt_path.name}")
    print(f"{'='*70}")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    
    print(f"\n📊 Checkpoint Metadata:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Loss: {ckpt.get('loss', 'N/A'):.6f}" if 'loss' in ckpt else "")
    
    print(f"\n📁 Top-level keys: {len(state)}")
    for i, key in enumerate(state.keys()):
        value = state[key]
        print(f"  {i}: {key:20s} → {type(value).__name__}")
    
    # Deep dive into each key
    total_params = 0
    
    for key, value in state.items():
        print(f"\n{'─'*70}")
        print(f"🔎 Inspecting: {key}")
        print(f"{'─'*70}")
        
        if isinstance(value, torch.nn.ModuleList):
            print(f"  Type: ModuleList with {len(value)} blocks")
            print(f"  Each block type: {type(value[0]).__name__}")
            
            # Count parameters in first block
            block_params = sum(p.numel() for p in value[0].parameters())
            total_block_params = block_params * len(value)
            total_params += total_block_params
            
            print(f"  Params per block: {block_params:,}")
            print(f"  Total params: {total_block_params:,}")
            
            # Show first block's structure
            print(f"\n  First block parameters:")
            for i, (name, param) in enumerate(value[0].named_parameters()):
                if i < 10:  # Show first 10
                    print(f"    └─ {name:40s} {param.shape}")
            if sum(1 for _ in value[0].named_parameters()) > 10:
                print(f"    └─ ... ({sum(1 for _ in value[0].named_parameters()) - 10} more)")
        
        elif isinstance(value, torch.nn.Module):
            print(f"  Type: {type(value).__name__}")
            params = sum(p.numel() for p in value.parameters())
            total_params += params
            print(f"  Parameters: {params:,}")
            
            # Show module parameters
            print(f"  Module parameters:")
            for name, param in value.named_parameters():
                print(f"    └─ {name:40s} {param.shape}")
        
        elif isinstance(value, torch.nn.Parameter) or isinstance(value, torch.Tensor):
            print(f"  Type: Parameter/Tensor")
            print(f"  Shape: {value.shape}")
            print(f"  Elements: {value.numel():,}")
            total_params += value.numel()
        
        else:
            print(f"  Type: {type(value).__name__}")
    
    print(f"\n{'='*70}")
    print(f"📊 TOTAL PARAMETERS IN CHECKPOINT: {total_params:,}")
    print(f"   Size estimate: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"{'='*70}")

# Inspect both
deep_inspect_checkpoint('checkpoints/mae_2d_pretrained.pt')
deep_inspect_checkpoint('checkpoints/stmem_1d_pretrained.pt')