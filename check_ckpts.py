import torch
from pathlib import Path

def inspect_checkpoint(path):
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"❌ {ckpt_path} not found")
        return
    
    print(f"\n🔍 Inspecting {ckpt_path.name}")
    
    # Load safely
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Check top-level keys
    print("Top-level keys:", list(ckpt.keys())[:10])
    
    # Extract state dict
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    if isinstance(state, dict):
        print("✓ Found state_dict")
        keys = list(state.keys())
        print("First 40 keys:")
        for i, k in enumerate(keys[:40]):
            print(f"  {i:2d}: {k}")
        print(f"\nTotal keys: {len(keys)}")
        if len(keys) > 40:
            print(f"Last 5 keys: {keys[-5:]}")
    else:
        print("❌ No model_state_dict or state_dict found")
        print("Checkpoint structure:", type(ckpt))

# Inspect both
inspect_checkpoint('checkpoints/mae_2d_pretrained.pt')
inspect_checkpoint('checkpoints/stmem_1d_pretrained.pt')
