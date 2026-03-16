import torch

# Check the "best" checkpoint
best_ckpt = torch.load('checkpoints/stmem_1d_pretrained.pt', weights_only=False)
print("="*70)
print("BEST MODEL CHECKPOINT (stmem_1d_pretrained.pt)")
print("="*70)
print(f"Epoch: {best_ckpt.get('epoch', 'N/A')}")
print(f"Batch: {best_ckpt.get('batch_idx', 'N/A')}")
print(f"Loss: {best_ckpt.get('loss', 'N/A')}")
print(f"Keys: {list(best_ckpt.keys())}")
print()

# Check the current checkpoint
curr_ckpt = torch.load('checkpoints/stmem_1d_checkpoint.pt', weights_only=False)
print("="*70)
print("CURRENT CHECKPOINT (stmem_1d_checkpoint.pt)")
print("="*70)
print(f"Epoch: {curr_ckpt.get('epoch', 'N/A')}")
print(f"Batch: {curr_ckpt.get('batch_idx', 'N/A')}")
print(f"Loss: {curr_ckpt.get('loss', 'N/A')}")
print(f"Best loss tracked: {curr_ckpt.get('best_loss', 'N/A')}")
print(f"Epoch complete: {curr_ckpt.get('epoch_complete', 'N/A')}")