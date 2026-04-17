import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

matplotlib.use("Agg")

def smooth(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

FOLD_COLORS = ['#2C6FAC', '#E07B2A', '#2DA050', '#7D3C98', '#1A8C8C']

os.makedirs('thesis_figures', exist_ok=True)

print("Generating Phase 2 Gradient Norm Graphs...")

for fold_idx in range(5):
    path = f'checkpoints/fold{fold_idx}_best.pt'
    if not os.path.exists(path):
        continue
    
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    history = ckpt.get('history', {})
    
    grad_norm = history.get('grad_norm', [])
    if not grad_norm:
        print(f"Fold {fold_idx} has no grad norm history.")
        continue
        
    p2_iters = ckpt.get('iteration', 24000)
    grad_norm = grad_norm[-p2_iters:] if len(grad_norm) > p2_iters else grad_norm
    
    s_grad = smooth(grad_norm, window=50)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.linspace(0, p2_iters, len(s_grad)), s_grad, color=FOLD_COLORS[fold_idx], lw=1.5)
    
    ax.set_xlabel('Phase 2 Iterations (Supervised Fine-Tuning)')
    ax.set_ylabel('Gradient L2 Norm')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    out_path = f'thesis_figures/fig_c8_fold{fold_idx}_gradient_norm.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

print("Done generating Phase 2 Gradient Norm graphs.")
