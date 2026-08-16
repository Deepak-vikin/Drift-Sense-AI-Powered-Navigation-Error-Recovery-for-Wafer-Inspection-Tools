import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Dict, Any

def save_metrics_csv(results: List[Dict[str, Any]], filepath: str):
    """Export benchmark results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    return df

def plot_error_distribution(results: List[Dict[str, Any]], out_dir: str):
    """Plot histogram of localization errors."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([r for r in results if r["success"]])
    
    if df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    plt.hist(df['error_pixels'].clip(upper=100), bins=50, color='skyblue', edgecolor='black')
    plt.title('Localization Error Distribution (Clipped at 100px)')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(out_dir, 'error_distribution.png'), dpi=150)
    plt.close()

def plot_accuracy_vs_noise(results: List[Dict[str, Any]], out_dir: str):
    """Plot accuracy by noise condition."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([r for r in results if r["success"]])
    
    if df.empty:
        return
        
    noise_conds = ['clean', 'low_noise', 'medium_noise', 'high_noise', 'search_noisier']
    df = df[df['condition'].isin(noise_conds)]
    
    if df.empty:
        return
        
    acc = []
    for cond in noise_conds:
        sub = df[df['condition'] == cond]
        if not sub.empty:
            acc.append(len(sub[sub['error_pixels'] <= 10]) / len(sub) * 100.0)
        else:
            acc.append(0)
            
    plt.figure(figsize=(10, 6))
    plt.bar(noise_conds, acc, color='coral')
    plt.title('Accuracy (<=10px error) vs Noise Level')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for i, v in enumerate(acc):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    plt.savefig(os.path.join(out_dir, 'accuracy_vs_noise.png'), dpi=150)
    plt.close()

def plot_baseline_vs_proposed(results: List[Dict[str, Any]], out_dir: str):
    """Plot comparison between baseline and proposed."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    
    if df.empty or 'method' not in df.columns:
        return
        
    df = df[df['success'] == True]
    
    conditions = df['condition'].unique()
    baseline_acc = []
    proposed_acc = []
    
    for cond in conditions:
        b_sub = df[(df['condition'] == cond) & (df['method'] == 'baseline')]
        p_sub = df[(df['condition'] == cond) & (df['method'] == 'proposed')]
        
        b_val = len(b_sub[b_sub['error_pixels'] <= 10]) / len(b_sub) * 100.0 if not b_sub.empty else 0
        p_val = len(p_sub[p_sub['error_pixels'] <= 10]) / len(p_sub) * 100.0 if not p_sub.empty else 0
        
        baseline_acc.append(b_val)
        proposed_acc.append(p_val)
        
    x = np.arange(len(conditions))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_acc, width, label='Baseline', color='gray')
    plt.bar(x + width/2, proposed_acc, width, label='Proposed', color='forestgreen')
    
    plt.title('Accuracy (<=10px error): Baseline vs Proposed')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, conditions, rotation=45)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'baseline_vs_proposed.png'), dpi=150)
    plt.close()

def save_case_visualization(item: Dict[str, Any], result: Dict[str, Any], out_path: str):
    """Save an annotated success or failure case image."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    search = item['search']
    ref = item['reference']
    vis = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    
    tx, ty = item['true_x'], item['true_y']
    px, py = result['pred_x'], result['pred_y']
    err = result['error_pixels']
    
    # Draw true target (Green Box)
    cv2.rectangle(vis, (tx - 50, ty - 50), (tx + 50, ty + 50), (0, 255, 0), 2)
    cv2.drawMarker(vis, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    
    # Draw predicted target (Red Box)
    cv2.rectangle(vis, (px - 50, py - 50), (px + 50, py + 50), (0, 0, 255), 2)
    cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    
    # Add Text
    label = f"True: ({tx},{ty}) | Pred: ({px},{py}) | Err: {err:.1f}px"
    cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Embed reference in top right
    vis[20:120, -120:-20] = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (vis.shape[1]-120, 20), (vis.shape[1]-20, 120), (255,255,255), 1)
    cv2.putText(vis, "REF", (vis.shape[1]-115, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.imwrite(out_path, vis)
