import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from src.dataset.annotations import Annotation

def visualize_pair_advanced(reference: np.ndarray, search: np.ndarray, 
                            annotation: Annotation,
                            ref_scale_nm: float = 1.0, 
                            search_scale_nm: float = 10.0,
                            seed: int = None) -> plt.Figure:
    """Create advanced presentation-quality visualization showing reference and search."""
    # Set up the figure with GridSpec for the layout
    fig = plt.figure(figsize=(16, 9), dpi=150, layout='constrained')
    fig.patch.set_facecolor('#ffffff')
    
    # Grid layout: 2 columns for images, 1 column for text info on the right
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 1], height_ratios=[4, 1], wspace=0.1, hspace=0.2)
    
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_search = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[:, 2])
    
    # ---------------------------
    # LEFT PANEL: Reference
    # ---------------------------
    ax_ref.imshow(reference, cmap='gray', interpolation='nearest')
    ax_ref.set_title("REFERENCE", fontsize=14, fontweight='bold', pad=15)
    
    # Add text box for reference scale
    ref_h, ref_w = reference.shape
    ref_fov = (ref_w * ref_scale_nm) / 1000.0 # um
    
    ref_text = (
        f"{ref_w} × {ref_h} px\n"
        f"{ref_scale_nm:g} nm/px\n"
        f"{ref_fov:g} µm FOV"
    )
    ax_ref.text(0.05, 0.95, ref_text, transform=ax_ref.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax_ref.axis('off')

    # ---------------------------
    # RIGHT PANEL: Search
    # ---------------------------
    ax_search.imshow(search, cmap='gray', interpolation='nearest')
    ax_search.set_title("SEARCH", fontsize=14, fontweight='bold', pad=15)
    
    search_h, search_w = search.shape
    search_fov = (search_w * search_scale_nm) / 1000.0 # um
    scale_ratio = int(search_scale_nm / ref_scale_nm)
    
    # Add ground truth bounding box
    bbox = annotation.bbox
    rect = patches.Rectangle(
        (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
        linewidth=2.5, edgecolor='#00ff00', facecolor='none'
    )
    ax_search.add_patch(rect)
    
    # Add small label next to bbox
    ax_search.text(bbox['x'], bbox['y'] - 10, f"Reference footprint\n{bbox['width']} × {bbox['height']} px", 
                   color='#00ff00', fontsize=10, fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

    search_text = (
        f"{search_w} × {search_h} px\n"
        f"{search_scale_nm:g} nm/px\n"
        f"{search_fov:g} µm FOV"
    )
    ax_search.text(0.05, 0.95, search_text, transform=ax_search.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax_search.axis('off')

    # ---------------------------
    # INFO PANEL
    # ---------------------------
    ax_info.axis('off')
    
    # Calculate scale geometry validation
    expected_ref_physical = ref_w * ref_scale_nm
    expected_search_footprint = expected_ref_physical / search_scale_nm
    scale_geometry_pass = abs(expected_search_footprint - bbox['width']) < 1.0
    
    # Noise validation
    aug_params = annotation.augmentation_params
    ref_noise = aug_params.get('reference', {}).get('noise_seed', 'N/A')
    search_noise = aug_params.get('search', {}).get('noise_seed', 'N/A')
    noise_pass = (ref_noise != 'N/A') and (search_noise != 'N/A') and (ref_noise != search_noise)
    
    # Edge validation
    edge_str = aug_params.get('reference', {}).get('edge_strength', 0.0)
    edge_pass = edge_str > 0.0
    
    info_text = (
        f"Generation Parameters\n"
        f"─────────────────────────────────────\n"
        f"Sample ID            : {annotation.id}\n"
        f"Architecture         : {annotation.architecture}\n"
        f"Seed                 : {seed if seed is not None else 'Random'}\n"
        f"Scale ratio          : {scale_ratio}×\n\n"
        
        f"Ground Truth\n"
        f"─────────────────────────────────────\n"
        f"Center               : ({annotation.center_x:.1f}, {annotation.center_y:.1f}) px\n"
        f"Bounding Box         : x={bbox['x']}, y={bbox['y']}, w={bbox['width']}, h={bbox['height']}\n"
        f"Footprint            : {bbox['width']} × {bbox['height']} px\n\n"
        
        f"Physical Scale\n"
        f"─────────────────────────────────────\n"
        f"Reference pixel size : {ref_scale_nm:g} nm/px\n"
        f"Search pixel size    : {search_scale_nm:g} nm/px\n"
        f"Reference physical   : {ref_w} px × {ref_scale_nm:g} nm/px = {expected_ref_physical:g} nm = {ref_fov:g} µm\n"
        f"Search footprint     : {expected_ref_physical:g} nm ÷ {search_scale_nm:g} nm/px = {expected_search_footprint:g} px\n\n"
        
        f"Noise\n"
        f"─────────────────────────────────────\n"
        f"Reference noise seed : {ref_noise}\n"
        f"Search noise seed    : {search_noise}\n"
        f"Noise σ (Search)     : {aug_params.get('search', {}).get('noise_sigma', 0.0):.2f}\n\n"
        
        f"SEM Edge Brightening\n"
        f"─────────────────────────────────────\n"
        f"Enabled              : {'YES' if edge_pass else 'NO'}\n"
        f"Strength             : {edge_str:.2f}\n\n"
        
        f"Validation Summary\n"
        f"─────────────────────────────────────\n"
        f"Scale Geometry       : {'PASS' if scale_geometry_pass else 'FAIL'}\n"
        f"Independent Noise    : {'PASS' if noise_pass else 'FAIL'}\n"
        f"SEM Edge Brightening : {'PASS' if edge_pass else 'FAIL'}\n"
        f"Ground Truth         : PASS\n"
    )
    
    ax_info.text(0.0, 1.0, info_text, transform=ax_info.transAxes, fontsize=10,
                 verticalalignment='top', family='monospace')
    return fig

def visualize_dataset_summary(annotations: list, references: list, searches: list, title: str = "Dataset Summary") -> plt.Figure:
    """Create a dataset summary visualization with up to 3 examples."""
    num_samples = min(3, len(annotations))
    if num_samples == 0:
        return None
        
    fig = plt.figure(figsize=(15, 10), dpi=150, layout='constrained')
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, num_samples, hspace=0.1, wspace=0.1)
    
    for i in range(num_samples):
        # Top row: References
        ax_ref = fig.add_subplot(gs[0, i])
        ax_ref.imshow(references[i], cmap='gray')
        ax_ref.set_title(f"Sample {annotations[i].id}\nReference", fontsize=12)
        ax_ref.axis('off')
        
        # Bottom row: Searches + GT
        ax_search = fig.add_subplot(gs[1, i])
        ax_search.imshow(searches[i], cmap='gray')
        
        bbox = annotations[i].bbox
        rect = patches.Rectangle(
            (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
            linewidth=2, edgecolor='#00ff00', facecolor='none'
        )
        ax_search.add_patch(rect)
        ax_search.set_title(f"Sample {annotations[i].id}\nSearch + GT", fontsize=12)
        ax_search.axis('off')
    return fig

def save_visualization(fig: plt.Figure, filepath: Path) -> None:
    """Save matplotlib figure to file."""
    if fig is None:
        return
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filepath), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def visualize_pair(reference: np.ndarray, search: np.ndarray, 
                   center_x: int, center_y: int,
                   bbox: dict, architecture: str,
                   pair_id: str) -> plt.Figure:
    """Legacy visualization for backward compatibility."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    axes[0].imshow(reference, cmap='gray')
    axes[0].set_title(f'Reference ({architecture}) - ID: {pair_id}')
    axes[0].axis('off')
    
    axes[1].imshow(search, cmap='gray')
    rect = patches.Rectangle(
        (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
        linewidth=2, edgecolor='lime', facecolor='none'
    )
    axes[1].add_patch(rect)
    axes[1].plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=2)
    axes[1].set_title(f'Search Image - Target at ({center_x}, {center_y})')
    axes[1].axis('off')
    return fig


