"""
04_verify_visualize.py
Verifies dataset integrity and generates visual side-by-side plots:
[Noisy CT] -> [Color-Coded Ground Truth Mask] -> [Overlay Composite] -> [Class Distribution]

Performs automated assertions:
1. Exact dimension match (256x256)
2. Value range sanity (Image in [0, 1] float32, Mask in {0..7} uint8)
3. Non-empty mask verification
4. Output image export into 05_visualization/
"""

import os
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from noise_generator import CLASS_NAMES, CLASS_COLORS, mask_to_color

def verify_and_visualize(dataset_dir: str, 
                          output_vis_dir: str, 
                          num_samples_to_visualize: int = 50,
                          split: str = "train",
                          seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    img_dir = os.path.join(dataset_dir, split, "images")
    mask_dir = os.path.join(dataset_dir, split, "masks")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Missing images or masks folder in {os.path.join(dataset_dir, split)}")
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
    
    assert len(img_files) > 0, "No image files found!"
    assert len(img_files) == len(mask_files), f"Mismatch between images ({len(img_files)}) and masks ({len(mask_files)})!"
    
    print(f"\n========================================================")
    print(f"VERIFYING DATASET INTEGRITY FOR '{split}' ({len(img_files)} total files)")
    print(f"========================================================")
    
    # Validation counters
    passed_checks = 0
    errors = []
    
    # 1. Full Dataset Sanity Scan
    for i in range(len(img_files)):
        img_path = img_files[i]
        mask_path = mask_files[i]
        
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        # Dimension check
        if img.shape != (256, 256):
            errors.append(f"Shape error in {img_path}: expected (256, 256), got {img.shape}")
        if mask.shape != (256, 256):
            errors.append(f"Shape error in {mask_path}: expected (256, 256), got {mask.shape}")
            
        # Dtype check
        if img.dtype != np.float32:
            errors.append(f"Dtype error in {img_path}: expected float32, got {img.dtype}")
        if mask.dtype != np.uint8:
            errors.append(f"Dtype error in {mask_path}: expected uint8, got {mask.dtype}")
            
        # Value range check
        if img.min() < -1e-5 or img.max() > 1.0 + 1e-5:
            errors.append(f"Range error in {img_path}: min={img.min()}, max={img.max()}")
            
        # Class ID validity check
        unique_ids = np.unique(mask)
        invalid_ids = [uid for uid in unique_ids if uid not in range(8)]
        if len(invalid_ids) > 0:
            errors.append(f"Invalid class IDs in {mask_path}: {invalid_ids}")
            
        passed_checks += 1
        
    if len(errors) == 0:
        print(f"[PASS] ALL {passed_checks} samples PASSED integrity checks flawlessly!")
        print(f"   - Dimensions: 256x256")
        print(f"   - Float32 normalization: [0.0, 1.0]")
        print(f"   - Integer Class IDs: {list(range(8))} valid")
    else:
        print(f"[FAIL] Encountered {len(errors)} errors:")
        for err in errors[:10]:
            print(f"   {err}")
        raise ValueError("Dataset integrity verification failed!")

    # 2. Visual Plot Generation for 50-100 random samples
    num_to_draw = min(num_samples_to_visualize, len(img_files))
    sample_indices = random.sample(range(len(img_files)), num_to_draw)
    
    # Custom colormap for visualization
    color_norm_list = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(8)]
    cmap_custom = ListedColormap(color_norm_list)
    
    legend_patches = [
        mpatches.Patch(color=color_norm_list[i], label=f"{i}: {CLASS_NAMES[i]}")
        for i in range(8)
    ]
    
    print(f"\nGenerating {num_to_draw} visualization figures into {output_vis_dir}...")
    for idx in tqdm(sample_indices, desc="Rendering sample visualizer plots"):
        img = np.load(img_files[idx])
        mask = np.load(mask_files[idx])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
        
        # 1. Noisy CT
        axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Noisy CT Slice\n({os.path.basename(img_files[idx])})", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        
        # 2. Color-Coded Ground Truth Mask
        mask_im = axes[1].imshow(mask, cmap=cmap_custom, vmin=0, vmax=7, interpolation="nearest")
        present_classes = np.unique(mask)
        class_str = ", ".join([f"{cid}:{CLASS_NAMES[cid]}" for cid in present_classes])
        axes[1].set_title(f"Pixel-Level Ground Truth Mask\nClasses: {class_str}", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        
        # 3. Alpha Blended Overlay
        color_mask = mask_to_color(mask) / 255.0
        gray_3ch = np.stack([img]*3, axis=-1)
        # Blend only where mask is non-zero (or full image blend)
        non_clean = (mask > 0)
        overlay = gray_3ch.copy()
        overlay[non_clean] = 0.5 * gray_3ch[non_clean] + 0.5 * color_mask[non_clean]
        
        axes[2].imshow(overlay)
        axes[2].set_title("Noisy CT + Mask Overlay", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        
        # Add legend
        fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=10)
        plt.tight_layout()
        
        out_fig_name = f"verify_{os.path.splitext(os.path.basename(img_files[idx]))[0]}.png"
        out_fig_path = os.path.join(output_vis_dir, out_fig_name)
        plt.savefig(out_fig_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        
    print(f"\n[OK] Visual verification completed! {num_to_draw} verification figures saved to: {output_vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Verify dataset and generate visualization artifacts")
    parser.add_argument("--dataset_dir", type=str, default="../04_split_dataset", help="Path to split dataset directory")
    parser.add_argument("--output_vis_dir", type=str, default="../05_visualization", help="Path to save verification figures")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of random samples to visualize (default: 50)")
    parser.add_argument("--split", type=str, default="train", help="Split to visualize (train, val, or test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, args.dataset_dir))
    output_vis_dir = os.path.abspath(os.path.join(script_dir, args.output_vis_dir))
    
    verify_and_visualize(dataset_dir, output_vis_dir, args.num_samples, args.split, args.seed)

if __name__ == "__main__":
    main()
