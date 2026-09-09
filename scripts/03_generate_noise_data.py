"""
03_generate_noise_data.py
Generates the 7 noise types, multi-noise patterns, and paired pixel-level ground truth masks
for train, val, and test splits formatted specifically for U-Net++ segmentation.

Class IDs:
0 = Clean
1 = Gaussian
2 = Salt & Pepper
3 = Speckle
4 = Poisson
5 = Quantization
6 = RVIN
7 = Periodic Digital

Output Structure:
04_split_dataset/ (and mirrored to dataset/)
├── train/
│   ├── images/  (img_00001.npy, ...)
│   └── masks/   (mask_00001.npy, ...)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm

# Import noise generators from local script
from noise_generator import synthesize_multi_noise_sample, CLASS_NAMES

def generate_split_noisy_dataset(preprocessed_dir: str,
                                 manifest_path: str,
                                 output_base_dir: str,
                                 samples_per_clean_image: int = 1,
                                 seed: int = 42):
    """
    Generates paired (image, mask) datasets for each split.
    """
    np.random.seed(seed)
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run 02_split_dataset.py first.")
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    splits = manifest["splits"]
    
    stats = {
        "splits": {},
        "class_distribution_pixels": {str(k): 0 for k in range(8)},
        "dataset_types_count": {"single": 0, "multi": 0, "complex": 0}
    }
    
    for split_name in ["train", "val", "test"]:
        file_list = splits.get(split_name, [])
        if not file_list and split_name == "val":
            file_list = splits.get("validation", [])
            
        print(f"\nProcessing split '{split_name}' with {len(file_list)} base clean images...")
        
        images_out = os.path.join(output_base_dir, split_name, "images")
        masks_out = os.path.join(output_base_dir, split_name, "masks")
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(masks_out, exist_ok=True)
        
        sample_counter = 0
        
        for file_name in tqdm(file_list, desc=f"Generating {split_name} noise"):
            clean_path = os.path.join(preprocessed_dir, file_name)
            if not os.path.exists(clean_path):
                continue
            clean_img = np.load(clean_path).astype(np.float32)
            
            for aug_idx in range(samples_per_clean_image):
                sample_counter += 1
                
                # Balanced cycle across Single, Multi, Complex noise types
                type_selector = ["single", "multi", "complex"][(sample_counter - 1) % 3]
                
                noisy_img, gt_mask, meta = synthesize_multi_noise_sample(clean_img, dataset_type=type_selector)
                
                # Save numpy arrays
                img_name = f"img_{split_name}_{sample_counter:05d}.npy"
                mask_name = f"mask_{split_name}_{sample_counter:05d}.npy"
                
                np.save(os.path.join(images_out, img_name), noisy_img)
                np.save(os.path.join(masks_out, mask_name), gt_mask)
                
                # Update stats
                dtype = meta["dataset_type"]
                stats["dataset_types_count"][dtype] = stats["dataset_types_count"].get(dtype, 0) + 1
                
                unique_classes, counts = np.unique(gt_mask, return_counts=True)
                for u, c in zip(unique_classes, counts):
                    stats["class_distribution_pixels"][str(int(u))] += int(c)
                    
        stats["splits"][split_name] = {
            "num_samples": sample_counter,
            "images_dir": images_out,
            "masks_dir": masks_out
        }
        print(f"Generated {sample_counter} paired samples for '{split_name}'.")

    # Generate dataset_info.json
    dataset_info = {
        "dataset_name": "LungCT_MultiNoise_UNetPlusPlus",
        "image_size": "256x256",
        "channels": 1,
        "dtype_image": "float32 (normalized [0, 1])",
        "dtype_mask": "uint8 (class IDs 0-7)",
        "num_classes": 8,
        "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "split_ratio": manifest["split_ratios"],
        "sample_counts": {k: v["num_samples"] for k, v in stats["splits"].items()},
        "dataset_types_distribution": stats["dataset_types_count"],
        "pixel_class_distribution": {f"{k} ({CLASS_NAMES[int(k)]})": v for k, v in stats["class_distribution_pixels"].items()}
    }
    
    # Save dataset_info.json both in output_base_dir and root
    info_path = os.path.join(output_base_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)
        
    root_info_path = os.path.abspath(os.path.join(output_base_dir, "..", "dataset_info.json"))
    with open(root_info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)
        
    print(f"\n========================================================")
    print(f"Dataset generation complete!")
    print(f"Dataset info saved to: {info_path} & {root_info_path}")
    print(f"========================================================")
    return dataset_info

def main():
    parser = argparse.ArgumentParser(description="Generate 7-noise dataset and pixel masks for U-Net++")
    parser.add_argument("--preprocessed_dir", type=str, default="../02_preprocessed", help="Path to clean .npy files")
    parser.add_argument("--manifest_path", type=str, default="../02_preprocessed/dataset_split_manifest.json", help="Path to split manifest")
    parser.add_argument("--output_dir", type=str, default="../04_split_dataset", help="Output directory for split dataset")
    parser.add_argument("--samples_per_image", type=int, default=1, help="Number of noisy samples per clean image (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, args.preprocessed_dir))
    manifest_path = os.path.abspath(os.path.join(script_dir, args.manifest_path))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    generate_split_noisy_dataset(
        preprocessed_dir=preprocessed_dir,
        manifest_path=manifest_path,
        output_base_dir=output_dir,
        samples_per_clean_image=args.samples_per_image,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
