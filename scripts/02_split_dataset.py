"""
02_split_dataset.py
Splits preprocessed clean CT images into train (80%), val (10%), test (10%) sets
BEFORE noise injection to strictly prevent data leakage between splits.

Usage:
    python 02_split_dataset.py --input_dir ../02_preprocessed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

import os
import glob
import json
import random
import argparse
import numpy as np

def split_preprocessed_data(input_dir: str, 
                            output_manifest: str = None,
                            train_ratio: float = 0.8, 
                            val_ratio: float = 0.1, 
                            test_ratio: float = 0.1,
                            seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    # Assert ratios sum to ~1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
    
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    total_files = len(npy_files)
    
    if total_files == 0:
        raise FileNotFoundError(f"No .npy files found in {input_dir}. Please run 01_extract_preprocess.py first.")
        
    print(f"Total clean CT images found in {input_dir}: {total_files}")
    
    # Shuffle file list with fixed seed
    shuffled_files = [os.path.basename(f) for f in npy_files]
    random.shuffle(shuffled_files)
    
    n_train = int(total_files * train_ratio)
    n_val = int(total_files * val_ratio)
    n_test = total_files - (n_train + n_val)
    
    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train:n_train + n_val]
    test_files = shuffled_files[n_train + n_val:]
    
    split_info = {
        "seed": seed,
        "total_images": total_files,
        "split_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio
        },
        "counts": {
            "train": len(train_files),
            "val": len(val_files),
            "validation": len(val_files),
            "test": len(test_files)
        },
        "splits": {
            "train": train_files,
            "val": val_files,
            "validation": val_files,
            "test": test_files
        }
    }
    
    if output_manifest is None:
        output_manifest = os.path.join(input_dir, "dataset_split_manifest.json")
        
    with open(output_manifest, "w") as f:
        json.dump(split_info, f, indent=2)
        
    print(f"\n--- Split Summary ---")
    print(f"  Train : {len(train_files)} images ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Val   : {len(val_files)} images ({len(val_files)/total_files*100:.1f}%)")
    print(f"  Test  : {len(test_files)} images ({len(test_files)/total_files*100:.1f}%)")
    print(f"Split manifest saved to: {output_manifest}")
    return split_info

def main():
    parser = argparse.ArgumentParser(description="Split preprocessed CT images before noise generation")
    parser.add_argument("--input_dir", type=str, default="../02_preprocessed", help="Path to preprocessed .npy directory")
    parser.add_argument("--output_manifest", type=str, default=None, help="Path to save split json manifest")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, args.input_dir))
    output_manifest = os.path.abspath(os.path.join(script_dir, args.output_manifest)) if args.output_manifest else None
    
    split_preprocessed_data(input_dir, output_manifest, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()
