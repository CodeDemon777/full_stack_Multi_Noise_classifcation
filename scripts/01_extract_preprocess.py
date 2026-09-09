"""
01_extract_preprocess.py
Extracts LoDoPaB-CT HDF5 files, resizes slices to 256x256, ensures float32,
normalizes pixel values to [0, 1], and stores them in 02_preprocessed/.

Usage:
    python 01_extract_preprocess.py --raw_dir ../ground_truth_train --max_slices 1000
    python 01_extract_preprocess.py --raw_dir ../ground_truth_train --all
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import h5py
from tqdm import tqdm

def preprocess_slice(raw_slice: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Preprocesses a single 2D CT slice:
    - Cast to float32
    - Resize/crop to (target_size, target_size)
    - Normalize to range [0.0, 1.0]
    """
    img = raw_slice.astype(np.float32)
    
    # Resize to target dimension
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
    # Normalization to [0, 1]
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img, dtype=np.float32)
        
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def extract_and_preprocess(raw_dir: str, output_dir: str, target_size: int = 256, max_slices: int = None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all hdf5 files
    hdf5_files = sorted(glob.glob(os.path.join(raw_dir, "*.hdf5")))
    if not hdf5_files:
        # Check subdirectories or .npy files
        npy_files = sorted(glob.glob(os.path.join(raw_dir, "*.npy")))
        if npy_files:
            print(f"Found {len(npy_files)} .npy files in {raw_dir}.")
            total_count = len(npy_files) if max_slices is None else min(len(npy_files), max_slices)
            for idx, npy_path in enumerate(tqdm(npy_files[:total_count], desc="Preprocessing .npy")):
                arr = np.load(npy_path)
                processed = preprocess_slice(arr, target_size)
                out_path = os.path.join(output_dir, f"image_{idx+1:05d}.npy")
                np.save(out_path, processed)
            print(f"Successfully preprocessed {total_count} slices into {output_dir}")
            return
        else:
            raise FileNotFoundError(f"No .hdf5 or .npy files found in {raw_dir}")
            
    print(f"Found {len(hdf5_files)} HDF5 files in {raw_dir}")
    
    global_idx = 0
    done = False
    
    with tqdm(total=max_slices if max_slices else 280 * 128, desc="Extracting & Preprocessing") as pbar:
        for h5_file in hdf5_files:
            if done:
                break
            try:
                with h5py.File(h5_file, 'r') as f:
                    # LoDoPaB ground_truth contains key 'data'
                    data_key = 'data' if 'data' in f.keys() else list(f.keys())[0]
                    dataset = f[data_key]
                    num_in_file = dataset.shape[0]
                    
                    for i in range(num_in_file):
                        if max_slices is not None and global_idx >= max_slices:
                            done = True
                            break
                        
                        raw_slice = dataset[i]
                        preprocessed = preprocess_slice(raw_slice, target_size=target_size)
                        
                        global_idx += 1
                        out_name = f"image_{global_idx:05d}.npy"
                        out_path = os.path.join(output_dir, out_name)
                        np.save(out_path, preprocessed)
                        pbar.update(1)
            except Exception as e:
                print(f"Error reading {h5_file}: {e}")
                continue
                
    print(f"Successfully preprocessed {global_idx} clean CT slices saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract and Preprocess LoDoPaB CT slices")
    parser.add_argument("--raw_dir", type=str, default="../../ground_truth_train", help="Path to raw LoDoPaB HDF5 directory")
    parser.add_argument("--output_dir", type=str, default="../02_preprocessed", help="Path to save preprocessed .npy files")
    parser.add_argument("--target_size", type=int, default=256, help="Target image size (default: 256)")
    parser.add_argument("--max_slices", type=int, default=1000, help="Maximum number of slices to extract (default: 1000, use --all for entire dataset)")
    parser.add_argument("--all", action="store_true", help="Process all available slices in raw_dir")
    
    args = parser.parse_args()
    
    # Resolve relative paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(script_dir, args.raw_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    max_slices = None if args.all else args.max_slices
    print(f"Source Directory: {raw_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Image Resolution: {args.target_size}x{args.target_size}")
    print(f"Total slices limit: {'ALL' if max_slices is None else max_slices}")
    
    extract_and_preprocess(raw_dir, output_dir, target_size=args.target_size, max_slices=max_slices)

if __name__ == "__main__":
    main()
