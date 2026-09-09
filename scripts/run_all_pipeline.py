"""
run_all_pipeline.py
Master end-to-end orchestrator for LungCT_UNetPlusPlus dataset pipeline.

Executes:
Step 1: Folder initialization (01_raw_data, 02_preprocessed, 03_noisy_dataset, 04_split_dataset, 05_visualization, dataset)
Step 2: Raw data verification/linking
Step 3: LoDoPaB HDF5 extraction -> 256x256 float32 normalization
Step 4: 80/10/10 Train/Val/Test splitting BEFORE noise generation
Step 5: 7-class single, multi, and complex multi-noise injection
Step 6: Paired pixel-level ground truth mask generation (classes 0-7)
Step 7: Verification scan & 50-100 visual inspection figure generation
Step 8: dataset_info.json configuration generation and final dataset mirroring
"""

import os
import sys
import shutil
import argparse
import subprocess

def run_command(cmd, cwd=None):
    print(f"\n>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result

def main():
    parser = argparse.ArgumentParser(description="Master LoDoPaB-CT Preprocessing & Multi-Noise Generator")
    parser.add_argument("--raw_dir", type=str, default="../../ground_truth_train", help="Path to raw HDF5 directory")
    parser.add_argument("--max_slices", type=int, default=1000, help="Number of slices to extract and process (default: 1000, use --all for full dataset)")
    parser.add_argument("--all", action="store_true", help="Process ALL available CT slices")
    parser.add_argument("--num_vis", type=int, default=50, help="Number of verification visualization plots to generate (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    # Step 1: Create all project folders
    folders = [
        "01_raw_data",
        "02_preprocessed",
        "03_noisy_dataset",
        "04_split_dataset",
        "05_visualization",
        "dataset",
        "scripts"
    ]
    # Set utf-8 encoding for standard output if needed
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print(f"Creating project directories under: {project_root}")
    for folder in folders:
        fpath = os.path.join(project_root, folder)
        os.makedirs(fpath, exist_ok=True)
        print(f"  [OK] {folder}/")

    python_bin = sys.executable

    # Step 3: Extract & Preprocess
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING & PREPROCESSING RAW LODOPAB-CT SLICES")
    print("="*60)
    extract_cmd = [
        python_bin, os.path.join(script_dir, "01_extract_preprocess.py"),
        "--raw_dir", args.raw_dir,
        "--output_dir", os.path.join(project_root, "02_preprocessed"),
        "--target_size", "256",
        "--max_slices", str(args.max_slices)
    ]
    if args.all:
        extract_cmd.append("--all")
    run_command(extract_cmd, cwd=script_dir)

    # Step 4: Split clean data before noise generation
    print("\n" + "="*60)
    print("STEP 4: SPLITTING CLEAN DATASET (80% TRAIN, 10% VAL, 10% TEST)")
    print("="*60)
    split_cmd = [
        python_bin, os.path.join(script_dir, "02_split_dataset.py"),
        "--input_dir", os.path.join(project_root, "02_preprocessed"),
        "--output_manifest", os.path.join(project_root, "02_preprocessed", "dataset_split_manifest.json"),
        "--train_ratio", "0.8",
        "--val_ratio", "0.1",
        "--test_ratio", "0.1",
        "--seed", str(args.seed)
    ]
    run_command(split_cmd, cwd=script_dir)

    # Steps 5, 6, 7: Generate 7-Class Noise Dataset and Pixel Masks
    print("\n" + "="*60)
    print("STEPS 5-7: GENERATING 7-CLASS NOISE & PIXEL-LEVEL GROUND TRUTH MASKS")
    print("="*60)
    noise_cmd = [
        python_bin, os.path.join(script_dir, "03_generate_noise_data.py"),
        "--preprocessed_dir", os.path.join(project_root, "02_preprocessed"),
        "--manifest_path", os.path.join(project_root, "02_preprocessed", "dataset_split_manifest.json"),
        "--output_dir", os.path.join(project_root, "04_split_dataset"),
        "--samples_per_image", "1",
        "--seed", str(args.seed)
    ]
    run_command(noise_cmd, cwd=script_dir)

    # Mirror / link to final dataset/ folder (Step 9)
    final_dataset_dir = os.path.join(project_root, "dataset")
    split_src_dir = os.path.join(project_root, "04_split_dataset")
    for sname in ["train", "val", "test"]:
        dst_s = os.path.join(final_dataset_dir, sname)
        src_s = os.path.join(split_src_dir, sname)
        if os.path.exists(dst_s):
            shutil.rmtree(dst_s)
        shutil.copytree(src_s, dst_s)
    print(f"\nMirrored dataset to final training directory: {final_dataset_dir}")

    # Step 8: Verification & Visualization
    print("\n" + "="*60)
    print("STEP 8: VERIFYING DATASET INTEGRITY & GENERATING VISUALIZATIONS")
    print("="*60)
    vis_cmd = [
        python_bin, os.path.join(script_dir, "04_verify_visualize.py"),
        "--dataset_dir", os.path.join(project_root, "dataset"),
        "--output_vis_dir", os.path.join(project_root, "05_visualization"),
        "--num_samples", str(args.num_vis),
        "--split", "train",
        "--seed", str(args.seed)
    ]
    run_command(vis_cmd, cwd=script_dir)

    print("\n" + "="*60)
    print("[OK] PART A - LAPTOP PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final Dataset Path : {final_dataset_dir}")
    print(f"Visualizations Path: {os.path.join(project_root, '05_visualization')}")
    print(f"Metadata Info JSON : {os.path.join(project_root, 'dataset_info.json')}")
    print("="*60)

if __name__ == "__main__":
    main()
