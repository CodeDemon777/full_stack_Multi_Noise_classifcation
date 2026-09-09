# Multi-Noise Pixel-Level Classification in Lung CT Images (U-Net++)

This project contains the complete preprocessing and synthetic noise generation pipeline for preparing LoDoPaB-CT lung images for pixel-level noise segmentation using **U-Net++**.

---

## 📁 Directory Hierarchy

```
LungCT_UNetPlusPlus/
│
├── 01_raw_data/              # Raw LoDoPaB-CT HDF5 files or symlinks
├── 02_preprocessed/          # Clean 256x256 float32 normalized CT images (.npy)
│   └── dataset_split_manifest.json
├── 03_noisy_dataset/         # Intermediate/reference noisy data
├── 04_split_dataset/         # Split dataset with paired images & ground truth masks
│   ├── train/ (images/, masks/)
│   ├── val/   (images/, masks/)
│   └── test/  (images/, masks/)
├── 05_visualization/         # Side-by-side visual verification plots (.png)
├── dataset/                  # Mirrored final U-Net++ training dataset
│   ├── train/ (images/, masks/)
│   ├── val/   (images/, masks/)
│   └── test/  (images/, masks/)
├── dataset_info.json         # Dataset metadata and 8-class configuration
└── scripts/
    ├── noise_generator.py       # Math algorithms for all 7 noise types & spatial masks
    ├── 01_extract_preprocess.py # LoDoPaB extraction, resize (256x256), float32, [0, 1]
    ├── 02_split_dataset.py      # 80/10/10 split BEFORE noise injection
    ├── 03_generate_noise_data.py# 7-Class Single, Multi, & Complex Multi-noise generator
    ├── 04_verify_visualize.py   # Integrity verification scan & visualization generator
    └── run_all_pipeline.py      # Master orchestrator script
```

---

## 🏷️ Class ID Mapping

The ground-truth mask is a 2D integer matrix of size $256 \times 256$ with `uint8` values:

| Class ID | Noise Type | Description |
| :--- | :--- | :--- |
| **0** | **Clean** | Original uncorrupted CT attenuation values |
| **1** | **Gaussian** | Additive Gaussian noise: $I_{noisy} = I + \mathcal{N}(0, \sigma^2)$ |
| **2** | **Salt & Pepper** | Impulse noise: random pixels set to $0$ (pepper) or $1$ (salt) |
| **3** | **Speckle** | Multiplicative noise: $I_{noisy} = I + I \cdot \mathcal{N}(0, \sigma^2)$ |
| **4** | **Poisson** | Shot noise modeled by Poisson distribution: $\text{Poisson}(I \cdot \lambda) / \lambda$ |
| **5** | **Quantization** | Bit-depth reduction / step quantization to discrete steps |
| **6** | **RVIN** | Random-Valued Impulse Noise: pixels replaced with uniform random values in $[0, 1]$ |
| **7** | **Periodic Digital** | Sinusoidal interference: $I_{noisy} = I + A \sin(2\pi(f_x x + f_y y) + \phi)$ |

---

## 🚀 Execution Guide

### 1. Run Complete Pipeline in One Command
```bash
cd scripts
python run_all_pipeline.py --max_slices 1000 --num_vis 50
```
> Use `--all` to process all 35,000+ available CT slices in the dataset.

### 2. Run Step-by-Step

#### Step 3: Extract & Preprocess
```bash
python scripts/01_extract_preprocess.py --raw_dir ../../ground_truth_train --output_dir ../02_preprocessed --max_slices 1000
```

#### Step 4: Split Clean CT Slices (80% / 10% / 10%)
```bash
python scripts/02_split_dataset.py --input_dir ../02_preprocessed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

#### Steps 5-7: Generate Multi-Noise Images & Ground Truth Masks
```bash
python scripts/03_generate_noise_data.py --preprocessed_dir ../02_preprocessed --manifest_path ../02_preprocessed/dataset_split_manifest.json --output_dir ../dataset
```

#### Step 8: Verify Dataset & Generate Inspection Plots
```bash
python scripts/04_verify_visualize.py --dataset_dir ../dataset --output_vis_dir ../05_visualization --num_samples 50 --split train
```
