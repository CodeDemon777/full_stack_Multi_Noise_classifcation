# 🏥 LungCT UNet++ — Complete Local PC / Laptop Run Guide & Project Manual

Welcome to the **LungCT UNet++ Multi-Noise Pixel Classification & Clinical Restoration Portal**. This guide provides comprehensive, step-by-step instructions to run, evaluate, and develop this deep learning platform on **Windows**, **macOS**, and **Linux** systems.

---

## ⚡ Quick 1-Click Launch

### Windows:
Double-click `run_app.bat` in the root folder or inside `LungCT_UNetPlusPlus/`.

### macOS / Linux:
```bash
chmod +x run_app.sh
./run_app.sh
```

Then open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

## 💻 Manual Setup Instructions (Step-by-Step)

### Prerequisites:
- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- Minimum 4 GB RAM (Inference uses ~400 MB)
- (Optional) NVIDIA GPU with CUDA 11.8+ or Apple Silicon (M1/M2/M3/M4) for accelerated inference.

---

### 1. Windows (PowerShell / Command Prompt)

```powershell
# 1. Navigate to the project directory
cd d:\Downloads\ct_preprocessing\LungCT_UNetPlusPlus

# 2. Create and activate a Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1
# If on Command Prompt (CMD), run: .venv\Scripts\activate.bat

# 3. Install PyTorch (CPU or CUDA)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU (CUDA 12.1):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Launch the Web Application
python app.py --port 5000
```

---

### 2. macOS (Apple Silicon M1-M4 & Intel)

```bash
# 1. Navigate to project
cd ~/Downloads/ct_preprocessing/LungCT_UNetPlusPlus

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch web application
python3 app.py --port 5000
```
> **Note for Mac users:** PyTorch automatically leverages **Apple Metal Performance Shaders (MPS)** for hardware acceleration on M-series chips.

---

### 3. Linux (Ubuntu / Debian / Fedora)

```bash
# 1. Install prerequisites
sudo apt update && sudo apt install -y python3-venv python3-pip libgl1-mesa-glx

# 2. Navigate and set up environment
cd /path/to/LungCT_UNetPlusPlus
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies & launch
pip install -r requirements.txt
python3 app.py --port 5000
```

---

### 4. Anaconda / Miniconda

```bash
conda create -n lungct_unet python=3.11 -y
conda activate lungct_unet
pip install -r requirements.txt
python app.py
```

---

## 🛠️ CLI Inference & Evaluation Commands

### 1. Single CT Slice Inference:
```bash
python predict.py --input test_images/sample_000.npy --output outputs/
```

### 2. Test Set Evaluation (mIoU, Accuracy, F1, Confusion Matrix):
```bash
python ../LungCT_UNetPlusPlus_Local_Inference/evaluate_test_set.py
```

### 3. Generate Paper Figure 7 (Publication Visuals):
```bash
python ../LungCT_UNetPlusPlus_Local_Inference/generate_paper_figure7.py
```

---

## 📊 Model Specifications & Benchmark Results

| Parameter | Specification |
| :--- | :--- |
| **Model Architecture** | 5-Level Nested U-Net (UNet++) with Dense Skip Pathways |
| **Deep Supervision** | Multi-level intermediate decoder supervision ($L_{DS}$) |
| **Channel Filters** | `[32, 64, 128, 256, 512]` |
| **Total Parameters** | ~9.16 Million (105.0 MB checkpoint) |
| **Input Resolution** | $1 \times 256 \times 256$ float32 normalized $[0, 1]$ |
| **Classes (8)** | Clean (0), Gaussian (1), Salt & Pepper (2), Speckle (3), Poisson (4), Quantization (5), RVIN (6), Periodic Digital (7) |
| **Validation mIoU** | **96.65%** |
| **Pixel Accuracy** | **98.42%** |
| **Trained Checkpoint** | `models/best_unetplusplus.pth` (Epoch 84) |
| **Dataset Source** | LoDoPaB-CT Low-Dose CT Benchmark (35,820 slices) |

---

## ❓ Troubleshooting & FAQs

1. **Port 5000 is already in use:**
   - Run on a custom port: `python app.py --port 8080` (access at `http://localhost:8080`).
2. **Missing Checkpoint (`models/best_unetplusplus.pth`):**
   - Ensure Git LFS pulled the binary weights or copy `best_unetplusplus.pth` (105 MB) into `LungCT_UNetPlusPlus/models/`.
3. **Out of Memory on small devices:**
   - The application is optimized with single-thread CPU execution and immediate GC clearing.

---
