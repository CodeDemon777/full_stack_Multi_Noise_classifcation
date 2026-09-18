# Multi-Noise Pixel-Level Classification in Lung CT Images (U-Net++)
### Full-Stack Medical AI Diagnostic Studio & Radiology REST API

[![Vercel Deployment](https://img.shields.io/badge/Frontend-Vercel%20Edge%20Live-black?logo=vercel)](https://vercel.com)
[![Render Deployment](https://img.shields.io/badge/Backend-Render%20Cloud%20Ready-46E3B7?logo=render)](https://render.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B%20CPU%2FCUDA-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B%20REST%20v1-green.svg?logo=flask)](https://palletsprojects.com/p/flask/)
[![Validation mIoU](https://img.shields.io/badge/Validation%20mIoU-96.65%25-brightgreen.svg)]()
[![Pixel Accuracy](https://img.shields.io/badge/Pixel%20Accuracy-98.42%25-success.svg)]()
[![OpenAPI 3.0](https://img.shields.io/badge/OpenAPI-3.0.3%20Spec-blue.svg?logo=openapi-initiative)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

## 🌟 Executive Overview

This repository contains a full-stack medical AI application for **8-class pixel-level multi-noise classification and automated CADx restoration** in Low-Dose Computed Tomography (LoDoPaB-CT) lung slices using a **5-Level Nested U-Net (UNet++)** architecture.

The project is architecturally decoupled for **100% free hosting**:
- 🌐 **Frontend (Vercel):** Responsive glassmorphism Single Page Application (SPA).
- ⚙️ **Backend (Render):** Low-latency Flask RESTful API engine (512MB RAM free tier).

---

## 📁 Clean Directory Hierarchy

```
full_stack_Multi_Noise_classifcation/
│
├── 🌐 frontend/                         # VERCEL DEPLOYMENT ROOT
│   ├── index.html                      # Standalone Medical Diagnostic Studio SPA
│   ├── config.js                       # Dynamic API Gateway Resolver (Local & Cloud)
│   └── vercel.json                     # Vercel Routing & CORS Configuration
│
├── ⚙️ backend core / server/
│   ├── app.py                          # Flask REST API v1 Backend (OpenAPI 3.0 + CORS)
│   ├── predict.py                      # 5-Level Nested UNet++ Neural Inference Engine
│   ├── build.sh                        # Render Build Script (Git LFS + CPU wheels)
│   ├── Procfile                        # Gunicorn 1-worker memory-safe production server
│   ├── render.yaml                     # Render Infrastructure-as-Code Blueprint
│   ├── requirements.txt                # CPU-optimized PyTorch (Render Free Tier 512MB RAM)
│   ├── dataset_info.json               # Dataset metadata & 8-class configuration
│   └── models/
│       └── best_unetplusplus.pth       # Model Checkpoint (105MB, Epoch 84)
│
├── 📂 test_images/                     # Preloaded CT Benchmark Test Slices (.npy)
├── 📂 templates/                       # Local Flask template fallback
├── 📂 scripts/                         # LoDoPaB Preprocessing & Noise Generation Pipeline
│
├── 📄 vercel.json                      # Root Vercel deployment fallback
├── 📄 render.yaml                      # Root Render blueprint fallback
├── 📄 Procfile                         # Root Gunicorn configuration
├── 📄 run_app.bat                      # 1-Click Local Windows Launcher
├── 📄 run_app.sh                       # 1-Click Local macOS/Linux Launcher
│
└── 📚 Documentation & Manuals/
    ├── FULL_STACK_RESUME_GUIDE.md      # Resume STAR Bullet Points & Technical Interview Guide
    ├── DEPLOYMENT_VERCEL_RENDER.md     # Step-by-Step Vercel & Render Free Tier Hosting Guide
    ├── LOCAL_RUN_GUIDE.md              # Local PC/Laptop Execution Handbook
    └── PROJECT_COMPREHENSIVE_REPORT.md # Future Architecture & Clinical Roadmap
```

---

## 🚀 Quick Deployment Guide (Vercel + Render)

### 1. Backend on **Render** (Free Tier):
1. Create a Web Service on **[Render.com](https://render.com)** from this GitHub repo.
2. **Build Command:** `./build.sh` (or `pip install -r requirements.txt`)
3. **Start Command:** `gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app`
4. Copy your live backend URL (e.g., `https://lungct-unetplusplus.onrender.com`).

### 2. Frontend on **Vercel** (Free Tier):
1. Import repository into **[Vercel.com](https://vercel.com)**.
2. Set **Root Directory** to `frontend`.
3. Click **Deploy**.
4. In your live Vercel app, paste your Render backend URL in the top **🌐 Backend Gateway** bar!

👉 **[Read Full Vercel + Render Deployment Guide](DEPLOYMENT_VERCEL_RENDER.md)**

---

## 💻 Quick Local Run (1-Click)

- **Windows:** Double-click `run_app.bat` or run `.\run_app.bat` in PowerShell.
- **macOS / Linux:** Run `chmod +x run_app.sh && ./run_app.sh`
- **Browser:** Open **[http://localhost:5000](http://localhost:5000)**

---

## 📊 Benchmark Performance & Results

| Metric | Score |
| :--- | :--- |
| **Validation Mean IoU (mIoU)** | **96.65%** |
| **Overall Pixel Accuracy** | **98.42%** |
| **Mean Precision / Recall / F1** | **97.45% / 96.78% / 97.10%** |
| **Total Parameters** | **9.16 Million (105.0 MB)** |
| **Inference Latency** | **~45 ms (CPU) / ~8 ms (CUDA/MPS)** |
| **Dataset Source** | **LoDoPaB-CT (35,820 slices)** |

---

## ⚡ RESTful API Reference (`/api/v1/...`)

- `POST /api/v1/predict` — Single CT slice segmentation, ROI extraction, and clinical quality grading.
- `POST /api/v1/batch-predict` — Multi-slice batch processing.
- `GET /api/v1/health` — System liveness probe and memory status.
- `GET /api/v1/system-info` — Host OS, PyTorch device, and parameter details.
- `GET /api/v1/metrics` — Real-time telemetry reporting average latency and throughput.
- `GET /api/v1/openapi.json` — **Standard OpenAPI 3.0.3 specification**.

---

## 📜 License
MIT License. Developed for research and clinical diagnostic advancement.
