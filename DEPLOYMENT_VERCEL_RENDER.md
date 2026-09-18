# 🚀 Dual Cloud Deployment Guide: Vercel (Frontend) + Render (Backend)

This guide walks you through deploying the **LungCT UNet++ Full-Stack Medical AI Platform** for **100% free hosting** using:
- 🌐 **Vercel** for the responsive Single Page Application (SPA) frontend.
- ⚙️ **Render** for the PyTorch / Flask REST API inference backend (512MB free tier).

---

## 📁 Clean Repository Structure

```
full_stack_Multi_Noise_classifcation/
│
├── 🌐 frontend/                         # VERCEL DEPLOYMENT ROOT
│   ├── index.html                      # Standalone Medical Diagnostic Studio SPA
│   ├── config.js                       # Dynamic API Gateway Resolver (Local & Cloud)
│   └── vercel.json                     # Vercel Routing, Rewrites, and CORS Headers
│
├── ⚙️ backend / core server/
│   ├── app.py                          # Flask REST API v1 Backend with CORS & Telemetry
│   ├── predict.py                      # 5-Level Nested UNet++ Model Architecture
│   ├── build.sh                        # Render Build Script (Git LFS + CPU-wheel install)
│   ├── Procfile                        # Gunicorn 1-worker memory-safe production server
│   ├── render.yaml                     # Render Infrastructure-as-Code Blueprint
│   ├── requirements.txt                # CPU-optimized PyTorch (Render Free Tier 512MB RAM)
│   └── models/
│       └── best_unetplusplus.pth       # Model Checkpoint (105MB, Epoch 84)
│
├── 📂 test_images/                     # Preloaded CT Benchmark Test Slices (.npy)
├── 📂 scripts/                         # LoDoPaB Preprocessing & Noise Generation Pipeline
│
├── 📄 vercel.json                      # Root Vercel config fallback
├── 📄 render.yaml                      # Root Render blueprint fallback
├── 📄 Procfile                         # Root Gunicorn process configuration
├── 📄 run_app.bat                      # 1-Click Local Windows Launcher
├── 📄 run_app.sh                       # 1-Click Local macOS/Linux Launcher
│
└── 📚 docs/
    ├── FULL_STACK_RESUME_GUIDE.md      # Resume Bullet Points & Technical Interview Guide
    ├── LOCAL_RUN_GUIDE.md              # Local PC/Laptop Execution Handbook
    ├── PROJECT_COMPREHENSIVE_REPORT.md # Future Architecture & Clinical Roadmap
    └── DEPLOYMENT_VERCEL_RENDER.md     # This Hosting Handbook
```

---

## ⚙️ PART 1: Deploy Backend on Render (Free Tier)

Render provides free hosting for Python web services (512 MB RAM limit). Our backend is pre-optimized with single-thread PyTorch execution, memory clearing, and `opencv-python-headless` to stay safely below 350 MB RAM.

### Step 1: Create a Render Account
1. Go to **[https://render.com](https://render.com)** and sign in with your GitHub account.

### Step 2: Create a New Web Service
1. In the Render Dashboard, click **New +** $\rightarrow$ **Web Service**.
2. Select **Build and deploy from a Git repository** and connect:
   `https://github.com/CodeDemon777/full_stack_Multi_Noise_classifcation`
3. Configure settings:
   - **Name:** `lungct-unetplusplus-api` (or custom name)
   - **Region:** Choose the region closest to you (e.g., *Frankfurt*, *Oregon*, *Singapore*)
   - **Branch:** `main`
   - **Root Directory:** *(Leave blank or `/`)*
   - **Runtime:** `Python 3`
   - **Build Command:** `./build.sh` (or `pip install -r requirements.txt`)
   - **Start Command:** `gunicorn --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT app:app`
   - **Instance Type:** **Free (512 MB RAM / 0.1 CPU)**

### Step 3: Add Git LFS Pull (Required for 105MB model checkpoint)
Under **Environment Variables**, add:
- `PYTHON_VERSION` = `3.11.8`

Click **Deploy Web Service**. Render will build and launch your backend!

> ⏱️ Once deployed, copy your backend URL:  
> **Example:** `https://lungct-unetplusplus-api.onrender.com`

---

## 🌐 PART 2: Deploy Frontend on Vercel (Free Tier)

Vercel provides blazing-fast global edge hosting for static Single Page Applications.

### Step 1: Create a Vercel Account
1. Go to **[https://vercel.com](https://vercel.com)** and sign in with GitHub.

### Step 2: Import Your Repository
1. In the Vercel Dashboard, click **Add New...** $\rightarrow$ **Project**.
2. Select **full_stack_Multi_Noise_classifcation**.
3. In **Project Settings**:
   - **Framework Preset:** `Other`
   - **Root Directory:** Set to `frontend` *(Click Edit $\rightarrow$ select `frontend`)*
   - **Build & Output Settings:** Default
4. Click **Deploy**.

> ⏱️ Within 10 seconds, your site is live globally on a `.vercel.app` domain!

---

## 🔗 PART 3: Connect Frontend to Backend

1. Open your live Vercel URL in your browser (e.g., `https://lungct-unetplusplus.vercel.app`).
2. At the top of the interface, in the **🌐 Backend Gateway** input bar, enter your Render URL:
   ```
   https://lungct-unetplusplus-api.onrender.com
   ```
3. Click **Ping API**.
4. The status badge will glow **`ONLINE (CPU)`** in emerald green!
5. The URL is automatically saved in browser `localStorage` so it persists on every visit.

---

## ⚡ Free Tier Optimization Highlights

| Cloud Constraint | How It's Solved in This Project |
| :--- | :--- |
| **Render 512MB RAM Ceiling** | `torch.set_num_threads(1)` + `gc.collect()` + `opencv-python-headless` keeps memory ~320MB. |
| **Render Free Tier Spin-Down** | Top gateway banner has a 1-click **Ping API** button to wake up free instances. |
| **CORS Cross-Origin Requests** | Flask backend includes `@app.after_request` headers enabling full cross-origin requests from Vercel. |
| **Dynamic URL Switching** | `config.js` enables 1-click switching between `http://localhost:5000` and Render URLs. |
