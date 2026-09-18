#!/usr/bin/env bash
# Render Build Script for LungCT UNet++ Backend
set -e

echo "[BUILD 1/3] Updating pip..."
pip install --upgrade pip

echo "[BUILD 2/3] Pulling Git LFS binary model weights..."
git lfs install || true
git lfs pull || true

echo "[BUILD 3/3] Installing CPU-optimized PyTorch and dependencies..."
pip install -r requirements.txt

echo "[BUILD COMPLETE] Ready for deployment!"
