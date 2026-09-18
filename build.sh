#!/usr/bin/env bash
# Render Automated Build Script for LungCT UNet++
set -e

echo "[BUILD 1/3] Updating pip..."
pip install --upgrade pip

echo "[BUILD 2/3] Installing CPU-optimized PyTorch and dependencies..."
pip install -r requirements.txt

echo "[BUILD 3/3] Reassembling model weights if necessary..."
python -c "
import os
part1 = os.path.join('models', 'best_unetplusplus.part1')
part2 = os.path.join('models', 'best_unetplusplus.part2')
out_path = os.path.join('models', 'best_unetplusplus.pth')

if (not os.path.exists(out_path) or os.path.getsize(out_path) < 1000) and os.path.exists(part1) and os.path.exists(part2):
    print('>> Reassembling weights from chunks...')
    with open(out_path, 'wb') as out_f:
        with open(part1, 'rb') as f1: out_f.write(f1.read())
        with open(part2, 'rb') as f2: out_f.write(f2.read())
    print('>> Successfully reassembled', out_path, os.path.getsize(out_path), 'bytes')
"

echo "[BUILD COMPLETE] Ready for high-performance zero-downtime serving!"
