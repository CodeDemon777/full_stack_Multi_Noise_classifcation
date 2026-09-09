"""
app.py - Interactive Web Application for Multi-Noise Pixel-Level & Bounding Box Lung CT Segmentation (U-Net++)
"""

import os
import io
import time
import base64
import json
import glob
from typing import Tuple, Dict, List, Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, send_file

from predict import UNetPlusPlus, load_model, CLASS_NAMES, CLASS_COLORS, mask_to_color_image

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_unetplusplus.pth")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
TEST_MASKS_DIR = os.path.join(BASE_DIR, "dataset", "test", "masks")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Determine device and load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Loading UNet++ on device: {DEVICE}")
model = load_model(MODEL_PATH, DEVICE)
print("[INIT] UNet++ model loaded and ready!")

def numpy_to_base64_png(img_uint8: np.ndarray, is_bgr: bool = False) -> str:
    """Encodes a uint8 numpy image (gray or RGB) to a base64 PNG data URI string."""
    if is_bgr:
        rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
    else:
        rgb = img_uint8
    success, encoded = cv2.imencode('.png', rgb if len(rgb.shape) == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        return ""
    b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def generate_confidence_heatmap(conf_map: np.ndarray) -> str:
    """Generates an Inferno colored heatmap from 0.0-1.0 confidence map."""
    cmap = plt.get_cmap('inferno')
    colored = (cmap(conf_map)[:, :, :3] * 255).astype(np.uint8)
    return numpy_to_base64_png(colored)

def extract_bounding_boxes(mask: np.ndarray, conf_map: np.ndarray, min_area: int = 80) -> List[Dict]:
    """Extracts bounding boxes and metadata for each noise region."""
    boxes = []
    box_id = 1
    h, w = mask.shape
    total_px = mask.size

    for cid in range(1, 8):
        bin_mask = (mask == cid).astype(np.uint8) * 255
        # Morphological close to bridge small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area >= min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Compute average confidence inside this contour/box
                box_region_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(box_region_mask, [cnt], -1, 1, -1)
                region_conf = float(np.mean(conf_map[box_region_mask == 1])) if np.sum(box_region_mask) > 0 else 0.95
                
                boxes.append({
                    "id": box_id,
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "color": CLASS_COLORS[cid],
                    "box": [int(x), int(y), int(bw), int(bh)],
                    "area_pixels": int(area),
                    "area_percentage": round(area / total_px * 100, 2),
                    "confidence": round(region_conf * 100, 1)
                })
                box_id += 1

    # Sort boxes by area descending
    boxes = sorted(boxes, key=lambda b: b['area_pixels'], reverse=True)
    return boxes

def generate_bbox_overlay_image(ct_uint8: np.ndarray, boxes: List[Dict], mask_np: np.ndarray, include_mask_blend: bool = False) -> np.ndarray:
    """Draws bounding boxes and labels on CT image."""
    h, w = ct_uint8.shape[:2]
    canvas = np.stack([ct_uint8]*3, axis=-1).copy()

    if include_mask_blend:
        color_mask = mask_to_color_image(mask_np)
        non_clean = (mask_np > 0)
        canvas[non_clean] = (0.55 * canvas[non_clean] + 0.45 * color_mask[non_clean]).astype(np.uint8)

    for b in boxes:
        cid = b["class_id"]
        col = tuple(CLASS_COLORS[cid]) # RGB
        # Draw on RGB canvas
        x, y, bw, bh = b["box"]
        # Rectangle border
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), col, 2)
        
        # Corner brackets for premium medical HUD look
        line_len = min(12, bw // 3, bh // 3)
        cv2.line(canvas, (x, y), (x + line_len, y), (255, 255, 255), 2)
        cv2.line(canvas, (x, y), (x, y + line_len), (255, 255, 255), 2)
        cv2.line(canvas, (x + bw, y + bh), (x + bw - line_len, y + bh), (255, 255, 255), 2)
        cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - line_len), (255, 255, 255), 2)

        # Label tag background
        label = f"{b['class_name']} {b['confidence']}%"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tag_y = max(y - 6, lh + 4)
        cv2.rectangle(canvas, (x, tag_y - lh - 4), (x + lw + 6, tag_y + baseline), col, -1)
        cv2.putText(canvas, label, (x + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0) if sum(col) > 400 else (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

def preprocess_image_bytes(file_bytes: bytes, filename: str, target_size: int = 256) -> Tuple[np.ndarray, torch.Tensor]:
    """Preprocesses raw bytes of .npy or image file to normalized [0, 1] 256x256 tensor."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.npy':
        with io.BytesIO(file_bytes) as f:
            arr = np.load(f).astype(np.float32)
            if len(arr.shape) == 3:
                arr = arr[:, :, 0]
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError("Invalid image file format")
        arr = arr.astype(np.float32) / 255.0

    # Resize to 256x256
    if arr.shape[0] != target_size or arr.shape[1] != target_size:
        arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)

    min_val, max_val = float(np.min(arr)), float(np.max(arr))
    if max_val > 1.0 or min_val < 0.0:
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    img_np = np.clip(arr, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_np, tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples', methods=['GET'])
def get_samples():
    """Returns list of preloaded sample test CT images."""
    if not os.path.exists(TEST_IMAGES_DIR):
        return jsonify({"samples": []})
    
    files = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.npy")) + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png")))
    sample_list = []
    
    for fpath in files:
        bname = os.path.basename(fpath)
        mask_name = bname.replace("img_", "mask_")
        has_gt = os.path.exists(os.path.join(TEST_MASKS_DIR, mask_name)) or os.path.exists(os.path.join(TEST_MASKS_DIR, bname))
        
        sample_list.append({
            "filename": bname,
            "has_gt": has_gt
        })
        
    return jsonify({"samples": sample_list, "classes": CLASS_NAMES, "colors": CLASS_COLORS})

@app.route('/api/predict', methods=['POST'])
def run_prediction():
    """Runs U-Net++ pixel-level & bounding-box segmentation."""
    start_time = time.time()
    sample_filename = request.form.get('sample_filename')
    gt_mask_np = None
    
    if sample_filename:
        img_path = os.path.join(TEST_IMAGES_DIR, sample_filename)
        if not os.path.exists(img_path):
            return jsonify({"error": f"Sample file not found: {sample_filename}"}), 404
        with open(img_path, 'rb') as f:
            file_bytes = f.read()
        filename = sample_filename
        
        mask_name = sample_filename.replace("img_", "mask_")
        cand1 = os.path.join(TEST_MASKS_DIR, mask_name)
        cand2 = os.path.join(TEST_MASKS_DIR, sample_filename)
        for cand in [cand1, cand2]:
            if os.path.exists(cand):
                gt_mask_np = np.load(cand).astype(np.uint8) if cand.endswith('.npy') else cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
                break
    else:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        file_bytes = file.read()
        filename = file.filename

    try:
        img_np, tensor_in = preprocess_image_bytes(file_bytes, filename)
    except Exception as e:
        return jsonify({"error": f"Error preprocessing image: {str(e)}"}), 400

    # Model inference
    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1)
        conf_map, pred_mask = torch.max(probs, dim=1)

    inference_ms = round((time.time() - start_time) * 1000, 1)

    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    conf_map_np = conf_map.squeeze().cpu().numpy().astype(np.float32)

    # Save predicted mask to predictions folder
    base_id = os.path.splitext(filename)[0]
    out_mask_path = os.path.join(PREDICTIONS_DIR, f"{base_id}_pred_mask.npy")
    np.save(out_mask_path, pred_mask_np)

    # 1. CT Grayscale Base64
    ct_uint8 = (img_np * 255.0).astype(np.uint8)
    ct_b64 = numpy_to_base64_png(ct_uint8)

    # 2. Pixel-Level Noise Map (Color-coded)
    color_mask = mask_to_color_image(pred_mask_np)
    noise_map_b64 = numpy_to_base64_png(color_mask)

    # 3. CT + Prediction Overlay
    gray_3ch = np.stack([ct_uint8]*3, axis=-1)
    non_clean = (pred_mask_np > 0)
    overlay = gray_3ch.copy()
    overlay[non_clean] = (0.5 * gray_3ch[non_clean] + 0.5 * color_mask[non_clean]).astype(np.uint8)
    overlay_b64 = numpy_to_base64_png(overlay)

    # 4. Confidence Heatmap
    confidence_b64 = generate_confidence_heatmap(conf_map_np)

    # 5. Bounding Box Feature Extraction
    bounding_boxes = extract_bounding_boxes(pred_mask_np, conf_map_np)
    bbox_only_img = generate_bbox_overlay_image(ct_uint8, bounding_boxes, pred_mask_np, include_mask_blend=False)
    bbox_b64 = numpy_to_base64_png(bbox_only_img)

    # 6. Hybrid View (Pixel-Level Mask + Bounding Boxes)
    hybrid_img = generate_bbox_overlay_image(ct_uint8, bounding_boxes, pred_mask_np, include_mask_blend=True)
    hybrid_b64 = numpy_to_base64_png(hybrid_img)

    # Class distribution
    unique_classes, counts = np.unique(pred_mask_np, return_counts=True)
    total_px = pred_mask_np.size
    class_stats = []
    for c, cnt in zip(unique_classes, counts):
        cid = int(c)
        class_stats.append({
            "class_id": cid,
            "name": CLASS_NAMES[cid],
            "color": CLASS_COLORS[cid],
            "percentage": round(float(cnt) / total_px * 100, 2),
            "pixels": int(cnt)
        })

    # Optional Ground Truth Evaluation
    gt_stats = None
    if gt_mask_np is not None:
        if gt_mask_np.shape != pred_mask_np.shape:
            gt_mask_np = cv2.resize(gt_mask_np, (pred_mask_np.shape[1], pred_mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        correct_px = int(np.sum(pred_mask_np == gt_mask_np))
        pixel_acc = round(float(correct_px) / total_px * 100.0, 2)
        
        ious = []
        for c in range(8):
            p_c = (pred_mask_np == c)
            g_c = (gt_mask_np == c)
            intersection = np.sum(p_c & g_c)
            union = np.sum(p_c | g_c)
            if union > 0:
                ious.append(intersection / union)
        miou = round(float(np.mean(ious)) * 100.0, 2) if ious else 100.0
        
        gt_color = mask_to_color_image(gt_mask_np)
        gt_b64 = numpy_to_base64_png(gt_color)
        
        error_map = np.zeros((256, 256, 3), dtype=np.uint8)
        mismatched = (pred_mask_np != gt_mask_np)
        error_map[mismatched] = [230, 25, 75]
        error_map_b64 = numpy_to_base64_png(error_map)

        gt_stats = {
            "gt_mask_b64": gt_b64,
            "error_map_b64": error_map_b64,
            "pixel_accuracy": pixel_acc,
            "miou": miou
        }

    response_data = {
        "filename": filename,
        "inference_ms": inference_ms,
        "mean_confidence": round(float(conf_map_np.mean()) * 100, 2),
        "ct_image_b64": ct_b64,
        "noise_map_b64": noise_map_b64,
        "overlay_b64": overlay_b64,
        "confidence_b64": confidence_b64,
        "bbox_b64": bbox_b64,
        "hybrid_b64": hybrid_b64,
        "bounding_boxes": bounding_boxes,
        "class_stats": class_stats,
        "gt_stats": gt_stats,
        "download_mask_url": f"/api/download-mask/{base_id}_pred_mask.npy"
    }
    return jsonify(response_data)

@app.route('/api/download-mask/<filename>', methods=['GET'])
def download_mask(filename):
    fpath = os.path.join(PREDICTIONS_DIR, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "File not found"}), 404
    return send_file(fpath, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*65)
    print(f">> LungCT U-Net++ Web Application running at: http://localhost:{port}")
    print(f"   Model Checkpoint : {MODEL_PATH}")
    print(f"   Device           : {DEVICE}")
    print("="*65 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
