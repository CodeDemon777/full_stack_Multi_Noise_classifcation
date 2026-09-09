"""
predict.py
Local Inference Script for Multi-Noise Pixel-Level Classification in Lung CT Images
using trained Nested U-Net (UNet++) Checkpoint (best_unetplusplus.pth).

Pipeline:
  CT Image (.npy, .png, .jpg)
        ↓
  Preprocessing (256x256, float32, [0, 1])
        ↓
  UNet++ (8-class pixel prediction)
        ↓
  Argmax Classification & Confidence
        ↓
  Outputs:
    - Predicted Class Mask (.npy, .png)
    - Color-Coded Multi-Noise Map (.png)
    - Overlay on CT slice (.png)
    - Error Map & Accuracy (if Ground Truth mask is provided)
    - 4-Panel / 5-Panel Side-by-Side Summary Figure (.png)
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

# Class definitions
CLASS_NAMES = [
    "Clean",
    "Gaussian",
    "Salt & Pepper",
    "Speckle",
    "Poisson",
    "Quantization",
    "RVIN",
    "Periodic Digital"
]

CLASS_COLORS = {
    0: [30, 30, 30],       # Clean: Dark Charcoal
    1: [230, 25, 75],      # Gaussian: Red
    2: [60, 180, 75],      # Salt & Pepper: Green
    3: [255, 225, 25],     # Speckle: Yellow
    4: [0, 130, 200],      # Poisson: Blue
    5: [245, 130, 48],     # Quantization: Orange
    6: [145, 30, 180],     # RVIN: Purple
    7: [70, 240, 240]      # Periodic: Cyan
}


# ==========================================
# UNet++ (Nested U-Net) Model Architecture
# ==========================================

class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes: int = 8, input_channels: int = 1, deep_supervision: bool = False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


def load_model(checkpoint_path: str, device: torch.device) -> UNetPlusPlus:
    """Loads the UNet++ model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    model = UNetPlusPlus(num_classes=8, input_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_input_image(image_path: str, target_size: int = 256) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Loads and normalizes an input image (.npy, .png, .jpg, etc.)
    Returns (numpy float32 image in [0, 1], torch 1x1xHxW Tensor).
    """
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.npy':
        img = np.load(image_path).astype(np.float32)
        if len(img.shape) == 3:
            img = img[:, :, 0]
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = img.astype(np.float32) / 255.0

    # Resize if needed
    if img.shape[0] != target_size or img.shape[1] != target_size:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] if not already
    min_val, max_val = float(np.min(img)), float(np.max(img))
    if max_val > 1.0 or min_val < 0.0:
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img, dtype=np.float32)
            
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    return img, tensor


def mask_to_color_image(mask: np.ndarray) -> np.ndarray:
    """Converts 2D integer class mask (H, W) uint8 to RGB color image (H, W, 3) uint8."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_img[mask == class_id] = color
    return color_img


def predict_single_image(model: nn.Module, 
                         image_path: str, 
                         device: torch.device, 
                         gt_mask_path: Optional[str] = None,
                         output_dir: str = "predictions",
                         save_visuals: bool = True) -> Dict:
    """
    Runs inference on a single CT image, saves predictions, and optionally evaluates against GT.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img_np, tensor_in = preprocess_input_image(image_path)
    tensor_in = tensor_in.to(device)

    with torch.no_grad():
        logits = model(tensor_in) # (1, 8, H, W)
        probs = torch.softmax(logits, dim=1)
        conf_map, pred_mask = torch.max(probs, dim=1)
        
    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    conf_map_np = conf_map.squeeze().cpu().numpy().astype(np.float32)

    # Make output dirs
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predicted mask as .npy and colored .png
    mask_npy_path = os.path.join(output_dir, f"{base_name}_pred_mask.npy")
    np.save(mask_npy_path, pred_mask_np)

    colored_mask = mask_to_color_image(pred_mask_np)
    colored_png_path = os.path.join(output_dir, f"{base_name}_noise_map.png")
    cv2.imwrite(colored_png_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))

    # Calculate detected classes and distribution
    unique_classes, counts = np.unique(pred_mask_np, return_counts=True)
    total_pixels = pred_mask_np.size
    class_percentages = {
        CLASS_NAMES[int(c)]: round(float(cnt) / total_pixels * 100, 2)
        for c, cnt in zip(unique_classes, counts)
    }

    # Evaluate with Ground Truth if available
    gt_mask_np = None
    pixel_acc = None
    mean_iou = None
    error_map = None
    
    if gt_mask_path and os.path.exists(gt_mask_path):
        if gt_mask_path.endswith('.npy'):
            gt_mask_np = np.load(gt_mask_path).astype(np.uint8)
        else:
            gt_mask_np = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            
        if gt_mask_np.shape != pred_mask_np.shape:
            gt_mask_np = cv2.resize(gt_mask_np, (pred_mask_np.shape[1], pred_mask_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        correct_pixels = np.sum(pred_mask_np == gt_mask_np)
        pixel_acc = float(correct_pixels) / total_pixels * 100.0
        error_map = (pred_mask_np != gt_mask_np).astype(np.uint8) * 255
        
        # Calculate mIoU
        ious = []
        for c in range(8):
            p_c = (pred_mask_np == c)
            g_c = (gt_mask_np == c)
            intersection = np.sum(p_c & g_c)
            union = np.sum(p_c | g_c)
            if union > 0:
                ious.append(intersection / union)
        mean_iou = float(np.mean(ious)) * 100.0 if ious else 100.0

    # Save visual comparison chart
    if save_visuals:
        color_norm_list = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(8)]
        cmap_custom = ListedColormap(color_norm_list)
        legend_patches = [
            mpatches.Patch(color=color_norm_list[i], label=f"{i}: {CLASS_NAMES[i]}")
            for i in range(8)
        ]

        if gt_mask_np is not None:
            fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=120)
            
            # 1. CT Image
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title(f"Input CT Image\n{base_name}", fontsize=11, fontweight="bold")
            axes[0].axis("off")
            
            # 2. Ground Truth Mask
            axes[1].imshow(gt_mask_np, cmap=cmap_custom, vmin=0, vmax=7, interpolation="nearest")
            axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
            axes[1].axis("off")
            
            # 3. Predicted Noise Map
            axes[2].imshow(pred_mask_np, cmap=cmap_custom, vmin=0, vmax=7, interpolation="nearest")
            axes[2].set_title(f"Predicted Noise Map\nAcc: {pixel_acc:.2f}% | mIoU: {mean_iou:.2f}%", fontsize=11, fontweight="bold")
            axes[2].axis("off")
            
            # 4. CT + Prediction Overlay
            gray_3ch = np.stack([img_np]*3, axis=-1)
            pred_color = colored_mask / 255.0
            non_clean = (pred_mask_np > 0)
            overlay = gray_3ch.copy()
            overlay[non_clean] = 0.5 * gray_3ch[non_clean] + 0.5 * pred_color[non_clean]
            axes[3].imshow(overlay)
            axes[3].set_title("CT + Prediction Overlay", fontsize=11, fontweight="bold")
            axes[3].axis("off")
            
            # 5. Error Map
            axes[4].imshow(error_map, cmap='Reds', vmin=0, vmax=255)
            axes[4].set_title(f"Error Map\n({100 - pixel_acc:.2f}% mismatch)", fontsize=11, fontweight="bold")
            axes[4].axis("off")
            
            fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=10)
            plt.tight_layout()
            chart_path = os.path.join(output_dir, f"{base_name}_prediction_eval.png")
            plt.savefig(chart_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=120)
            
            # 1. CT Image
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title(f"Input CT Image\n{base_name}", fontsize=11, fontweight="bold")
            axes[0].axis("off")
            
            # 2. Predicted Noise Map
            axes[1].imshow(pred_mask_np, cmap=cmap_custom, vmin=0, vmax=7, interpolation="nearest")
            axes[1].set_title("Predicted Noise Map (8-Class)", fontsize=11, fontweight="bold")
            axes[1].axis("off")
            
            # 3. CT + Prediction Overlay
            gray_3ch = np.stack([img_np]*3, axis=-1)
            pred_color = colored_mask / 255.0
            non_clean = (pred_mask_np > 0)
            overlay = gray_3ch.copy()
            overlay[non_clean] = 0.5 * gray_3ch[non_clean] + 0.5 * pred_color[non_clean]
            axes[2].imshow(overlay)
            axes[2].set_title("CT + Prediction Overlay", fontsize=11, fontweight="bold")
            axes[2].axis("off")
            
            # 4. Confidence Heatmap
            cax = axes[3].imshow(conf_map_np, cmap='inferno', vmin=0.0, vmax=1.0)
            axes[3].set_title(f"Confidence Heatmap\n(Mean: {conf_map_np.mean():.2f})", fontsize=11, fontweight="bold")
            axes[3].axis("off")
            fig.colorbar(cax, ax=axes[3], fraction=0.046, pad=0.04)
            
            fig.legend(handles=legend_patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=10)
            plt.tight_layout()
            chart_path = os.path.join(output_dir, f"{base_name}_prediction_result.png")
            plt.savefig(chart_path, bbox_inches="tight", dpi=120)
            plt.close(fig)

    result_info = {
        "image": image_path,
        "classes_detected": [CLASS_NAMES[int(c)] for c in unique_classes],
        "class_pixel_percentages": class_percentages,
        "mean_confidence": float(conf_map_np.mean()),
        "pixel_accuracy": pixel_acc,
        "mIoU": mean_iou,
        "outputs": {
            "pred_mask_npy": mask_npy_path,
            "noise_map_png": colored_png_path
        }
    }
    return result_info


def main():
    parser = argparse.ArgumentParser(description="Run local U-Net++ prediction for CT noise classification")
    parser.add_argument("--image", type=str, default=None, help="Path to a single CT image (.npy or .png/.jpg)")
    parser.add_argument("--gt_mask", type=str, default=None, help="Optional path to ground truth mask (.npy or .png)")
    parser.add_argument("--input_dir", type=str, default="test_images", help="Directory containing multiple CT images")
    parser.add_argument("--masks_dir", type=str, default=None, help="Optional directory containing corresponding ground truth masks")
    parser.add_argument("--model_path", type=str, default="models/best_unetplusplus.pth", help="Path to best_unetplusplus.pth")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save prediction results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading UNet++ checkpoint from: {args.model_path}")
    
    model = load_model(args.model_path, device)
    print("[OK] Model loaded successfully!")

    if args.image:
        print(f"\nRunning inference on single image: {args.image}")
        res = predict_single_image(model, args.image, device, gt_mask_path=args.gt_mask, output_dir=args.output_dir)
        print("\n--- Prediction Results ---")
        print(f"  Detected Noises : {', '.join(res['classes_detected'])}")
        print(f"  Mean Confidence : {res['mean_confidence']:.2%}")
        if res['pixel_accuracy'] is not None:
            print(f"  Pixel Accuracy  : {res['pixel_accuracy']:.2f}%")
            print(f"  Mean IoU        : {res['mIoU']:.2f}%")
        print(f"  Saved Outputs   : {args.output_dir}/")
    else:
        # Process multiple images in input_dir
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
            
        extensions = ["*.npy", "*.png", "*.jpg", "*.jpeg", "*.tif"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        img_paths = sorted(img_paths)
        
        if not img_paths:
            raise FileNotFoundError(f"No image files found in {args.input_dir}")
            
        print(f"\nFound {len(img_paths)} images in {args.input_dir}. Running batch inference...")
        
        all_results = []
        acc_list = []
        miou_list = []
        
        for ipath in tqdm(img_paths, desc="Predicting noise maps"):
            bname = os.path.splitext(os.path.basename(ipath))[0]
            gt_path = None
            if args.masks_dir and os.path.exists(args.masks_dir):
                # Look for matching mask file (e.g. mask_test_00001.npy or img_test_00001.npy)
                cand1 = os.path.join(args.masks_dir, f"{bname}.npy")
                cand2 = os.path.join(args.masks_dir, f"{bname.replace('img_', 'mask_')}.npy")
                cand3 = os.path.join(args.masks_dir, f"{bname}.png")
                cand4 = os.path.join(args.masks_dir, f"{bname.replace('img_', 'mask_')}.png")
                for cand in [cand2, cand1, cand4, cand3]:
                    if os.path.exists(cand):
                        gt_path = cand
                        break
                        
            res = predict_single_image(model, ipath, device, gt_mask_path=gt_path, output_dir=args.output_dir)
            all_results.append(res)
            if res['pixel_accuracy'] is not None:
                acc_list.append(res['pixel_accuracy'])
                miou_list.append(res['mIoU'])

        # Save summary report JSON
        summary_path = os.path.join(args.output_dir, "batch_prediction_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "total_images_processed": len(img_paths),
                "average_pixel_accuracy": float(np.mean(acc_list)) if acc_list else None,
                "average_mIoU": float(np.mean(miou_list)) if miou_list else None,
                "individual_results": all_results
            }, f, indent=2)
            
        print(f"\n========================================================")
        print(f"Batch prediction complete! Processed {len(img_paths)} images.")
        if acc_list:
            print(f"  Overall Average Pixel Accuracy : {np.mean(acc_list):.2f}%")
            print(f"  Overall Average mIoU          : {np.mean(miou_list):.2f}%")
        print(f"Results & visualization figures saved to: {args.output_dir}/")
        print(f"Summary JSON saved to: {summary_path}")
        print(f"========================================================")

if __name__ == "__main__":
    main()
