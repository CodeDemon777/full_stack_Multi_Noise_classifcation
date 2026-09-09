"""
noise_generator.py
Mathematical implementations of 7 CT Noise Types & Multi-Noise Spatial Mask Generator
for Multi-Noise Pixel-Level Classification in Lung CT using U-Net++.

Class ID Mapping:
0: Clean
1: Gaussian
2: Salt & Pepper
3: Speckle
4: Poisson
5: Quantization
6: RVIN (Random-Valued Impulse Noise)
7: Periodic Digital
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict

CLASS_NAMES = {
    0: "Clean",
    1: "Gaussian",
    2: "Salt & Pepper",
    3: "Speckle",
    4: "Poisson",
    5: "Quantization",
    6: "RVIN",
    7: "Periodic Digital"
}

# Color palette for 8 classes (RGB in [0, 255])
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

def apply_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma_range: Tuple[float, float] = (0.05, 0.18)) -> np.ndarray:
    """Additive Gaussian noise: I_noisy = I + N(mean, sigma^2)"""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image + gauss
    return np.clip(noisy, 0.0, 1.0)

def apply_salt_and_pepper_noise(image: np.ndarray, amount_range: Tuple[float, float] = (0.03, 0.12), s_vs_p: float = 0.5) -> np.ndarray:
    """Impulse noise: random pixels set to 0 (pepper) or 1 (salt)"""
    amount = np.random.uniform(amount_range[0], amount_range[1])
    noisy = image.copy()
    
    # Salt (1.0)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords_salt = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy[tuple(coords_salt)] = 1.0
    
    # Pepper (0.0)
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords_pepper = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords_pepper)] = 0.0
    
    return noisy

def apply_speckle_noise(image: np.ndarray, sigma_range: Tuple[float, float] = (0.1, 0.35)) -> np.ndarray:
    """Multiplicative Speckle noise: I_noisy = I + I * N(0, sigma^2)"""
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    gauss = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image + image * gauss
    return np.clip(noisy, 0.0, 1.0)

def apply_poisson_noise(image: np.ndarray, peak_range: Tuple[float, float] = (20.0, 80.0)) -> np.ndarray:
    """Poisson (shot) noise: modeled by scaling to photon counts and sampling Poisson distribution"""
    peak = np.random.uniform(peak_range[0], peak_range[1])
    # Avoid zero/negative values for poisson
    scaled = np.maximum(image, 1e-6) * peak
    poisson_sample = np.random.poisson(scaled).astype(np.float32)
    noisy = poisson_sample / peak
    return np.clip(noisy, 0.0, 1.0)

def apply_quantization_noise(image: np.ndarray, levels_choice: List[int] = [4, 8, 12, 16]) -> np.ndarray:
    """Quantization / Bit-depth reduction noise"""
    levels = np.random.choice(levels_choice)
    step = 1.0 / (levels - 1)
    quantized = np.round(image / step) * step
    return np.clip(quantized, 0.0, 1.0)

def apply_rvin_noise(image: np.ndarray, amount_range: Tuple[float, float] = (0.04, 0.15)) -> np.ndarray:
    """Random-Valued Impulse Noise (RVIN): corrupted pixels replaced with uniformly distributed random values in [0, 1]"""
    amount = np.random.uniform(amount_range[0], amount_range[1])
    noisy = image.copy()
    num_corrupt = int(np.ceil(amount * image.size))
    coords = [np.random.randint(0, i, num_corrupt) for i in image.shape]
    random_values = np.random.uniform(0.0, 1.0, num_corrupt).astype(np.float32)
    noisy[tuple(coords)] = random_values
    return noisy

def apply_periodic_digital_noise(image: np.ndarray, 
                                 freq_range: Tuple[float, float] = (0.05, 0.25),
                                 amp_range: Tuple[float, float] = (0.08, 0.22)) -> np.ndarray:
    """Periodic / Digital Pattern interference: I_noisy = I + A * sin(2*pi*(fx*x + fy*y) + phi)"""
    h, w = image.shape[:2]
    fx = np.random.uniform(freq_range[0], freq_range[1]) * np.random.choice([-1, 1])
    fy = np.random.uniform(freq_range[0], freq_range[1]) * np.random.choice([-1, 1])
    amp = np.random.uniform(amp_range[0], amp_range[1])
    phi = np.random.uniform(0, 2 * np.pi)
    
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    periodic = amp * np.sin(2.0 * np.pi * (fx * x_idx + fy * y_idx) + phi)
    noisy = image + periodic
    return np.clip(noisy, 0.0, 1.0)

# Noise dispatcher function
def apply_single_noise(image: np.ndarray, class_id: int) -> np.ndarray:
    """Applies noise corresponding to class_id (1-7) to the entire given image."""
    if class_id == 0:
        return image.copy()
    elif class_id == 1:
        return apply_gaussian_noise(image)
    elif class_id == 2:
        return apply_salt_and_pepper_noise(image)
    elif class_id == 3:
        return apply_speckle_noise(image)
    elif class_id == 4:
        return apply_poisson_noise(image)
    elif class_id == 5:
        return apply_quantization_noise(image)
    elif class_id == 6:
        return apply_rvin_noise(image)
    elif class_id == 7:
        return apply_periodic_digital_noise(image)
    else:
        raise ValueError(f"Unknown class_id: {class_id}")

def generate_spatial_regions(height: int = 256, width: int = 256, num_regions: int = 3, mode: str = "mixed") -> List[np.ndarray]:
    """
    Generates non-overlapping boolean masks for distinct spatial regions.
    Supports random rectangular patches, circular/elliptical regions, and Voronoi/cellular patches.
    """
    h, w = height, width
    region_masks = []
    occupied = np.zeros((h, w), dtype=bool)
    
    if mode == "voronoi":
        # Random Voronoi tessellation
        num_seeds = max(num_regions, 2)
        seed_y = np.random.randint(0, h, num_seeds)
        seed_x = np.random.randint(0, w, num_seeds)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        # Compute distances to all seeds
        dists = np.stack([(y_grid - sy)**2 + (x_grid - sx)**2 for sy, sx in zip(seed_y, seed_x)], axis=0)
        voronoi_map = np.argmin(dists, axis=0)
        
        chosen_labels = np.random.choice(num_seeds, size=min(num_regions, num_seeds), replace=False)
        for lab in chosen_labels:
            mask = (voronoi_map == lab)
            if np.sum(mask) > 100:
                region_masks.append(mask)
        return region_masks

    # Polygonal / Ellipse / Rectangle random patch generation
    attempts = 0
    while len(region_masks) < num_regions and attempts < 40:
        attempts += 1
        shape_type = np.random.choice(["rect", "ellipse", "polygon"])
        candidate = np.zeros((h, w), dtype=np.uint8)
        
        if shape_type == "rect":
            rw = np.random.randint(int(w * 0.2), int(w * 0.55))
            rh = np.random.randint(int(h * 0.2), int(h * 0.55))
            rx = np.random.randint(0, w - rw)
            ry = np.random.randint(0, h - rh)
            candidate[ry:ry+rh, rx:rx+rw] = 1
        elif shape_type == "ellipse":
            cx = np.random.randint(int(w * 0.2), int(w * 0.8))
            cy = np.random.randint(int(h * 0.2), int(h * 0.8))
            ax1 = np.random.randint(int(w * 0.1), int(w * 0.35))
            ax2 = np.random.randint(int(h * 0.1), int(h * 0.35))
            angle = np.random.randint(0, 180)
            cv2.ellipse(candidate, (cx, cy), (ax1, ax2), angle, 0, 360, 1, -1)
        elif shape_type == "polygon":
            num_pts = np.random.randint(3, 7)
            center_x, center_y = np.random.randint(int(w * 0.25), int(w * 0.75)), np.random.randint(int(h * 0.25), int(h * 0.75))
            radius = np.random.randint(int(w * 0.15), int(w * 0.35))
            angles = np.sort(np.random.uniform(0, 2 * np.pi, num_pts))
            pts = []
            for a in angles:
                r = radius * np.random.uniform(0.6, 1.2)
                px = int(np.clip(center_x + r * np.cos(a), 0, w - 1))
                py = int(np.clip(center_y + r * np.sin(a), 0, h - 1))
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(candidate, [pts], 1)
            
        candidate_mask = (candidate == 1)
        # Ensure candidate doesn't overlap heavily with already occupied space
        overlap = candidate_mask & occupied
        valid_pixels = candidate_mask & (~occupied)
        if np.sum(valid_pixels) > 500: # at least 500 pixels
            region_masks.append(valid_pixels)
            occupied = occupied | valid_pixels
            
    return region_masks

def synthesize_multi_noise_sample(clean_image: np.ndarray, 
                                  dataset_type: str = "auto",
                                  clean_background_prob: float = 0.2) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Synthesizes a noisy CT image and its corresponding pixel-level integer ground-truth mask.
    
    dataset_type:
      - "single": exactly 1 noise type (either full image or 1 region)
      - "multi": 2-3 noise types in distinct regions
      - "complex": 4-7 noise types across different regions
      - "auto": randomly picks among single (35%), multi (40%), complex (25%)
      
    Returns:
      (noisy_image [256, 256] float32, mask [256, 256] uint8, metadata dict)
    """
    h, w = clean_image.shape[:2]
    noisy_image = clean_image.copy().astype(np.float32)
    ground_truth_mask = np.zeros((h, w), dtype=np.uint8) # Class 0: Clean
    
    if dataset_type == "auto":
        dataset_type = np.random.choice(["single", "multi", "complex"], p=[0.35, 0.40, 0.25])
        
    applied_classes = []
    
    if dataset_type == "single":
        # Single noise: randomly choose class 1-7
        class_id = int(np.random.randint(1, 8))
        full_coverage = (np.random.rand() > 0.4) # 60% full image, 40% patch
        if full_coverage:
            noisy_image = apply_single_noise(clean_image, class_id)
            ground_truth_mask[:] = class_id
            applied_classes.append(class_id)
        else:
            regions = generate_spatial_regions(h, w, num_regions=1, mode=np.random.choice(["mixed", "voronoi"]))
            if len(regions) > 0:
                region = regions[0]
                full_noisy = apply_single_noise(clean_image, class_id)
                noisy_image[region] = full_noisy[region]
                ground_truth_mask[region] = class_id
                applied_classes.append(class_id)
            else:
                noisy_image = apply_single_noise(clean_image, class_id)
                ground_truth_mask[:] = class_id
                applied_classes.append(class_id)
                
    elif dataset_type == "multi":
        # 2 to 3 distinct noise types
        num_noises = int(np.random.randint(2, 4))
        chosen_classes = list(np.random.choice(range(1, 8), size=num_noises, replace=False))
        regions = generate_spatial_regions(h, w, num_regions=num_noises, mode=np.random.choice(["mixed", "voronoi"]))
        
        for idx, region in enumerate(regions):
            cid = int(chosen_classes[idx])
            full_noisy = apply_single_noise(clean_image, cid)
            noisy_image[region] = full_noisy[region]
            ground_truth_mask[region] = cid
            applied_classes.append(cid)
            
    elif dataset_type == "complex":
        # 4 to 7 noise types
        num_noises = int(np.random.randint(4, 8))
        chosen_classes = list(np.random.choice(range(1, 8), size=num_noises, replace=False))
        regions = generate_spatial_regions(h, w, num_regions=num_noises, mode="voronoi")
        
        # If voronoi generated fewer, fallback to mixed
        if len(regions) < 3:
            regions = generate_spatial_regions(h, w, num_regions=num_noises, mode="mixed")
            
        for idx, region in enumerate(regions):
            if idx < len(chosen_classes):
                cid = int(chosen_classes[idx])
                full_noisy = apply_single_noise(clean_image, cid)
                noisy_image[region] = full_noisy[region]
                ground_truth_mask[region] = cid
                applied_classes.append(cid)

    # Class 0 remains for unassigned pixels
    clean_set = {0} if np.any(ground_truth_mask == 0) else set()
    applied_classes = sorted(list(clean_set.union(set(applied_classes))))
    
    meta = {
        "dataset_type": dataset_type,
        "classes_present": applied_classes,
        "class_names": [CLASS_NAMES[c] for c in applied_classes]
    }
    
    return noisy_image.astype(np.float32), ground_truth_mask.astype(np.uint8), meta

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Converts integer class mask (256, 256) uint8 to RGB color image (256, 256, 3) uint8."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_img[mask == class_id] = color
    return color_img
