"""
Heuristic analysis for texture, compression, and visual artifacts.
Uses Pillow and numpy - no OpenCV dependency.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageStat
from scipy import ndimage, signal
import math

def analyze_heuristics(image_path):
    """
    Analyze image using visual heuristics for AI indicators.
    
    Returns:
        dict: {
            'score': float (0-1, higher means more likely AI),
            'signals': list of detected indicators,
            'details': str explanation
        }
    """
    signals = []
    score = 0.0
    
    try:
        # Load image with Pillow
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to grayscale numpy array for processing
        gray_img = img.convert('L')
        gray_array = np.array(gray_img, dtype=np.float32)
        
        # Texture analysis
        texture_score = analyze_texture_consistency(gray_array)
        if texture_score > 0.6:
            signals.append(f"Inconsistent texture patterns detected (score: {texture_score:.2f})")
            score += 0.4
        elif texture_score > 0.3:
            signals.append("Slight texture irregularities")
            score += 0.2
        
        # Sharpness uniformity
        sharpness_score = analyze_sharpness_uniformity(gray_array)
        if sharpness_score > 0.7:
            signals.append("Unusually uniform sharpness throughout image")
            score += 0.4
        elif sharpness_score > 0.4:
            signals.append("Somewhat uniform sharpness detected")
            score += 0.2
        
        # Compression artifacts (JPEG blocking)
        compression_score = analyze_compression_artifacts(img)
        if compression_score > 0.5:
            signals.append(f"Compression pattern anomalies detected")
            score += 0.3
        
        # Color distribution
        color_score = analyze_color_distribution(img)
        if color_score > 0.6:
            signals.append("Unnatural color distribution patterns")
            score += 0.3
        
        # Edge consistency
        edge_score = analyze_edge_consistency(gray_array)
        if edge_score > 0.6:
            signals.append("Inconsistent edge patterns detected")
            score += 0.3
        
        # Noise analysis
        noise_score = analyze_noise_patterns(gray_array)
        if noise_score > 0.6:
            signals.append("Unusual noise patterns detected")
            score += 0.3
        
    except Exception as e:
        signals.append(f"Heuristic analysis failed: {str(e)[:50]}")
        score += 0.2
    
    # Normalize score to 0-1 range
    final_score = min(max(score, 0.0), 1.0)
    
    # Generate explanation
    if final_score > 0.6:
        details = "Multiple texture and pattern irregularities detected"
    elif final_score > 0.3:
        details = "Some visual inconsistencies found"
    else:
        details = "Visual patterns appear generally natural"
    
    return {
        'score': final_score,
        'signals': signals[:5],
        'details': details
    }

def analyze_texture_consistency(gray_array):
    """Analyze texture consistency across the image using local variance"""
    try:
        # Calculate local variance using convolution
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Mean filter
        mean_filtered = signal.convolve2d(gray_array, kernel, mode='same', boundary='symm')
        
        # Variance = E[X^2] - (E[X])^2
        squared = gray_array ** 2
        mean_squared = signal.convolve2d(squared, kernel, mode='same', boundary='symm')
        local_variance = mean_squared - (mean_filtered ** 2)
        
        # Calculate coefficient of variation
        mean_variance = np.mean(local_variance)
        std_variance = np.std(local_variance)
        
        if mean_variance > 0:
            cv_score = std_variance / mean_variance
            return min(cv_score, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5

def analyze_sharpness_uniformity(gray_array):
    """Analyze if sharpness is unnaturally uniform using gradient magnitude"""
    try:
        # Calculate gradient magnitude using Sobel-like operation
        gy, gx = np.gradient(gray_array)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Divide image into regions
        h, w = gray_array.shape
        regions = []
        
        for i in range(0, h, h//4):
            for j in range(0, w, w//4):
                region = gradient_magnitude[i:min(i+h//4, h), j:min(j+w//4, w)]
                if region.size > 0:
                    regions.append(np.var(region))
        
        if len(regions) > 1:
            # Low variance between regions = uniform sharpness
            uniformity = 1.0 - (np.std(regions) / (np.mean(regions) + 1e-6))
            return min(uniformity, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5

def analyze_compression_artifacts(img):
    """Analyze compression artifacts and blocking (JPEG artifacts)"""
    try:
        # Convert to numpy array
        img_array = np.array(img)
        
        # Check for 8x8 block patterns (JPEG artifacts)
        h, w = img_array.shape[:2]
        block_score = 0.0
        blocks_checked = 0
        
        # Sample blocks from different positions
        for i in range(0, min(h, 128), 8):
            for j in range(0, min(w, 128), 8):
                if i+16 < h and j+16 < w:
                    # Check 8x8 block boundaries
                    block1 = img_array[i:i+8, j:j+8]
                    block2 = img_array[i+8:i+16, j:j+8]
                    
                    if block1.shape[:2] == (8, 8) and block2.shape[:2] == (8, 8):
                        # Calculate boundary difference
                        boundary_diff = np.abs(block1[-1, :] - block2[0, :])
                        if np.mean(boundary_diff) < 10:  # Very similar blocks
                            block_score += 0.1
                    
                    blocks_checked += 1
        
        if blocks_checked > 0:
            return min(block_score / blocks_checked, 1.0)
        
        return 0.5
        
    except Exception:
        return 0.5

def analyze_color_distribution(img):
    """Analyze color distribution for unnatural patterns"""
    try:
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert RGB to HSV manually (more efficient than PIL's convert for analysis)
        img_normalized = img_array / 255.0
        r, g, b = img_normalized[:, :, 0], img_normalized[:, :, 1], img_normalized[:, :, 2]
        
        # Calculate HSV
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        diff = max_rgb - min_rgb
        
        # Saturation
        saturation = np.where(max_rgb == 0, 0, diff / max_rgb)
        
        # Analyze saturation distribution
        sat_mean = np.mean(saturation) * 255
        sat_std = np.std(saturation) * 255
        
        # Unnatural if saturation is too uniform or extreme
        if sat_mean > 200 or sat_mean < 30:
            return 0.7
        elif sat_std < 20:
            return 0.6
        elif sat_std < 40:
            return 0.4
        
        return 0.2
        
    except Exception:
        return 0.5

def analyze_edge_consistency(gray_array):
    """Analyze edge consistency using gradient and Laplacian"""
    try:
        # Calculate gradient
        gy, gx = np.gradient(gray_array)
        gradient = np.sqrt(gx**2 + gy**2)
        
        # Threshold to get edges (using adaptive threshold)
        threshold = np.percentile(gradient, 70)
        edges = gradient > threshold
        
        # Label connected components (edges)
        labeled_edges, num_components = ndimage.label(edges)
        
        if num_components < 5:
            return 0.3
        
        # Analyze edge smoothness by checking component sizes
        component_sizes = []
        for i in range(1, num_components + 1):
            size = np.sum(labeled_edges == i)
            if size > 10:  # Only consider meaningful edges
                component_sizes.append(size)
        
        if len(component_sizes) > 1:
            # High variance in edge sizes = inconsistent
            size_std = np.std(component_sizes)
            size_mean = np.mean(component_sizes)
            
            if size_mean > 0:
                inconsistency = min(size_std / size_mean, 1.0)
                return inconsistency
        
        return 0.5
        
    except Exception:
        return 0.5

def analyze_noise_patterns(gray_array):
    """Analyze noise patterns for synthetic characteristics"""
    try:
        # Apply median filter to estimate noise
        median_filtered = ndimage.median_filter(gray_array, size=3)
        noise = gray_array - median_filtered
        
        # Analyze noise distribution
        noise_std = np.std(noise)
        noise_mean = np.mean(noise)
        
        # Check for unnatural noise patterns
        noise_skewness = stats.skew(noise.flatten())
        noise_kurtosis = stats.kurtosis(noise.flatten())
        
        # Synthetic noise often has different statistical properties
        score = 0.0
        
        if abs(noise_skewness) < 0.5:
            score += 0.2  # Too symmetric (synthetic)
        
        if noise_kurtosis < 1.0:
            score += 0.3  # Too uniform (synthetic)
        
        if noise_std < 2.0:
            score += 0.2  # Too little noise (oversmoothed)
        
        return min(score, 1.0)
        
    except Exception:
        return 0.5

# Import stats for noise analysis
from scipy import stats