"""
Frequency domain analysis using FFT (Fast Fourier Transform).
Uses numpy and scipy - no OpenCV dependency.
Detects synthetic patterns, oversmoothing, and unnatural frequency distributions.
"""

import numpy as np
from scipy import fftpack, stats
from PIL import Image

def analyze_frequency(image_path):
    """
    Analyze image frequency patterns for AI-generation indicators.
    
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
        # Load image with Pillow and convert to grayscale
        img = Image.open(image_path).convert('L')  # 'L' mode = grayscale
        img_array = np.array(img, dtype=np.float32)
        
        # Perform 2D FFT
        fft = fftpack.fft2(img_array)
        fft_shifted = fftpack.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # Add small constant to avoid log(0)
        magnitude_spectrum = np.maximum(magnitude_spectrum, 1e-8)
        
        # Calculate frequency statistics
        magnitude_flat = magnitude_spectrum.flatten()
        
        # Check for unusual high-frequency patterns
        high_freq_ratio = analyze_high_frequencies(magnitude_spectrum)
        
        if high_freq_ratio > 0.7:
            signals.append(f"Unusual high-frequency pattern detected (ratio: {high_freq_ratio:.2f})")
            score += 0.5
        elif high_freq_ratio < 0.3:
            signals.append(f"Limited high-frequency content - possible oversmoothing")
            score += 0.3
        
        # Check for synthetic noise distribution
        noise_uniformity = check_noise_uniformity(magnitude_flat)
        
        if noise_uniformity > 0.6:
            signals.append("Uniform noise distribution - possible synthetic pattern")
            score += 0.4
        
        # Analyze frequency peak distribution
        peak_unusual = analyze_peak_distribution(magnitude_spectrum)
        
        if peak_unusual:
            signals.append("Unusual frequency peak distribution detected")
            score += 0.3
        
        # Detect repetitive patterns (common in AI)
        repetitive_score = detect_repetitive_patterns(magnitude_spectrum)
        
        if repetitive_score > 0.5:
            signals.append(f"Repetitive frequency patterns detected (score: {repetitive_score:.2f})")
            score += 0.4
        
        # Check frequency decay rate (natural vs AI)
        decay_rate = analyze_frequency_decay(magnitude_spectrum)
        if decay_rate < 1.5:
            signals.append(f"Unusual frequency decay pattern (rate: {decay_rate:.2f})")
            score += 0.3
        
    except Exception as e:
        signals.append(f"Frequency analysis failed: {str(e)[:50]}")
        score += 0.2  # Neutral score when analysis fails
    
    # Normalize score to 0-1 range
    final_score = min(max(score, 0.0), 1.0)
    
    # Generate explanation
    if final_score > 0.6:
        details = "Frequency patterns suggest synthetic generation"
    elif final_score > 0.3:
        details = "Mixed frequency characteristics detected"
    else:
        details = "Frequency patterns appear natural"
    
    return {
        'score': final_score,
        'signals': signals[:5],
        'details': details
    }

def analyze_high_frequencies(magnitude_spectrum):
    """Analyze ratio of high-frequency components"""
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Define high-frequency region (edges of spectrum)
    high_freq_mask = np.zeros_like(magnitude_spectrum, dtype=bool)
    high_freq_mask[:h//4, :] = True
    high_freq_mask[3*h//4:, :] = True
    high_freq_mask[:, :w//4] = True
    high_freq_mask[:, 3*w//4:] = True
    
    high_freq_magnitudes = magnitude_spectrum[high_freq_mask]
    all_magnitudes = magnitude_spectrum.flatten()
    
    if len(high_freq_magnitudes) > 0 and len(all_magnitudes) > 0:
        ratio = np.mean(high_freq_magnitudes) / np.mean(all_magnitudes)
        return min(max(ratio, 0.0), 1.0)
    
    return 0.5

def check_noise_uniformity(magnitude_flat):
    """Check if noise distribution is too uniform (synthetic)"""
    # Calculate statistical uniformity using entropy and kurtosis
    try:
        # Normalize magnitudes
        magnitudes_norm = magnitude_flat / (np.sum(magnitude_flat) + 1e-8)
        
        # Calculate entropy (lower entropy = more uniform)
        entropy = -np.sum(magnitudes_norm * np.log2(magnitudes_norm + 1e-8))
        max_entropy = np.log2(len(magnitudes_norm))
        
        if max_entropy > 0:
            uniformity_score = 1.0 - (entropy / max_entropy)
        else:
            uniformity_score = 0.5
        
        # Also check kurtosis (peakiness)
        kurtosis = stats.kurtosis(magnitude_flat)
        
        # Combine metrics
        if kurtosis < 1.0:  # Low kurtosis = more uniform
            uniformity_score = min(uniformity_score + 0.2, 1.0)
        
        return uniformity_score
        
    except Exception:
        return 0.5

def analyze_peak_distribution(magnitude_spectrum):
    """Detect unusual peak patterns in frequency domain"""
    try:
        # Find peaks (values above 95th percentile)
        threshold = np.percentile(magnitude_spectrum, 95)
        peaks = magnitude_spectrum > threshold
        peak_count = np.sum(peaks)
        
        # Expected peaks (about 5% of total)
        expected_peaks = magnitude_spectrum.size * 0.05
        
        # Check if peak count deviates significantly
        if expected_peaks > 0:
            deviation = abs(peak_count - expected_peaks) / expected_peaks
            return deviation > 1.5  # Returns True if unusually many/few peaks
        
        return False
        
    except Exception:
        return False

def detect_repetitive_patterns(magnitude_spectrum):
    """Detect repetitive frequency patterns (grid-like structures)"""
    try:
        # Look for periodic patterns using autocorrelation
        # Normalize spectrum
        spectrum_norm = (magnitude_spectrum - np.mean(magnitude_spectrum)) / (np.std(magnitude_spectrum) + 1e-8)
        
        # Compute 2D autocorrelation using FFT (more efficient)
        fft_spectrum = fftpack.fft2(spectrum_norm)
        power_spectrum = np.abs(fft_spectrum) ** 2
        autocorr = np.real(fftpack.ifft2(power_spectrum))
        
        # Normalize autocorrelation
        autocorr = autocorr / (autocorr[0, 0] + 1e-8)
        
        # Check for strong periodic patterns (high correlation at non-zero shifts)
        center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
        radius = min(autocorr.shape) // 8
        
        # Extract region around center (excluding exact center)
        y_start = max(0, center_y - radius)
        y_end = min(autocorr.shape[0], center_y + radius + 1)
        x_start = max(0, center_x - radius)
        x_end = min(autocorr.shape[1], center_x + radius + 1)
        
        region = autocorr[y_start:y_end, x_start:x_end]
        
        # Mask out the center peak
        center_mask = np.ones_like(region, dtype=bool)
        c_y, c_x = region.shape[0] // 2, region.shape[1] // 2
        center_mask[c_y-1:c_y+2, c_x-1:c_x+2] = False
        
        if np.any(center_mask):
            max_corr = np.max(region[center_mask])
            return min(max_corr, 1.0)
        
        return 0.0
        
    except Exception:
        return 0.0

def analyze_frequency_decay(magnitude_spectrum):
    """Analyze how frequency magnitudes decay (natural vs AI)"""
    try:
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Compute radial frequency distribution
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        radius = radius.astype(int)
        
        # Average magnitude at each radius
        max_radius = min(center_h, center_w)
        radial_profile = []
        
        for r in range(1, max_radius, max_radius // 20):  # Sample every 5% of radius
            mask = (radius >= r) & (radius < r + max_radius // 20)
            if np.any(mask):
                mean_magnitude = np.mean(magnitude_spectrum[mask])
                radial_profile.append(mean_magnitude)
        
        if len(radial_profile) > 5:
            # Check decay rate (should follow power law for natural images)
            x_vals = np.arange(len(radial_profile))
            log_profile = np.log(np.array(radial_profile) + 1e-8)
            
            # Fit linear regression to log-log plot
            slope, _ = np.polyfit(x_vals[1:], log_profile[1:], 1)
            return abs(slope)  # Return decay rate
        
        return 2.0  # Default natural decay rate
        
    except Exception:
        return 2.0