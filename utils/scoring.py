"""
Weighted scoring system for combining analysis signals.
"""

import numpy as np

def calculate_final_score(metadata_result, frequency_result, heuristic_result, weights):
    """
    Calculate final weighted authenticity score.
    
    Args:
        metadata_result: dict with 'score' key
        frequency_result: dict with 'score' key
        heuristic_result: dict with 'score' key
        weights: dict with weight for each component
    
    Returns:
        dict: {
            'final_score': float (0-1, higher = more likely AI),
            'confidence': str (High/Medium/Low),
            'components': dict with individual scores
        }
    """
    # Extract scores
    metadata_score = metadata_result['score']
    frequency_score = frequency_result['score']
    heuristic_score = heuristic_result['score']
    
    # Calculate weighted score
    final_score = (
        weights['metadata'] * metadata_score +
        weights['frequency'] * frequency_score +
        weights['heuristic'] * heuristic_score
    )
    
    # Determine confidence level based on consistency of signals
    scores = [metadata_score, frequency_score, heuristic_score]
    std_dev = np.std(scores)
    
    # Higher consistency = higher confidence
    if std_dev < 0.15:  # Very consistent signals
        confidence = "High"
    elif std_dev < 0.3:  # Moderately consistent
        confidence = "Medium"
    else:  # Conflicting signals
        confidence = "Low"
    
    return {
        'final_score': final_score,
        'confidence': confidence,
        'components': {
            'metadata': metadata_score,
            'frequency': frequency_score,
            'heuristic': heuristic_score
        }
    }

def get_verdict(final_score):
    """
    Convert final score to human-readable verdict.
    """
    if final_score >= 0.7:
        return "Likely AI-Generated"
    elif final_score >= 0.4:
        return "Possibly AI-Generated"
    elif final_score >= 0.2:
        return "Possibly Authentic"
    else:
        return "Likely Authentic"