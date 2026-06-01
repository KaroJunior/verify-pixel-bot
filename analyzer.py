"""
Main analysis orchestrator that combines all analysis modules.
"""

from utils import metadata, frequency, heuristics, scoring
from config import WEIGHTS
import os

class ImageAnalyzer:
    """Main class for analyzing images"""
    
    def __init__(self):
        self.weights = WEIGHTS
    
    async def analyze(self, image_path):
        """
        Complete analysis pipeline for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Complete analysis results
        """
        # Run all analyses
        metadata_result = metadata.analyze_metadata(image_path)
        frequency_result = frequency.analyze_frequency(image_path)
        heuristic_result = heuristics.analyze_heuristics(image_path)
        
        # Calculate final score
        final_result = scoring.calculate_final_score(
            metadata_result,
            frequency_result,
            heuristic_result,
            self.weights
        )
        
        # Get human-readable verdict
        verdict = scoring.get_verdict(final_result['final_score'])
        
        # Collect all signals
        all_signals = []
        all_signals.extend(metadata_result.get('signals', []))
        all_signals.extend(frequency_result.get('signals', []))
        all_signals.extend(heuristic_result.get('signals', []))
        
        # Remove duplicates (by converting to set and back)
        unique_signals = list(dict.fromkeys(all_signals))
        
        # Prepare final result
        result = {
            'verdict': verdict,
            'ai_probability': round(final_result['final_score'] * 100, 1),
            'confidence': final_result['confidence'],
            'signals': unique_signals[:5],  # Top 5 signals
            'details': {
                'metadata': metadata_result['details'],
                'frequency': frequency_result['details'],
                'heuristic': heuristic_result['details']
            },
            'scores': final_result['components']
        }
        
        return result