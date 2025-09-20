#!/usr/bin/env python3
"""
Detailed bubble intensity analysis to understand detection errors.
This will examine the actual pixel intensities at detected bubble positions.
"""

import cv2
import numpy as np
import os
from utils.bubbledetection import extract_responses
import pandas as pd

def analyze_bubble_intensities():
    # Load the image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print("=== BUBBLE INTENSITY ANALYSIS ===\n")
    print(f"Image loaded: {image.shape}")
    
    # Get bubble detection results
    results = extract_responses(image)
    print(f"Total results: {len(results)}")
    
    # Answer key for the 6 actual questions
    answer_key = {
        'Python': ['A', 'C', 'C', 'C', 'C', 'A'],
        'EDA': ['A', 'D', 'B', 'A', 'C', 'B'],
        'SQL': ['C', 'C', 'C', 'B', 'B', 'A'],
        'POWER BI': ['B', 'C', 'A', 'B', 'C', 'B'],
        'Satistics': ['A', 'B', 'C', 'B', 'C', 'B']
    }
    
    print("\n=== INTENSITY ANALYSIS BY QUESTION ===")
    
    subjects = ['Python', 'EDA', 'SQL', 'POWER BI', 'Satistics']
    
    for subject_idx, subject in enumerate(subjects):
        print(f"\n{subject}:")
        print("="*60)
        
        subject_results = [r for r in results if r['subject'] == subject]
        
        for q_num in range(1, 7):  # Questions 1-6
            question_results = [r for r in subject_results if r['question'] == q_num]
            
            if not question_results:
                print(f"  Q{q_num}: No results found")
                continue
            
            print(f"\n  Q{q_num} (Expected: {answer_key[subject][q_num-1]}):")
            
            # Get all option intensities
            option_intensities = {}
            for result in question_results:
                option = result['selected_option']
                confidence = result['confidence']
                x, y = result['position']
                
                # Extract bubble region for intensity analysis
                bubble_radius = 15
                y1, y2 = max(0, y-bubble_radius), min(image.shape[0], y+bubble_radius)
                x1, x2 = max(0, x-bubble_radius), min(image.shape[1], x+bubble_radius)
                bubble_region = image[y1:y2, x1:x2]
                
                if bubble_region.size > 0:
                    mean_intensity = np.mean(bubble_region)
                    min_intensity = np.min(bubble_region)
                    filled_pixels = np.sum(bubble_region < 180)  # Dark pixels
                    total_pixels = bubble_region.size
                    fill_ratio = filled_pixels / total_pixels
                    
                    option_intensities[option] = {
                        'mean': mean_intensity,
                        'min': min_intensity,
                        'fill_ratio': fill_ratio,
                        'confidence': confidence,
                        'position': (x, y)
                    }
            
            # Sort by confidence and show analysis
            sorted_options = sorted(option_intensities.items(), 
                                  key=lambda x: x[1]['confidence'], reverse=True)
            
            detected = sorted_options[0][0] if sorted_options else 'NONE'
            expected = answer_key[subject][q_num-1]
            
            print(f"    Detected: {detected} | Expected: {expected} | {'✅' if detected == expected else '❌'}")
            print(f"    Intensity Analysis:")
            
            for option, data in sorted_options:
                marker = "🔵" if option == detected else "⚪"
                expected_marker = "⭐" if option == expected else ""
                print(f"      {marker} {option}: mean={data['mean']:.1f}, "
                      f"min={data['min']:.1f}, fill={data['fill_ratio']:.3f}, "
                      f"conf={data['confidence']:.3f} {expected_marker}")
    
    print("\n=== THRESHOLD ANALYSIS ===")
    
    # Analyze overall threshold distribution
    all_confidences = [r['confidence'] for r in results]
    all_mean_intensities = []
    
    for result in results:
        x, y = result['position']
        bubble_radius = 15
        y1, y2 = max(0, y-bubble_radius), min(image.shape[0], y+bubble_radius)
        x1, x2 = max(0, x-bubble_radius), min(image.shape[1], x+bubble_radius)
        bubble_region = image[y1:y2, x1:x2]
        if bubble_region.size > 0:
            all_mean_intensities.append(np.mean(bubble_region))
    
    if all_confidences:
        print(f"Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
        print(f"Mean confidence: {np.mean(all_confidences):.3f}")
        print(f"Confidence threshold used in detection: varies by region")
    
    if all_mean_intensities:
        print(f"Intensity range: {min(all_mean_intensities):.1f} - {max(all_mean_intensities):.1f}")
        print(f"Mean intensity: {np.mean(all_mean_intensities):.1f}")
    
    print(f"\nTotal analyzed bubbles: {len(results)}")

if __name__ == "__main__":
    analyze_bubble_intensities()