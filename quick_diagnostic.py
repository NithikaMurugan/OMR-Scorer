#!/usr/bin/env python3
"""
Quick diagnostic analysis of the specific wrong answers we're getting.
Focus on understanding why detection is wrong for specific cases.
"""

import cv2
import numpy as np
import os

def analyze_wrong_detections():
    # Load the image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print("=== QUICK DIAGNOSTIC FOR WRONG DETECTIONS ===\n")
    
    # Answer key for the 6 actual questions
    answer_key = {
        'Python': ['A', 'C', 'C', 'C', 'C', 'A'],
        'EDA': ['A', 'D', 'B', 'A', 'C', 'B'],
        'SQL': ['C', 'C', 'C', 'B', 'B', 'A'],
        'POWER BI': ['B', 'C', 'A', 'B', 'C', 'B'],
        'Satistics': ['A', 'B', 'C', 'B', 'C', 'B']
    }
    
    # Get current detection results
    from utils.bubbledetection import extract_responses
    current_results = extract_responses(image)
    
    print("=== COMPARISON OF CURRENT vs EXPECTED ===")
    
    # Focus on the first 6 questions only (the actual questions)
    wrong_count = 0
    correct_count = 0
    
    for subject in ['Python', 'EDA', 'SQL', 'POWER BI', 'Satistics']:
        print(f"\n{subject}:")
        
        detected_answers = current_results[subject][:6]  # Only first 6
        expected_answers = answer_key[subject]
        
        for q_num in range(6):
            detected = detected_answers[q_num] if q_num < len(detected_answers) else ''
            expected = expected_answers[q_num]
            
            if detected == expected:
                status = "✅"
                correct_count += 1
            else:
                status = "❌"
                wrong_count += 1
            
            print(f"  Q{q_num+1}: Detected='{detected}' | Expected='{expected}' {status}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Correct: {correct_count}/30 ({correct_count/30*100:.1f}%)")
    print(f"Wrong: {wrong_count}/30 ({wrong_count/30*100:.1f}%)")
    
    # Now let's examine some specific wrong cases
    print(f"\n=== DETAILED ANALYSIS OF ERRORS ===")
    
    # Let's look at Python Q1 which should be A but detects B
    print(f"\nCase Study: Python Q1 (Expected: A, Detected: B)")
    print("Position analysis from previous output:")
    print("  A:(191,235) | B:(222,236) | C:(236,253) | D:(288,237)")
    print("  Detection picked B instead of A")
    print("  This suggests the bubble at position B has higher confidence/darkness")
    
    # Check the actual pixel values at these positions
    positions = {
        'A': (191, 235),
        'B': (222, 236),
        'C': (236, 253),
        'D': (288, 237)
    }
    
    print("\nPixel analysis at exact positions:")
    for option, (x, y) in positions.items():
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            # Extract small region around this point
            radius = 10
            y1, y2 = max(0, y-radius), min(image.shape[0], y+radius)
            x1, x2 = max(0, x-radius), min(image.shape[1], x+radius)
            region = image[y1:y2, x1:x2]
            
            if region.size > 0:
                mean_val = np.mean(region)
                min_val = np.min(region)
                dark_pixels = np.sum(region < 180)
                total_pixels = region.size
                fill_ratio = dark_pixels / total_pixels
                
                marker = "🔵" if option == 'B' else "⚪"  # B was detected
                expected_marker = "⭐" if option == 'A' else ""  # A is expected
                
                print(f"  {marker} {option}: mean={mean_val:.1f}, min={min_val:.1f}, "
                      f"fill_ratio={fill_ratio:.3f} {expected_marker}")
    
    # Quick fix suggestion
    print(f"\n=== POSSIBLE FIX DIRECTIONS ===")
    print("1. Threshold adjustment: Current threshold might be too sensitive")
    print("2. Bubble detection accuracy: Positions might be slightly off")
    print("3. Region size: Bubble extraction region might be wrong size")
    print("4. Pre-processing: Image might need better contrast/filtering")
    
    # Let's also check a correct detection for comparison
    print(f"\nCase Study: Python Q2 (Expected: C, Detected: C) ✅")
    print("This is working correctly - let's see why:")
    
    # Get Q2 positions from the analysis output
    q2_positions = {
        'A': (185, 296),
        'B': (218, 297),
        'C': (252, 297),
        'D': (284, 298)
    }
    
    print("Pixel analysis for correct detection:")
    for option, (x, y) in q2_positions.items():
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            radius = 10
            y1, y2 = max(0, y-radius), min(image.shape[0], y+radius)
            x1, x2 = max(0, x-radius), min(image.shape[1], x+radius)
            region = image[y1:y2, x1:x2]
            
            if region.size > 0:
                mean_val = np.mean(region)
                min_val = np.min(region)
                dark_pixels = np.sum(region < 180)
                total_pixels = region.size
                fill_ratio = dark_pixels / total_pixels
                
                marker = "🔵" if option == 'C' else "⚪"  # C was detected and expected
                
                print(f"  {marker} {option}: mean={mean_val:.1f}, min={min_val:.1f}, "
                      f"fill_ratio={fill_ratio:.3f}")

if __name__ == "__main__":
    analyze_wrong_detections()