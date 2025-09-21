#!/usr/bin/env python3
"""
Option Mapping Debug Tool
Analyzes how the algorithm assigns A, B, C, D labels to detected bubble positions
and compares with actual form layout.
"""

import cv2
import numpy as np
from utils.bubbledetection import detect_actual_bubble_grid

def debug_option_mapping():
    """Debug the option mapping by showing detected bubble positions and their assigned labels."""
    
    # Load and preprocess the image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("=== OPTION MAPPING DEBUG ===")
    print("Analyzing how A, B, C, D labels are assigned to detected bubble positions...\n")
    
    # Get organized bubbles
    organized_bubbles = detect_actual_bubble_grid(gray_image)
    
    # Focus on Python subject (since we know the answer for Q2 should be C)
    subject = "Python"
    
    print(f"=== {subject.upper()} SUBJECT ===")
    
    # Check questions 1 and 2 (we know Q2 answer should be C)
    for q_num in range(2):  # Questions 1 and 2
        print(f"\n--- Question {q_num + 1} ---")
        
        bubbles = organized_bubbles[subject][q_num]
        options = ["A", "B", "C", "D"]
        
        if len(bubbles) < 4:
            print(f"ERROR: Only {len(bubbles)} bubbles detected for Q{q_num + 1}")
            continue
            
        print("Bubble positions and assigned labels:")
        
        for i, (option, bubble) in enumerate(zip(options, bubbles)):
            if bubble is None:
                print(f"  {option}: No bubble detected")
                continue
                
            center = bubble['center']
            area = bubble['area']
            
            print(f"  {option}: Position({center[0]:3d}, {center[1]:3d}) Area={area:4.0f}")
            
        # Manual verification for Q2 (we know C should be at around (252, 297))
        if q_num == 1:  # Q2 (0-indexed)
            print("\n  Manual Reference: C should be around (252, 297)")
            
            # Find which detected bubble is closest to the manual C position
            manual_c_pos = (252, 297)
            min_distance = float('inf')
            closest_option = None
            
            for i, (option, bubble) in enumerate(zip(options, bubbles)):
                if bubble is None:
                    continue
                    
                center = bubble['center']
                distance = np.sqrt((center[0] - manual_c_pos[0])**2 + (center[1] - manual_c_pos[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_option = option
                    
            print(f"  Closest to manual C position: {closest_option} (distance={min_distance:.1f})")
            
            if closest_option != "C":
                print(f"  *** MAPPING ERROR: Manual C position is assigned to {closest_option}, not C! ***")
            else:
                print(f"  ✓ Correct: Manual C position correctly assigned to C")

def debug_bubble_sorting():
    """Debug how bubbles are sorted and distributed across options."""
    
    print("\n=== BUBBLE SORTING DEBUG ===")
    print("Analyzing how bubbles are sorted before option assignment...\n")
    
    # Load and preprocess the image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get organized bubbles
    organized_bubbles = detect_actual_bubble_grid(gray_image)
    
    subject = "Python"
    q_num = 1  # Q2 (0-indexed)
    
    bubbles = organized_bubbles[subject][q_num]
    
    print(f"{subject} Q{q_num + 1} - Bubbles sorted by X position (left to right):")
    
    if len(bubbles) >= 4:
        for i, bubble in enumerate(bubbles[:4]):
            if bubble is None:
                print(f"  Position {i+1}: No bubble")
                continue
                
            center = bubble['center']
            option = ["A", "B", "C", "D"][i]
            print(f"  Position {i+1} -> {option}: ({center[0]:3d}, {center[1]:3d})")
            
        print(f"\nExpected: C at (252, 297) should be assigned to option C")
        print(f"But if bubbles are sorted strictly by X position, the assignment might be wrong!")

if __name__ == "__main__":
    debug_option_mapping()
    debug_bubble_sorting()