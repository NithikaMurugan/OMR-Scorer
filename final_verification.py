#!/usr/bin/env python3
"""
Final Verification
Cross-reference manual answer positions with algorithm bubble positions
to verify which option each manual position actually corresponds to.
"""

import cv2
import numpy as np
from utils.bubbledetection import detect_actual_bubble_grid

def final_verification():
    """Final verification of manual vs algorithm position mapping."""
    
    # Load image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("=== FINAL VERIFICATION ===")
    print("Cross-referencing manual answer positions with algorithm bubble positions...\n")
    
    # Get organized bubbles
    organized_bubbles = detect_actual_bubble_grid(gray_image)
    
    # Manual answer positions from previous analysis
    manual_answers = [
        ("Python Q1", (193, 265), "A"),
        ("Python Q2", (252, 297), "C"),  # This is the one we've been debugging
        ("Python Q3", (194, 393), "A"),
        ("Python Q4", (222, 524), "B"),
        ("Python Q5", (253, 697), "C"),
        ("Python Q6", (191, 729), "A")
    ]
    
    subject = "Python"
    options = ["A", "B", "C", "D"]
    
    print("Manual Answer Position -> Closest Algorithm Bubble -> Real Option")
    print("==================================================================")
    
    for question_str, (manual_x, manual_y), expected_option in manual_answers:
        q_num = int(question_str.split()[1][1:]) - 1  # Extract question number and convert to 0-indexed
        
        if q_num >= 6:  # We only have 6 questions
            continue
            
        bubbles = organized_bubbles[subject][q_num]
        
        # Find closest algorithm bubble to manual position
        closest_option = None
        min_distance = float('inf')
        
        for i, (option, bubble) in enumerate(zip(options, bubbles)):
            if bubble is None:
                continue
                
            algo_x, algo_y = bubble['center']
            distance = np.sqrt((manual_x - algo_x)**2 + (manual_y - algo_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_option = option
        
        # Determine if our manual analysis was correct
        status = "✅" if closest_option == expected_option else "❌"
        
        print(f"{question_str}: ({manual_x:3d}, {manual_y:3d}) -> {closest_option} | Expected: {expected_option} {status}")
        
        if closest_option != expected_option:
            print(f"   *** Manual analysis error: Position actually corresponds to {closest_option}, not {expected_option}! ***")
    
    print("\n=== CONCLUSION ===")
    print("If many manual positions don't match expected options,")
    print("then the original answer key or manual analysis was incorrect,")
    print("not the algorithm!")

if __name__ == "__main__":
    final_verification()