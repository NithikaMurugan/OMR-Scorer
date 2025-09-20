#!/usr/bin/env python3
"""
Row Analysis Tool
Analyzes which question row the manual answer positions belong to.
"""

import cv2
import numpy as np
from utils.bubbledetection import detect_actual_bubble_grid

def analyze_question_rows():
    """Analyze which question rows contain the manual answer positions."""
    
    # Load image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("=== QUESTION ROW ANALYSIS ===")
    print("Checking which question rows correspond to manual answer positions...\n")
    
    # Get organized bubbles
    organized_bubbles = detect_actual_bubble_grid(gray_image)
    
    # Manual answer positions we know from previous analysis
    manual_answers = {
        "Python Q1": (193, 265),  # Should be A
        "Python Q2": (252, 297),  # Should be C
        "Python Q3": (194, 393),  # Should be A
        "Python Q4": (222, 524),  # Should be B
        "Python Q5": (253, 697),  # Should be C  
        "Python Q6": (191, 729),  # Should be A
    }
    
    # Show all detected question rows first
    print("Detected Question Rows:")
    subject = "Python"
    for q_num in range(6):  # We know there are 6 actual questions
        bubbles = organized_bubbles[subject][q_num]
        if len(bubbles) >= 4 and bubbles[0] is not None:
            y_pos = bubbles[0]['center'][1]  # Y position of first bubble in row
            print(f"  Q{q_num + 1}: Y={y_pos}")
    
    print("\nManual Answer Positions:")
    for question, (x, y) in manual_answers.items():
        print(f"  {question}: ({x}, {y})")
        
        # Find closest question row
        closest_q = None
        min_distance = float('inf')
        
        for q_num in range(6):
            bubbles = organized_bubbles[subject][q_num]
            if len(bubbles) >= 4 and bubbles[0] is not None:
                row_y = bubbles[0]['center'][1]
                distance = abs(y - row_y)
                if distance < min_distance:
                    min_distance = distance
                    closest_q = q_num + 1
                    
        print(f"    -> Closest to detected Q{closest_q} (distance={min_distance} pixels)")
    
    print("\n=== MAPPING ANALYSIS ===")
    print("This shows if the algorithm is mapping questions to the wrong rows!")

if __name__ == "__main__":
    analyze_question_rows()