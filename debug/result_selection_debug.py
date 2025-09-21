#!/usr/bin/env python3
"""
Result Selection Debug
Traces how fill percentages get converted to final option selections.
"""

import cv2
import numpy as np
from utils.bubbledetection import detect_actual_bubble_grid, get_question_responses_with_actual_bubbles

def debug_result_selection():
    """Debug the result selection process."""
    
    # Load image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("=== RESULT SELECTION DEBUG ===")
    print("Tracing fill percentages -> final option selection...\n")
    
    # Get organized bubbles
    organized_bubbles = detect_actual_bubble_grid(gray_image)
    
    # Focus on Python Q2 where we know the issue exists
    subject = "Python"
    question_num = 1  # Q2 (0-indexed)
    
    print(f"Debugging {subject} Q{question_num + 1}:")
    
    # Get the response using the same function as extract_responses
    response = get_question_responses_with_actual_bubbles(organized_bubbles, subject, question_num)
    
    print(f"Final response returned: '{response}'")
    
    # Let's manually trace through the bubble analysis
    bubbles = organized_bubbles[subject][question_num]
    options = ["A", "B", "C", "D"]
    
    print(f"\nBubble analysis for {subject} Q{question_num + 1}:")
    print("Option | Fill % | Mean Int | ROI Size | Center Position")
    print("-------|--------|----------|----------|----------------")
    
    for i, (option, bubble) in enumerate(zip(options, bubbles)):
        if bubble is None:
            print(f"  {option}    |   N/A  |   N/A    |   N/A    | N/A")
            continue
            
        # Calculate fill percentage the same way as the algorithm
        roi = bubble['roi']
        if roi.size == 0:
            fill_pct = 0.0
        else:
            # Use exact same method as calculate_bubble_fill_percentage
            dark_pixels = np.sum(roi < 180)
            total_pixels = roi.size
            fill_pct = dark_pixels / total_pixels
            
            # Apply same adjustments
            mean_intensity = np.mean(roi)
            if mean_intensity > 220:
                fill_pct *= 0.5
            elif mean_intensity < 160:
                fill_pct = min(1.0, fill_pct * 1.2)
            
        center = bubble['center']
        mean_intensity = np.mean(roi) if roi.size > 0 else 0
        print(f"  {option}    | {fill_pct:6.3f} | {mean_intensity:8.1f} | {roi.size:8d} | ({center[0]:3d}, {center[1]:3d})")
    
    print(f"\nExpected: Highest fill % should correspond to option C")
    print(f"Actual result: '{response}'")
    
    if response != 'C':
        print(f"*** BUG DETECTED: Algorithm returned '{response}' instead of 'C' ***")
        print("This indicates a problem in the result selection logic!")

if __name__ == "__main__":
    debug_result_selection()