#!/usr/bin/env python3
"""
Advanced bubble analysis to understand detection patterns.
"""

import cv2
import numpy as np
from PIL import Image
from utils.bubbledetection import detect_actual_bubble_grid, get_question_responses_with_actual_bubbles
import os

def analyze_bubble_detection(image_path):
    """Analyze bubble detection patterns for debugging."""
    
    print(f"Analyzing bubble detection for: {image_path}")
    
    # Load and process image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    print(f"Image shape: {gray.shape}")
    
    # Detect bubbles
    organized_bubbles = detect_actual_bubble_grid(gray)
    
    # Analyze first few questions in detail
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    
    for subject in subjects[:2]:  # Analyze first 2 subjects
        print(f"\n=== {subject} Analysis ===")
        
        if subject not in organized_bubbles:
            print(f"No bubbles found for {subject}")
            continue
            
        for q_num in range(min(5, len(organized_bubbles[subject]))):  # First 5 questions
            print(f"\nQuestion {q_num + 1}:")
            
            if q_num not in organized_bubbles[subject]:
                print("  No bubbles found for this question")
                continue
            
            bubbles = organized_bubbles[subject][q_num]
            options = ["A", "B", "C", "D"]
            
            # Analyze each bubble
            for i, (option, bubble) in enumerate(zip(options, bubbles)):
                if bubble is None:
                    print(f"  {option}: No bubble detected")
                    continue
                
                roi = bubble['roi']
                
                # Calculate fill percentage manually
                if roi.size > 0:
                    _, otsu_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    otsu_fill = np.sum(otsu_binary == 255) / otsu_binary.size
                    
                    adaptive_binary = cv2.adaptiveThreshold(
                        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
                    )
                    adaptive_fill = np.sum(adaptive_binary == 255) / adaptive_binary.size
                    
                    mean_intensity = np.mean(roi)
                    intensity_darkness = (255 - mean_intensity) / 255.0
                    
                    combined_fill = (otsu_fill * 0.4 + adaptive_fill * 0.4 + intensity_darkness * 0.2)
                    
                    print(f"  {option}: fill={combined_fill:.3f}, otsu={otsu_fill:.3f}, adaptive={adaptive_fill:.3f}, intensity={mean_intensity:.1f}")
                else:
                    print(f"  {option}: Empty ROI")
            
            # Get the response for this question
            response = get_question_responses_with_actual_bubbles(organized_bubbles, subject, q_num)
            print(f"  Response: '{response}'")


def save_bubble_visualization(image_path, output_path="bubble_debug.jpg"):
    """Save a visualization of detected bubbles."""
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        color_img = img_array.copy()
    else:
        gray = img_array
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Detect bubbles
    organized_bubbles = detect_actual_bubble_grid(gray)
    
    # Draw all detected bubbles
    bubble_count = 0
    for subject, questions in organized_bubbles.items():
        for q_num, bubbles in questions.items():
            for i, bubble in enumerate(bubbles):
                if bubble is not None:
                    x, y, w, h = bubble['bbox']
                    
                    # Draw rectangle around bubble
                    cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    
                    # Label the bubble
                    options = ["A", "B", "C", "D"]
                    label = f"{subject[:1]}{q_num+1}{options[i]}"
                    cv2.putText(color_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    
                    bubble_count += 1
    
    print(f"Visualized {bubble_count} bubbles")
    cv2.imwrite(output_path, color_img)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if os.path.exists(image_path):
        analyze_bubble_detection(image_path)
        save_bubble_visualization(image_path)
    else:
        print(f"Image not found: {image_path}")