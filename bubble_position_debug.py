#!/usr/bin/env python3
"""
Debug bubble position mapping - visualize what bubbles the algorithm detects
vs where the actual marks are on the OMR sheet.
"""

import cv2
import numpy as np
import os
from utils.bubbledetection import detect_actual_bubble_grid

def debug_bubble_positions():
    # Load the image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print("=== BUBBLE POSITION MAPPING DEBUG ===\n")
    
    # Get the detected bubbles from the algorithm
    try:
        bubbles_info = detect_actual_bubble_grid(image)
        print(f"Algorithm detected {len(bubbles_info)} bubbles")
        
        # Create visualization
        debug_img = original.copy()
        
        # Draw all detected bubbles
        for i, bubble in enumerate(bubbles_info):
            x, y = bubble['position']
            subject = bubble.get('subject', 'Unknown')
            question = bubble.get('question', 0)
            option = bubble.get('option', '?')
            fill_percent = bubble.get('fill_percent', 0)
            
            # Color code by subject
            if 'Python' in subject:
                color = (255, 0, 0)  # Blue
            elif 'EDA' in subject:
                color = (0, 255, 0)  # Green
            elif 'SQL' in subject:
                color = (0, 0, 255)  # Red
            elif 'POWER' in subject:
                color = (255, 255, 0)  # Cyan
            elif 'Satistics' in subject:
                color = (255, 0, 255)  # Magenta
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bubble with info
            cv2.circle(debug_img, (x, y), 8, color, 2)
            cv2.putText(debug_img, f"{subject[0]}{question}{option}", 
                       (x-15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.putText(debug_img, f"{fill_percent:.2f}", 
                       (x-15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Save debug image
        cv2.imwrite('algorithm_bubble_detection.jpg', debug_img)
        print("Algorithm bubble detection saved as: algorithm_bubble_detection.jpg")
        
        # Now let's manually check the expected positions from our previous analysis
        expected_positions = {
            'Python': {
                1: {'A': (191, 235), 'B': (222, 236), 'C': (236, 253), 'D': (288, 237)},
                2: {'A': (185, 296), 'B': (218, 297), 'C': (252, 297), 'D': (284, 298)},
            }
        }
        
        # Draw expected positions on a separate image
        expected_img = original.copy()
        
        for subject, questions in expected_positions.items():
            for q_num, options in questions.items():
                for option, (x, y) in options.items():
                    # Draw expected position
                    cv2.circle(expected_img, (x, y), 8, (0, 255, 255), 2)  # Yellow
                    cv2.putText(expected_img, f"E{subject[0]}{q_num}{option}", 
                               (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    
                    # Check pixel intensity at this position
                    radius = 10
                    y1, y2 = max(0, y-radius), min(image.shape[0], y+radius)
                    x1, x2 = max(0, x-radius), min(image.shape[1], x+radius)
                    region = image[y1:y2, x1:x2]
                    
                    if region.size > 0:
                        dark_pixels = np.sum(region < 180)
                        total_pixels = region.size
                        fill_ratio = dark_pixels / total_pixels
                        cv2.putText(expected_img, f"{fill_ratio:.2f}", 
                                   (x-15, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        cv2.imwrite('expected_bubble_positions.jpg', expected_img)
        print("Expected bubble positions saved as: expected_bubble_positions.jpg")
        
        # Compare algorithm detected vs expected for Python Q1
        print(f"\n=== PYTHON Q1 COMPARISON ===")
        python_q1_bubbles = [b for b in bubbles_info if 'Python' in b.get('subject', '') and b.get('question') == 1]
        
        print(f"Algorithm detected {len(python_q1_bubbles)} bubbles for Python Q1:")
        for bubble in python_q1_bubbles:
            x, y = bubble['position']
            option = bubble.get('option', '?')
            fill = bubble.get('fill_percent', 0)
            print(f"  {option}: ({x}, {y}) fill={fill:.3f}")
        
        print(f"\nExpected positions:")
        for option, (x, y) in expected_positions['Python'][1].items():
            print(f"  {option}: ({x}, {y})")
        
        print(f"\n=== POSITION DIFFERENCES ===")
        for option in ['A', 'B', 'C', 'D']:
            expected_pos = expected_positions['Python'][1][option]
            
            # Find closest algorithm detection
            closest_bubble = None
            min_distance = float('inf')
            
            for bubble in python_q1_bubbles:
                if bubble.get('option') == option:
                    bubble_pos = bubble['position']
                    distance = ((bubble_pos[0] - expected_pos[0])**2 + (bubble_pos[1] - expected_pos[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_bubble = bubble
            
            if closest_bubble:
                detected_pos = closest_bubble['position']
                fill = closest_bubble.get('fill_percent', 0)
                print(f"  {option}: Expected({expected_pos[0]}, {expected_pos[1]}) -> Detected({detected_pos[0]}, {detected_pos[1]}) "
                      f"Distance={min_distance:.1f} Fill={fill:.3f}")
            else:
                print(f"  {option}: Expected({expected_pos[0]}, {expected_pos[1]}) -> NOT DETECTED")
    
    except Exception as e:
        print(f"Error in bubble detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_bubble_positions()