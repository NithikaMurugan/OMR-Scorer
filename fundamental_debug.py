#!/usr/bin/env python3
"""
Fundamental debug: compare actual manual bubble positions with algorithm detections.
This will help us understand if the algorithm is detecting the right bubbles at all.
"""

import cv2
import numpy as np
import os

def compare_manual_vs_algorithm():
    # Load the image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print("=== MANUAL VS ALGORITHM BUBBLE DETECTION COMPARISON ===\n")
    
    # Manual expected positions from position analysis
    expected_positions = {
        'Python': {
            1: {'A': (191, 235), 'B': (222, 236), 'C': (236, 253), 'D': (288, 237)},
            2: {'A': (185, 296), 'B': (218, 297), 'C': (252, 297), 'D': (284, 298)},
        }
    }
    
    # Check pixel intensities at these manual positions
    print("=== MANUAL BUBBLE ANALYSIS ===")
    for subject, questions in expected_positions.items():
        for q_num, options in questions.items():
            print(f"\n{subject} Q{q_num}:")
            for option, (x, y) in options.items():
                # Extract bubble region
                radius = 15
                y1, y2 = max(0, y-radius), min(image.shape[0], y+radius)
                x1, x2 = max(0, x-radius), min(image.shape[1], x+radius)
                region = image[y1:y2, x1:x2]
                
                if region.size > 0:
                    mean_intensity = np.mean(region)
                    min_intensity = np.min(region)
                    dark_pixels = np.sum(region < 180)  # Dark pixels
                    total_pixels = region.size
                    fill_ratio = dark_pixels / total_pixels
                    
                    print(f"  {option} at ({x}, {y}): mean={mean_intensity:.1f}, "
                          f"min={min_intensity:.0f}, fill_ratio={fill_ratio:.3f}")
    
    # Now let's see what the algorithm actually detects
    print(f"\n=== ALGORITHM DETECTION ANALYSIS ===")
    
    # Get contours that the algorithm finds
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Algorithm found {len(contours)} total contours")
    
    # Filter for bubble-like contours
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 <= area <= 2000:  # Reasonable bubble size
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.4:  # Somewhat circular
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        bubble_candidates.append({
                            'center': (cx, cy),
                            'area': area,
                            'circularity': circularity
                        })
    
    print(f"Found {len(bubble_candidates)} bubble candidates")
    
    # Find closest algorithm detections to our manual positions
    print(f"\n=== CLOSEST ALGORITHM MATCHES TO MANUAL POSITIONS ===")
    
    visualization = color_image.copy()
    
    for subject, questions in expected_positions.items():
        for q_num, options in questions.items():
            print(f"\n{subject} Q{q_num}:")
            for option, (manual_x, manual_y) in options.items():
                # Draw manual position in GREEN
                cv2.circle(visualization, (manual_x, manual_y), 10, (0, 255, 0), 2)
                cv2.putText(visualization, f"M{option}", (manual_x-15, manual_y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Find closest algorithm detection
                closest_distance = float('inf')
                closest_bubble = None
                
                for bubble in bubble_candidates:
                    algo_x, algo_y = bubble['center']
                    distance = ((algo_x - manual_x)**2 + (algo_y - manual_y)**2)**0.5
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_bubble = bubble
                
                if closest_bubble:
                    algo_x, algo_y = closest_bubble['center']
                    area = closest_bubble['area']
                    circularity = closest_bubble['circularity']
                    
                    # Draw algorithm detection in RED
                    cv2.circle(visualization, (algo_x, algo_y), 8, (0, 0, 255), 2)
                    cv2.putText(visualization, f"A{option}", (algo_x-15, algo_y+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Draw line connecting them
                    cv2.line(visualization, (manual_x, manual_y), (algo_x, algo_y), (255, 0, 255), 1)
                    
                    print(f"  {option}: Manual({manual_x}, {manual_y}) -> Algorithm({algo_x}, {algo_y})")
                    print(f"      Distance: {closest_distance:.1f}, Area: {area:.0f}, Circularity: {circularity:.3f}")
                else:
                    print(f"  {option}: Manual({manual_x}, {manual_y}) -> NO ALGORITHM MATCH FOUND")
    
    # Save visualization
    cv2.imwrite('manual_vs_algorithm_comparison.jpg', visualization)
    print(f"\nVisualization saved as: manual_vs_algorithm_comparison.jpg")
    print("GREEN circles = Manual positions")
    print("RED circles = Closest algorithm detections")
    print("MAGENTA lines = Connection between manual and algorithm")
    
    # Analysis conclusion
    print(f"\n=== ANALYSIS CONCLUSION ===")
    
    # Check if the algorithm is finding bubbles in roughly the right area
    python_q1_manual_center = (np.mean([191, 222, 236, 288]), 235)  # Rough center of Q1
    nearby_algorithm_bubbles = []
    
    for bubble in bubble_candidates:
        algo_x, algo_y = bubble['center']
        distance_to_q1 = ((algo_x - python_q1_manual_center[0])**2 + (algo_y - python_q1_manual_center[1])**2)**0.5
        if distance_to_q1 < 100:  # Within 100 pixels of Q1 area
            nearby_algorithm_bubbles.append(bubble)
    
    print(f"Algorithm found {len(nearby_algorithm_bubbles)} bubbles near Python Q1 area")
    
    if len(nearby_algorithm_bubbles) < 4:
        print("❌ PROBLEM: Algorithm is not finding enough bubbles in the expected area!")
        print("   This suggests the bubble detection parameters are too strict.")
    else:
        print("✅ Algorithm finds bubbles in the right area, but positions may be wrong.")
        print("   This suggests the bubble detection is working but position mapping is off.")

if __name__ == "__main__":
    compare_manual_vs_algorithm()