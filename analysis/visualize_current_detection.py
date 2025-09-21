#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_current_detection(image_path):
    """Visualize the current grid-based detection approach"""
    
    # Load image
    image = Image.open(image_path)
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    print(f"Image shape: {gray_image.shape}")
    
    # Apply preprocessing (same as in detect_actual_bubble_grid)
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter bubble-like contours
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 1000:  # Current bubble size filter
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.2:  # Current circularity filter
                    x, y, w, h = cv2.boundingRect(contour)
                    bubble_candidates.append({
                        'center': (x + w//2, y + h//2),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'circularity': circularity,
                        'roi': gray_image[y:y+h, x:x+w]
                    })
    
    # Sort and take first 400 for questions
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    question_bubbles = bubble_candidates[:400] if len(bubble_candidates) >= 400 else bubble_candidates
    
    print(f"Using {len(question_bubbles)} bubbles for questions")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Original image
    axes[0,0].imshow(gray_image, cmap='gray')
    axes[0,0].set_title('Original Image')
    
    # Threshold image
    axes[0,1].imshow(thresh, cmap='gray')
    axes[0,1].set_title('Threshold Image')
    
    # All detected bubbles
    axes[1,0].imshow(gray_image, cmap='gray')
    for bubble in bubble_candidates:
        x, y, w, h = bubble['bbox']
        rect = plt.Rectangle((x, y), w, h, linewidth=1, 
                           edgecolor='red', facecolor='none', alpha=0.5)
        axes[1,0].add_patch(rect)
    axes[1,0].set_title(f'All {len(bubble_candidates)} Detected Bubbles')
    
    # Question bubbles organized by grid
    axes[1,1].imshow(gray_image, cmap='gray')
    
    # Visualize the first few questions with different colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    
    for q_num in range(min(5, 20)):  # Show first 5 questions
        start_idx = q_num * 20
        end_idx = min(start_idx + 20, len(question_bubbles))
        
        if end_idx > start_idx:
            question_row_bubbles = question_bubbles[start_idx:end_idx]
            question_row_bubbles.sort(key=lambda b: b['center'][0])
            
            # Draw each subject's bubbles in different colors
            for s_idx in range(5):
                subject_start = s_idx * 4
                subject_end = min(subject_start + 4, len(question_row_bubbles))
                
                if subject_end > subject_start:
                    color = colors[s_idx]
                    for bubble in question_row_bubbles[subject_start:subject_end]:
                        x, y, w, h = bubble['bbox']
                        rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor=color, facecolor='none', alpha=0.8)
                        axes[1,1].add_patch(rect)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='none', edgecolor=colors[i], 
                                   linewidth=2, label=subjects[i]) for i in range(5)]
    axes[1,1].legend(handles=legend_elements, loc='upper right')
    axes[1,1].set_title('Question Grid Mapping (First 5 Questions)')
    
    plt.tight_layout()
    plt.savefig('current_detection_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to current_detection_analysis.png")
    
    # Analyze bubble properties
    print("\nBubble Analysis:")
    areas = [b['area'] for b in bubble_candidates]
    circularities = [b['circularity'] for b in bubble_candidates]
    
    print(f"Area range: {min(areas):.1f} - {max(areas):.1f}, mean: {np.mean(areas):.1f}")
    print(f"Circularity range: {min(circularities):.3f} - {max(circularities):.3f}, mean: {np.mean(circularities):.3f}")
    
    # Sample some bubble ROIs to check darkness
    print("\nSample bubble darkness analysis:")
    for i in range(min(10, len(question_bubbles))):
        bubble = question_bubbles[i]
        roi = bubble['roi']
        if roi.size > 0:
            _, thresh_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            darkness_ratio = np.sum(thresh_roi == 255) / thresh_roi.size
            print(f"  Bubble {i}: darkness = {darkness_ratio:.3f}, area = {bubble['area']:.0f}")

if __name__ == "__main__":
    visualize_current_detection("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")