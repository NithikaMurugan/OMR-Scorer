#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def debug_bubble_detection(image_path):
    """Debug the new bubble detection algorithm"""
    
    # Load image
    image = Image.open(image_path)
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    print(f"Image shape: {gray_image.shape}")
    
    # Apply preprocessing
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} total contours")
    
    # Filter bubble-like contours
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 1000:  # Reasonable bubble size
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.2:  # Somewhat circular
                    x, y, w, h = cv2.boundingRect(contour)
                    bubble_candidates.append({
                        'center': (x + w//2, y + h//2),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'circularity': circularity
                    })
    
    print(f"Found {len(bubble_candidates)} bubble candidates")
    
    # Sort by position
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    
    # Group into rows
    rows = []
    current_row = []
    last_y = -100
    
    for bubble in bubble_candidates:
        y = bubble['center'][1]
        if abs(y - last_y) > 30:  # New row threshold
            if current_row:
                rows.append(current_row)
            current_row = [bubble]
        else:
            current_row.append(bubble)
        last_y = y
    
    if current_row:
        rows.append(current_row)
    
    print(f"Organized into {len(rows)} rows:")
    for i, row in enumerate(rows):
        print(f"  Row {i}: {len(row)} bubbles")
        if len(row) > 0:
            y_positions = [b['center'][1] for b in row]
            x_positions = [b['center'][0] for b in row]
            print(f"    Y range: {min(y_positions)}-{max(y_positions)}")
            print(f"    X range: {min(x_positions)}-{max(x_positions)}")
    
    # Find question rows
    question_rows = []
    for i, row in enumerate(rows):
        if 15 <= len(row) <= 25:  # Reasonable range for question rows
            row.sort(key=lambda b: b['center'][0])  # Sort by X position
            question_rows.append((i, row))
            
            # Analyze gaps
            x_positions = [b['center'][0] for b in row]
            gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
            avg_gap = np.mean(gaps)
            large_gaps = [i for i, gap in enumerate(gaps) if gap > avg_gap * 1.8]
            
            print(f"  Question row {i}: {len(row)} bubbles, avg gap: {avg_gap:.1f}, large gaps at: {large_gaps}")
    
    print(f"Found {len(question_rows)} potential question rows")
    
    # Visualize the first few question rows
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(gray_image, cmap='gray')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, (row_idx, row) in enumerate(question_rows[:5]):
        color = colors[idx % len(colors)]
        for bubble in row:
            x, y, w, h = bubble['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none', alpha=0.7)
            ax.add_patch(rect)
    
    ax.set_title('Detected Question Row Bubbles')
    plt.tight_layout()
    plt.savefig('debug_bubble_detection.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to debug_bubble_detection.png")

if __name__ == "__main__":
    debug_bubble_detection("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")