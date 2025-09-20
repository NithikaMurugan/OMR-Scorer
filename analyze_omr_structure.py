import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add project root to path
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def analyze_omr_structure(image_path):
    """
    Analyze the actual structure of an OMR sheet to understand the layout.
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    print(f"Image dimensions: {gray.shape}")
    
    # Apply preprocessing
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours (potential bubbles)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter bubble-like contours
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 2000:  # Reasonable bubble size
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
        if abs(y - last_y) > 30:  # New row
            if current_row:
                rows.append(current_row)
            current_row = [bubble]
        else:
            current_row.append(bubble)
        last_y = y
    
    if current_row:
        rows.append(current_row)
    
    print(f"Found {len(rows)} rows of bubbles")
    
    # Analyze row structure
    for i, row in enumerate(rows[:10]):  # First 10 rows
        row.sort(key=lambda b: b['center'][0])  # Sort by x position
        centers = [b['center'] for b in row]
        print(f"Row {i+1}: {len(row)} bubbles at y≈{centers[0][1]} "
              f"x positions: {[c[0] for c in centers]}")
    
    # Look for patterns
    if len(rows) >= 5:
        # Check if we have 5 columns (subjects)
        typical_row = max(rows, key=len)
        if len(typical_row) >= 20:  # 5 subjects × 4 options
            typical_row.sort(key=lambda b: b['center'][0])
            x_positions = [b['center'][0] for b in typical_row]
            
            print(f"\nTypical row has {len(typical_row)} bubbles")
            print("X positions:", x_positions)
            
            # Try to identify subject boundaries
            x_gaps = []
            for i in range(1, len(x_positions)):
                gap = x_positions[i] - x_positions[i-1]
                x_gaps.append(gap)
            
            print("Gaps between bubbles:", x_gaps)
            
            # Identify larger gaps (subject separators)
            avg_gap = np.mean(x_gaps)
            large_gaps = [(i, gap) for i, gap in enumerate(x_gaps) if gap > avg_gap * 1.5]
            print(f"Average gap: {avg_gap:.1f}, Large gaps: {large_gaps}")


if __name__ == "__main__":
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    analyze_omr_structure(image_path)