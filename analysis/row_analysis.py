import sys
import os
from PIL import Image
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bubbledetection import extract_responses, apply_balanced_precision_preprocessing, detect_balanced_bubble_rows
import cv2

def analyze_bubble_distribution():
    """Analyze bubble distribution to understand row organization"""
    print("=== BUBBLE DISTRIBUTION ANALYSIS ===")
    print()
    
    # Load and process the test image
    img = Image.open("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")
    img_array = np.array(img)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    print(f"Image shape: {gray.shape}")
    
    # Apply preprocessing
    processed = apply_balanced_precision_preprocessing(gray)
    
    # Find contours (copy from bubbledetection.py logic)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} total contours")
    
    # Filter contours to find bubble candidates
    bubble_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 50 or area > 2000:  # Size filter
            continue
        
        if perimeter < 20 or perimeter > 200:
            continue
        
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        if circularity < 0.3:  # Not circular enough
            continue
        
        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w//2, y + h//2)
        
        bubble_candidates.append({
            'contour': contour,
            'center': center,
            'area': area,
            'circularity': circularity,
            'roi': None  # Will be filled later if needed
        })
    
    print(f"Found {len(bubble_candidates)} bubble candidates")
    
    # Analyze Y-coordinate distribution
    y_coords = [b['center'][1] for b in bubble_candidates]
    y_coords.sort()
    
    print(f"Y-coordinate range: {min(y_coords)} to {max(y_coords)}")
    print(f"Y-coordinate spread: {max(y_coords) - min(y_coords)}")
    
    # Detect rows with detailed analysis
    rows = detect_balanced_bubble_rows(bubble_candidates, tolerance=25)
    
    print(f"\n=== ROW ANALYSIS ({len(rows)} rows detected) ===")
    
    for i, row in enumerate(rows):
        row_y = row[0]['center'][1]
        row_size = len(row)
        row_x_min = min(b['center'][0] for b in row)
        row_x_max = max(b['center'][0] for b in row)
        row_width = row_x_max - row_x_min
        
        print(f"Row {i:2d}: Y={row_y:4d}, Bubbles={row_size:2d}, X-range={row_x_min:4d}-{row_x_max:4d} (width={row_width:4d})")
    
    # Analyze which rows might contain actual questions
    print(f"\n=== QUESTION ROW IDENTIFICATION ===")
    
    # Look for rows with consistent bubble counts (likely 20 bubbles for 5 subjects × 4 options)
    expected_bubbles_per_row = 20  # 5 subjects × 4 options
    
    question_rows = []
    for i, row in enumerate(rows):
        if len(row) >= 15:  # Close to expected number
            question_rows.append((i, row))
    
    print(f"Potential question rows (≥15 bubbles): {len(question_rows)}")
    
    for i, (row_idx, row) in enumerate(question_rows):
        row_y = row[0]['center'][1]
        print(f"Q-Row {i:2d} (Row {row_idx:2d}): Y={row_y:4d}, Bubbles={len(row):2d}")
    
    # Check if we're missing the bottom rows
    if len(question_rows) < 20:
        print(f"\n⚠️  WARNING: Only {len(question_rows)} potential question rows found, need 20")
        print("Analyzing bottom rows that might be missed...")
        
        # Look at all rows, not just the well-populated ones
        all_rows_by_y = sorted(enumerate(rows), key=lambda x: x[1][0]['center'][1])
        
        print("\nAll rows sorted by Y position:")
        for i, (orig_idx, row) in enumerate(all_rows_by_y):
            row_y = row[0]['center'][1]
            is_question = len(row) >= 15
            status = "✓ Question" if is_question else "  Other"
            print(f"  {i:2d}: Y={row_y:4d}, Bubbles={len(row):2d} {status}")
    
    return rows, question_rows

if __name__ == "__main__":
    analyze_bubble_distribution()