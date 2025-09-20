#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import os

def analyze_bubble_detection_issue(image_path):
    """
    Focused analysis of why bubble detection is not working correctly for student answers.
    """
    
    print(f"=== BUBBLE DETECTION ANALYSIS ===")
    print(f"Analyzing: {image_path}")
    
    # Load image
    image = Image.open(image_path)
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    print(f"Image size: {gray_image.shape}")
    
    # Apply current preprocessing
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")
    
    # Analyze contours with different filter criteria
    results = {}
    
    # Test different area ranges
    area_ranges = [
        (50, 1000, "Original loose"),
        (100, 500, "Current strict"), 
        (80, 300, "Medium"),
        (120, 400, "Tight")
    ]
    
    for min_area, max_area, label in area_ranges:
        bubble_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.5:  # Keep current circularity
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.5 < aspect_ratio < 2.0:  # Keep current aspect ratio
                            bubble_candidates.append({
                                'center': (x + w//2, y + h//2),
                                'area': area,
                                'circularity': circularity,
                                'aspect_ratio': aspect_ratio
                            })
        
        bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
        results[label] = len(bubble_candidates)
        print(f"{label} filter ({min_area}-{max_area}): {len(bubble_candidates)} bubbles")
    
    # Test the current approach
    print(f"\n=== CURRENT DETECTION RESULTS ===")
    
    # Use current filters
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 500:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.5:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 2.0:
                        roi = gray_image[y:y+h, x:x+w]
                        if roi.size > 0:
                            mean_darkness = np.mean(roi)
                            if mean_darkness < 230:
                                bubble_candidates.append({
                                    'center': (x + w//2, y + h//2),
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'circularity': circularity,
                                    'roi': roi
                                })
    
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    print(f"Current detection found: {len(bubble_candidates)} valid bubbles")
    print(f"Expected for 20 questions × 5 subjects × 4 options = 400 bubbles")
    
    if len(bubble_candidates) < 400:
        print(f"⚠️  ISSUE: Only found {len(bubble_candidates)} bubbles, need 400!")
        print("   This explains why answers are not read correctly.")
    
    # Analyze the first few detected bubbles
    print(f"\n=== SAMPLE BUBBLE ANALYSIS ===")
    for i in range(min(10, len(bubble_candidates))):
        bubble = bubble_candidates[i]
        roi = bubble['roi']
        
        if roi.size > 0:
            # Test different darkness calculation methods
            _, thresh_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            darkness_otsu = np.sum(thresh_otsu == 255) / thresh_otsu.size
            
            _, thresh_fixed = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            darkness_fixed = np.sum(thresh_fixed == 255) / thresh_fixed.size
            
            mean_brightness = np.mean(roi)
            darkness_mean = (255 - mean_brightness) / 255
            
            # Would this bubble be considered "marked" with different thresholds?
            marked_35 = darkness_otsu >= 0.35
            marked_50 = darkness_otsu >= 0.50
            
            print(f"Bubble {i:2d}: OTSU={darkness_otsu:.3f}, Fixed={darkness_fixed:.3f}, "
                  f"Mean={darkness_mean:.3f} | 35%: {'✓' if marked_35 else '✗'}, "
                  f"50%: {'✓' if marked_50 else '✗'}")
    
    # Test a more relaxed detection
    print(f"\n=== TESTING RELAXED DETECTION ===")
    relaxed_bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 80 < area < 600:  # Wider area range
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Lower circularity requirement
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:  # Wider aspect ratio
                        relaxed_bubbles.append({'area': area, 'circularity': circularity})
    
    print(f"Relaxed detection found: {len(relaxed_bubbles)} potential bubbles")
    
    if len(relaxed_bubbles) >= 400:
        print("✓ Relaxed detection finds enough bubbles - filters are too strict!")
        print("💡 RECOMMENDATION: Loosen the detection criteria")
    else:
        print("✗ Even relaxed detection doesn't find enough bubbles")
        print("💡 RECOMMENDATION: Check if OMR sheet format is different than expected")
    
    return len(bubble_candidates)


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    bubble_count = analyze_bubble_detection_issue(image_path)
    
    print(f"\n=== SUMMARY ===")
    if bubble_count < 200:
        print("🔴 CRITICAL: Very few bubbles detected - detection is too strict")
    elif bubble_count < 350:
        print("🟡 WARNING: Insufficient bubbles detected - detection needs adjustment")
    else:
        print("🟢 GOOD: Sufficient bubbles detected - check darkness thresholds")