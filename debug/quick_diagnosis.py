#!/usr/bin/env python3
"""
Quick diagnosis of bubble detection issues
"""

import cv2
import numpy as np
from PIL import Image

def quick_bubble_analysis(image_path):
    """Quick analysis of bubble detection parameters"""
    
    # Load image
    image = Image.open(image_path)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    print(f"Image shape: {img.shape}")
    
    # Test different approaches
    approaches = {
        'simple_adaptive': lambda: cv2.adaptiveThreshold(
            cv2.GaussianBlur(img, (3, 3), 0), 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        ),
        'otsu': lambda: cv2.threshold(
            cv2.GaussianBlur(img, (5, 5), 0), 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1],
        'fixed_threshold': lambda: cv2.threshold(
            img, 180, 255, cv2.THRESH_BINARY_INV
        )[1]
    }
    
    for name, method in approaches.items():
        binary = method()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by basic criteria
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 1000:  # Very wide range
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.1:  # Very lenient
                        valid_contours.append(contour)
        
        print(f"{name}: {len(valid_contours)} potential bubbles")
        
        if valid_contours:
            areas = [cv2.contourArea(c) for c in valid_contours]
            circularities = [4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2) for c in valid_contours]
            print(f"  Areas: {min(areas):.0f}-{max(areas):.0f}")
            print(f"  Circularities: {min(circularities):.2f}-{max(circularities):.2f}")

if __name__ == "__main__":
    quick_bubble_analysis("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")