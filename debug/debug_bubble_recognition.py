#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def debug_bubble_recognition(image_path):
    """
    Debug tool to visualize exactly how bubbles are being detected and evaluated.
    This will help identify why student answers are not being read correctly.
    """
    
    # Load image
    image = Image.open(image_path)
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    print(f"Analyzing image: {image_path}")
    print(f"Image dimensions: {gray_image.shape}")
    
    # Apply the same preprocessing as in the main detection
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter bubble-like contours with current criteria
    bubble_candidates = []
    rejected_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check all our current criteria
            area_ok = 100 < area < 500
            circularity_ok = circularity > 0.5
            aspect_ok = 0.5 < aspect_ratio < 2.0
            
            roi = gray_image[y:y+h, x:x+w] if roi_valid(y, h, x, w, gray_image.shape) else np.array([])
            brightness_ok = True
            if roi.size > 0:
                mean_darkness = np.mean(roi)
                brightness_ok = mean_darkness < 230
            
            if area_ok and circularity_ok and aspect_ok and brightness_ok:
                bubble_candidates.append({
                    'center': (x + w//2, y + h//2),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'roi': roi,
                    'mean_brightness': np.mean(roi) if roi.size > 0 else 255
                })
            else:
                rejected_contours.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'reasons': [
                        f"area:{area:.0f}" if not area_ok else "",
                        f"circ:{circularity:.2f}" if not circularity_ok else "",
                        f"aspect:{aspect_ratio:.2f}" if not aspect_ok else "",
                        f"bright:{np.mean(roi) if roi.size > 0 else 'N/A'}" if not brightness_ok else ""
                    ]
                })
    
    print(f"Found {len(bubble_candidates)} valid bubbles")
    print(f"Rejected {len(rejected_contours)} contours")
    
    # Sort bubbles by position
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 1. Original image
    axes[0,0].imshow(gray_image, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # 2. Threshold image
    axes[0,1].imshow(thresh, cmap='gray')
    axes[0,1].set_title('Threshold Image')
    axes[0,1].axis('off')
    
    # 3. Valid bubbles
    axes[0,2].imshow(gray_image, cmap='gray')
    for i, bubble in enumerate(bubble_candidates[:50]):  # Show first 50
        x, y, w, h = bubble['bbox']
        color = 'green' if i < 20 else 'blue'  # First 20 in green
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none', alpha=0.8)
        axes[0,2].add_patch(rect)
        # Add bubble number
        axes[0,2].text(x, y-5, str(i), color=color, fontsize=8, fontweight='bold')
    axes[0,2].set_title(f'Valid Bubbles (showing first 50 of {len(bubble_candidates)})')
    axes[0,2].axis('off')
    
    # 4. Rejected contours (sample)
    axes[1,0].imshow(gray_image, cmap='gray')
    for i, rejected in enumerate(rejected_contours[:30]):  # Show first 30
        x, y, w, h = rejected['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                               edgecolor='red', facecolor='none', alpha=0.5)
        axes[1,0].add_patch(rect)
    axes[1,0].set_title(f'Rejected Contours (showing 30 of {len(rejected_contours)})')
    axes[1,0].axis('off')
    
    # 5. Grid mapping visualization - show how bubbles are assigned to questions
    axes[1,1].imshow(gray_image, cmap='gray')
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
    
    # Show first 5 questions mapping
    for q_num in range(min(5, 20)):
        start_idx = q_num * 20
        end_idx = min(start_idx + 20, len(bubble_candidates))
        
        if end_idx > start_idx:
            question_bubbles = bubble_candidates[start_idx:end_idx]
            question_bubbles.sort(key=lambda b: b['center'][0])
            
            for s_idx in range(5):
                subject_start = s_idx * 4
                subject_end = min(subject_start + 4, len(question_bubbles))
                
                if subject_end > subject_start:
                    color = colors[s_idx]
                    for option_idx, bubble in enumerate(question_bubbles[subject_start:subject_end]):
                        x, y, w, h = bubble['bbox']
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                               edgecolor=color, facecolor='none', alpha=0.8)
                        axes[1,1].add_patch(rect)
                        # Label with question and option
                        option_letter = chr(ord('A') + option_idx)
                        axes[1,1].text(x+w//2, y+h//2, f"Q{q_num+1}{option_letter}", 
                                     color=color, fontsize=6, ha='center', va='center',
                                     fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", 
                                                                facecolor='white', alpha=0.8))
    
    # Add legend for subjects
    legend_elements = [patches.Patch(facecolor='white', edgecolor=colors[i], 
                                   linewidth=2, label=subjects[i]) for i in range(5)]
    axes[1,1].legend(handles=legend_elements, loc='upper right', fontsize=8)
    axes[1,1].set_title('Question-Subject Mapping (First 5 Questions)')
    axes[1,1].axis('off')
    
    # 6. Darkness analysis for first few bubbles
    axes[1,2].set_title('Bubble Darkness Analysis')
    if len(bubble_candidates) >= 20:
        darkness_values = []
        bubble_labels = []
        
        for i in range(min(20, len(bubble_candidates))):
            bubble = bubble_candidates[i]
            roi = bubble['roi']
            if roi.size > 0:
                # Calculate darkness using the same method as main code
                _, thresh_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                darkness_ratio = np.sum(thresh_roi == 255) / thresh_roi.size
                darkness_values.append(darkness_ratio)
                bubble_labels.append(f"B{i}")
        
        if darkness_values:
            bars = axes[1,2].bar(range(len(darkness_values)), darkness_values)
            axes[1,2].axhline(y=0.5, color='red', linestyle='--', label='50% threshold')
            axes[1,2].axhline(y=0.35, color='orange', linestyle='--', label='35% threshold')
            axes[1,2].set_xlabel('Bubble Index')
            axes[1,2].set_ylabel('Darkness Ratio')
            axes[1,2].set_xticks(range(len(darkness_values)))
            axes[1,2].set_xticklabels(bubble_labels, rotation=45)
            axes[1,2].legend()
            
            # Color bars based on threshold
            for i, (bar, darkness) in enumerate(zip(bars, darkness_values)):
                if darkness >= 0.5:
                    bar.set_color('darkgreen')  # Would be marked
                elif darkness >= 0.35:
                    bar.set_color('orange')     # Borderline
                else:
                    bar.set_color('lightblue')  # Not marked
    
    plt.tight_layout()
    plt.savefig('bubble_recognition_debug.png', dpi=150, bbox_inches='tight')
    print("\nSaved detailed analysis to 'bubble_recognition_debug.png'")
    
    # Print detailed statistics
    print(f"\n=== BUBBLE DETECTION ANALYSIS ===")
    print(f"Total contours found: {len(contours)}")
    print(f"Valid bubbles: {len(bubble_candidates)}")
    print(f"Rejection rate: {len(rejected_contours)/len(contours)*100:.1f}%")
    
    if bubble_candidates:
        areas = [b['area'] for b in bubble_candidates]
        circularities = [b['circularity'] for b in bubble_candidates]
        brightness = [b['mean_brightness'] for b in bubble_candidates]
        
        print(f"\nBubble Statistics:")
        print(f"  Area: {min(areas):.0f} - {max(areas):.0f} (mean: {np.mean(areas):.0f})")
        print(f"  Circularity: {min(circularities):.2f} - {max(circularities):.2f} (mean: {np.mean(circularities):.2f})")
        print(f"  Brightness: {min(brightness):.0f} - {max(brightness):.0f} (mean: {np.mean(brightness):.0f})")
    
    # Show some rejection reasons
    print(f"\nSample rejection reasons:")
    for i, rejected in enumerate(rejected_contours[:10]):
        reasons = [r for r in rejected['reasons'] if r]
        print(f"  Rejected {i}: {', '.join(reasons)}")
    
    return bubble_candidates


def roi_valid(y, h, x, w, shape):
    """Check if ROI coordinates are valid for the image"""
    return (y >= 0 and y + h <= shape[0] and x >= 0 and x + w <= shape[1])


if __name__ == "__main__":
    import sys
    import os
    
    # Use command line argument or default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    bubbles = debug_bubble_recognition(image_path)
    print(f"\nTo view the detailed analysis, open 'bubble_recognition_debug.png'")
    print(f"For interactive debugging, run the Streamlit app and use Debug Mode")