#!/usr/bin/env python3
"""
Advanced OMR Processing inspired by professional OMR systems.
Incorporates adaptive thresholding, morphological operations, and template alignment.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def advanced_bubble_detection(image_path, debug=False):
    """
    Advanced bubble detection using morphological operations and adaptive thresholding.
    Based on professional OMR processing techniques.
    """
    
    print(f"=== ADVANCED OMR PROCESSING ===")
    print(f"Processing: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Resize to standard processing dimensions
    target_height, target_width = 1200, 1000
    img = cv2.resize(img, (target_width, target_height))
    
    print(f"Resized to: {img.shape}")
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    # Normalize image
    if img_clahe.max() > img_clahe.min():
        img_normalized = cv2.normalize(img_clahe, None, 0, 255, cv2.NORM_MINMAX)
    else:
        img_normalized = img_clahe
    
    # Gamma correction for better bubble contrast
    gamma = 0.8  # Lower gamma makes dark areas darker
    img_gamma = np.power(img_normalized / 255.0, gamma) * 255
    img_gamma = img_gamma.astype(np.uint8)
    
    # Morphological operations to enhance bubble structures
    # Vertical kernel to detect column structures
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    morph_v = cv2.morphologyEx(img_gamma, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # Truncate to remove very bright areas
    _, morph_v = cv2.threshold(morph_v, 200, 200, cv2.THRESH_TRUNC)
    morph_v = 255 - cv2.normalize(morph_v, None, 0, 255, cv2.NORM_MINMAX)
    
    # Binary threshold for structure detection
    morph_thr = 50  # Adaptive threshold for mobile/scanned images
    _, morph_binary = cv2.threshold(morph_v, morph_thr, 255, cv2.THRESH_BINARY)
    
    # Erosion to clean up noise
    morph_binary = cv2.erode(morph_binary, np.ones((3, 3), np.uint8), iterations=1)
    
    if debug:
        debug_images = {
            'original': img,
            'clahe': img_clahe,
            'gamma': img_gamma,
            'morphed': morph_v,
            'binary': morph_binary
        }
        visualize_processing_steps(debug_images)
    
    # Find contours for bubble detection
    contours, _ = cv2.findContours(morph_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Advanced bubble filtering
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Professional OMR bubble size range
        if 80 < area < 600:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # More lenient circularity for real-world OMR sheets
                if circularity > 0.3:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Allow for slightly oval bubbles
                    if 0.3 < aspect_ratio < 3.0:
                        # Extract ROI from original normalized image
                        roi = img_normalized[y:y+h, x:x+w]
                        
                        if roi.size > 0:
                            bubble_candidates.append({
                                'center': (x + w//2, y + h//2),
                                'bbox': (x, y, w, h),
                                'area': area,
                                'circularity': circularity,
                                'aspect_ratio': aspect_ratio,
                                'roi': roi,
                                'mean_intensity': np.mean(roi)
                            })
    
    # Sort by position (top to bottom, left to right)
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    
    print(f"Found {len(bubble_candidates)} bubble candidates")
    
    # Advanced thresholding using global and local analysis
    bubble_intensities = [b['mean_intensity'] for b in bubble_candidates]
    
    if len(bubble_intensities) > 0:
        # Global threshold using gap analysis
        global_threshold = calculate_global_threshold(bubble_intensities, debug=debug)
        print(f"Global threshold: {global_threshold:.2f}")
        
        # Apply intelligent marking detection
        marked_responses = apply_advanced_marking_detection(
            bubble_candidates, global_threshold, debug=debug
        )
        
        return bubble_candidates, marked_responses, global_threshold
    
    return [], {}, 255


def calculate_global_threshold(intensities, min_jump=15, debug=False):
    """
    Calculate global threshold using gap analysis method.
    Based on the professional OMR approach.
    """
    if len(intensities) < 3:
        return 128  # Default threshold
    
    sorted_intensities = sorted(intensities)
    
    # Find the largest gap in intensity values
    max_jump = min_jump
    threshold = 255
    
    for i in range(1, len(sorted_intensities) - 1):
        # Look at jumps in a small window
        jump = sorted_intensities[i + 1] - sorted_intensities[i - 1]
        if jump > max_jump:
            max_jump = jump
            threshold = sorted_intensities[i - 1] + jump / 2
    
    if debug and len(intensities) > 10:
        plt.figure(figsize=(10, 6))
        plt.hist(intensities, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.1f}')
        plt.xlabel('Bubble Intensity')
        plt.ylabel('Frequency')
        plt.title('Bubble Intensity Distribution with Global Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return threshold


def apply_advanced_marking_detection(bubble_candidates, global_threshold, debug=False):
    """
    Apply advanced marking detection with local threshold refinement.
    """
    marked_bubbles = []
    
    for bubble in bubble_candidates:
        roi = bubble['roi']
        
        if roi.size == 0:
            continue
        
        # Multiple intensity calculation methods
        mean_intensity = np.mean(roi)
        
        # OTSU threshold for this specific bubble
        _, roi_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        otsu_darkness = np.sum(roi_otsu == 255) / roi_otsu.size
        
        # Standard deviation check for uniform marking
        std_intensity = np.std(roi)
        
        # Combined marking criteria
        is_marked = (
            mean_intensity < global_threshold and  # Below global threshold
            otsu_darkness > 0.3 and               # Significant dark content
            std_intensity > 10                    # Some contrast (not uniform)
        )
        
        if is_marked:
            marked_bubbles.append(bubble)
        
        bubble['is_marked'] = is_marked
        bubble['otsu_darkness'] = otsu_darkness
        bubble['std_intensity'] = std_intensity
    
    print(f"Detected {len(marked_bubbles)} marked bubbles out of {len(bubble_candidates)} total")
    
    return {'marked_bubbles': marked_bubbles, 'all_bubbles': bubble_candidates}


def visualize_processing_steps(debug_images):
    """
    Visualize the image processing steps.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    titles = ['Original', 'CLAHE Enhanced', 'Gamma Corrected', 'Morphological', 'Binary', 'Final']
    
    for i, (title, img) in enumerate(zip(titles, debug_images.values())):
        if i < len(axes):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Remove empty subplot
    if len(debug_images) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('advanced_processing_steps.png', dpi=150, bbox_inches='tight')
    plt.show()


def organize_bubbles_to_grid(bubble_candidates, subjects=None, questions_per_subject=20):
    """
    Organize detected bubbles into a logical grid structure.
    """
    if subjects is None:
        subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
    
    # Take first 400 bubbles for the 20x5x4 grid
    question_bubbles = bubble_candidates[:400] if len(bubble_candidates) >= 400 else bubble_candidates
    
    organized = {}
    for subject in subjects:
        organized[subject] = {}
    
    bubbles_per_question = 20  # 5 subjects × 4 options
    
    for q_num in range(questions_per_subject):
        start_idx = q_num * bubbles_per_question
        end_idx = min(start_idx + bubbles_per_question, len(question_bubbles))
        
        if end_idx > start_idx:
            row_bubbles = question_bubbles[start_idx:end_idx]
            # Sort by X position within the row
            row_bubbles.sort(key=lambda b: b['center'][0])
            
            # Divide into subjects
            for s_idx, subject in enumerate(subjects):
                subject_start = s_idx * 4
                subject_end = min(subject_start + 4, len(row_bubbles))
                
                if subject_end > subject_start:
                    subject_bubbles = row_bubbles[subject_start:subject_end]
                    if len(subject_bubbles) >= 4:
                        organized[subject][q_num] = subject_bubbles[:4]
    
    return organized


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    # Run advanced processing
    bubbles, responses, threshold = advanced_bubble_detection(image_path, debug=True)
    
    # Organize into grid
    if bubbles:
        organized = organize_bubbles_to_grid(bubbles)
        
        print(f"\n=== GRID ORGANIZATION ===")
        for subject, questions in organized.items():
            print(f"{subject}: {len(questions)} questions organized")
        
        # Count marked bubbles per subject
        print(f"\n=== MARKED RESPONSES ===")
        marked_bubbles = responses.get('marked_bubbles', [])
        marked_by_subject = {}
        
        # This would need to be integrated with the grid organization
        # for proper subject assignment
        print(f"Total marked bubbles detected: {len(marked_bubbles)}")
    
    print(f"\nAdvanced processing complete!")
    print(f"Check 'advanced_processing_steps.png' for visual analysis")