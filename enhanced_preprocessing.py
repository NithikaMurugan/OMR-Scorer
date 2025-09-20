#!/usr/bin/env python3
"""
Enhanced OMR preprocessing with marker-based alignment and perspective correction.
Inspired by professional OMR systems for improved accuracy.
"""

import cv2
import numpy as np
from PIL import Image
import os

class OMRImagePreprocessor:
    """
    Professional OMR image preprocessor with marker detection and perspective correction.
    """
    
    def __init__(self, min_matching_threshold=0.3, max_matching_variation=0.41):
        self.min_matching_threshold = min_matching_threshold
        self.max_matching_variation = max_matching_variation
        self.marker_rescale_range = (35, 100)
        self.marker_rescale_steps = 10
    
    def preprocess_omr_image(self, image_path, debug=False):
        """
        Comprehensive OMR image preprocessing with alignment correction.
        """
        print(f"=== ENHANCED OMR PREPROCESSING ===")
        print(f"Processing: {image_path}")
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image = image_path
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        print(f"Original image shape: {gray.shape}")
        
        # Step 1: Enhance contrast and normalize
        enhanced = self.enhance_contrast(gray)
        
        # Step 2: Detect and correct perspective (if possible)
        corrected = self.detect_and_correct_perspective(enhanced, debug=debug)
        
        # Step 3: Apply final preprocessing for bubble detection
        final_processed = self.apply_final_preprocessing(corrected)
        
        print(f"Final processed shape: {final_processed.shape}")
        
        if debug:
            self.visualize_preprocessing_steps({
                'original': gray,
                'enhanced': enhanced,
                'corrected': corrected,
                'final': final_processed
            })
        
        return final_processed
    
    def enhance_contrast(self, gray_image):
        """
        Apply advanced contrast enhancement techniques.
        """
        # CLAHE for adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_image)
        
        # Gamma correction for better bubble contrast
        gamma = 0.85
        gamma_corrected = np.power(enhanced / 255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        # Normalize to full range
        normalized = cv2.normalize(gamma_corrected, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def detect_and_correct_perspective(self, image, debug=False):
        """
        Detect OMR sheet boundaries and correct perspective distortion.
        """
        # Try to detect the OMR sheet boundary for perspective correction
        sheet_contour = self.detect_sheet_boundary(image, debug=debug)
        
        if sheet_contour is not None:
            # Apply perspective correction
            corrected = self.apply_perspective_correction(image, sheet_contour)
            print("Applied perspective correction based on sheet boundary")
            return corrected
        else:
            print("No clear sheet boundary detected, using original image")
            return image
    
    def detect_sheet_boundary(self, image, debug=False):
        """
        Detect the main OMR sheet boundary using edge detection and contour analysis.
        """
        # Edge detection for boundary finding
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest rectangular contour (likely the sheet boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to get corner points
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Check if we have a quadrilateral
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            image_area = image.shape[0] * image.shape[1]
            
            # Ensure the detected boundary covers a significant portion of the image
            if area > image_area * 0.3:  # At least 30% of image area
                return approx.reshape(4, 2)
        
        return None
    
    def apply_perspective_correction(self, image, corners):
        """
        Apply four-point perspective transformation.
        """
        # Order the corners: top-left, top-right, bottom-right, bottom-left
        corners = self.order_corner_points(corners)
        
        # Calculate the dimensions of the corrected image
        width_top = np.linalg.norm(corners[0] - corners[1])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(corners[0] - corners[3])
        height_right = np.linalg.norm(corners[1] - corners[2])
        height = int(max(height_left, height_right))
        
        # Define destination points for the corrected image
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
        
        # Apply perspective correction
        corrected = cv2.warpPerspective(image, matrix, (width, height))
        
        return corrected
    
    def order_corner_points(self, corners):
        """
        Order corner points as: top-left, top-right, bottom-right, bottom-left.
        """
        # Sum and difference of coordinates
        sum_coords = corners.sum(axis=1)
        diff_coords = np.diff(corners, axis=1)
        
        # Top-left: smallest sum, bottom-right: largest sum
        top_left = corners[np.argmin(sum_coords)]
        bottom_right = corners[np.argmax(sum_coords)]
        
        # Top-right: smallest difference, bottom-left: largest difference  
        top_right = corners[np.argmin(diff_coords)]
        bottom_left = corners[np.argmax(diff_coords)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left])
    
    def apply_final_preprocessing(self, image):
        """
        Apply final preprocessing optimized for bubble detection.
        """
        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Slight blur to smooth bubble boundaries
        final = cv2.GaussianBlur(filtered, (3, 3), 0)
        
        return final
    
    def visualize_preprocessing_steps(self, steps_dict):
        """
        Visualize the preprocessing steps for debugging.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (step_name, img) in enumerate(steps_dict.items()):
            if i < len(axes):
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'{step_name.title()} Processing')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('enhanced_preprocessing_steps.png', dpi=150, bbox_inches='tight')
        plt.show()


def apply_professional_preprocessing(image_path, debug=False):
    """
    Apply professional-grade preprocessing to an OMR image.
    """
    preprocessor = OMRImagePreprocessor()
    processed_image = preprocessor.preprocess_omr_image(image_path, debug=debug)
    return processed_image


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    # Apply enhanced preprocessing
    processed = apply_professional_preprocessing(image_path, debug=True)
    
    print(f"Enhanced preprocessing complete!")
    print(f"Processed image shape: {processed.shape}")
    print(f"Check 'enhanced_preprocessing_steps.png' for visual analysis")