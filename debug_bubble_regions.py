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

from utils.bubbledetection import _compute_roi_bounds

def debug_bubble_regions(image_path, output_path="debug_regions.jpg"):
    """
    Visualize bubble detection regions on an OMR image to debug positioning issues.
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale for processing, but keep color for visualization
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        display_img = img_array.copy()
    else:
        gray = img_array
        display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print(f"Image shape: {gray.shape}")
    
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    options = ["A", "B", "C", "D"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # Draw regions for first few questions to see alignment
    for subject_idx, subject in enumerate(subjects):
        color = colors[subject_idx]
        
        # Draw first 3 questions for each subject
        for question_num in range(3):
            for option_idx, option in enumerate(options):
                try:
                    y1, y2, x1, x2 = _compute_roi_bounds(gray, subject, question_num, option)
                    
                    # Draw rectangle around each bubble region
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text label
                    label = f"{subject[0]}{question_num+1}{option}"
                    cv2.putText(display_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.3, color, 1)
                    
                    # Extract ROI and check darkness
                    roi = gray[y1:y2, x1:x2]
                    if roi.size > 0:
                        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        filled_ratio = np.sum(thresh == 255) / thresh.size
                        
                        print(f"{subject} Q{question_num+1} {option}: "
                              f"region=({x1},{y1})-({x2},{y2}), "
                              f"size={roi.shape}, filled_ratio={filled_ratio:.3f}")
                    
                except Exception as e:
                    print(f"Error processing {subject} Q{question_num+1} {option}: {e}")
    
    # Save debug image
    cv2.imwrite(output_path, display_img)
    print(f"Debug image saved to: {output_path}")
    
    # Also draw grid lines to show subject boundaries
    height, width = gray.shape
    col_width = width // 5
    row_height = height // 25
    
    # Vertical lines for subjects
    for i in range(1, 5):
        x = i * col_width
        cv2.line(display_img, (x, 0), (x, height), (0, 0, 0), 3)
    
    # Horizontal lines for questions (first few)
    for i in range(1, 6):
        y = i * row_height
        cv2.line(display_img, (0, y), (width, y), (0, 0, 0), 1)
    
    cv2.imwrite(output_path.replace('.jpg', '_with_grid.jpg'), display_img)
    print(f"Grid debug image saved to: {output_path.replace('.jpg', '_with_grid.jpg')}")


if __name__ == "__main__":
    # Debug the problematic image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    debug_bubble_regions(image_path)