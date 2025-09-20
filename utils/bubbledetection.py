import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
from typing import Optional, Union, Dict, Any

# Lazy imports to avoid circular dependencies at module import time
# We'll import utils.answerkey and utils.scoring inside helper function


def detect_bubble_in_region(gray_image, subject, question_num, option):
    """
    Detect if a bubble is filled in a specific region of the OMR sheet.
    
    Args:
        gray_image: Grayscale image (numpy array)
        subject: Subject name (e.g., "Python", "EDA")
        question_num: Question number (0-19)
        option: Option letter ("A", "B", "C", "D")
    
    Returns:
        bool: True if bubble appears to be filled
    """
    
    # Define approximate regions for OMR sheets
    # These coordinates need to be adjusted based on your actual OMR sheet layout
    
    height, width = gray_image.shape
    
    # Calculate approximate positions based on standard OMR layout
    # Assuming subjects are arranged horizontally in 5 columns
    subject_list = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    subject_idx = subject_list.index(subject)
    
    # Calculate subject column position
    col_width = width // 5
    subject_x_start = subject_idx * col_width
    subject_x_end = (subject_idx + 1) * col_width
    
    # Calculate question row position (20 questions per subject)
    row_height = height // 25  # Approximate space for 20 questions + header
    question_y_start = int(row_height * (question_num + 2))  # +2 for header space
    question_y_end = question_y_start + row_height
    
    # Calculate option position within the question row
    option_idx = ord(option) - ord('A')  # A=0, B=1, C=2, D=3
    option_width = (subject_x_end - subject_x_start) // 4
    option_x_start = subject_x_start + (option_idx * option_width)
    option_x_end = option_x_start + option_width
    
    # Extract the region of interest
    try:
        roi = gray_image[question_y_start:question_y_end, option_x_start:option_x_end]
        
        if roi.size == 0:
            return False
        
        # Apply threshold to detect filled bubbles
        # Filled bubbles will be darker than unfilled ones
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate the percentage of dark pixels (filled area)
        filled_ratio = np.sum(thresh == 255) / thresh.size
        
        # Consider bubble filled if more than 15% of the area is dark
        # This threshold may need adjustment based on your OMR sheets
        threshold = 0.15
        
        return filled_ratio > threshold
        
    except Exception as e:
        print(f"Error detecting bubble for {subject}, Q{question_num+1}, {option}: {e}")
        return False


def detect_bubbles_with_contours(gray_image, subject, question_num, option):
    """
    Alternative bubble detection method using contour detection.
    More robust for different OMR sheet layouts.
    """
    
    height, width = gray_image.shape
    
    # Similar region calculation as above
    subject_list = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    subject_idx = subject_list.index(subject)
    
    col_width = width // 5
    subject_x_start = subject_idx * col_width
    subject_x_end = (subject_idx + 1) * col_width
    
    row_height = height // 25
    question_y_start = int(row_height * (question_num + 2))
    question_y_end = question_y_start + row_height
    
    option_idx = ord(option) - ord('A')
    option_width = (subject_x_end - subject_x_start) // 4
    option_x_start = subject_x_start + (option_idx * option_width)
    option_x_end = option_x_start + option_width
    
    try:
        roi = gray_image[question_y_start:question_y_end, option_x_start:option_x_end]
        
        if roi.size == 0:
            return False
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any significant contours (filled bubbles) are found
        for contour in contours:
            area = cv2.contourArea(contour)
            # If contour area is significant relative to ROI size, consider it filled
            if area > (roi.size * 0.1):  # 10% of ROI area
                return True
        
        return False
    except Exception as e:
        print(f"Error in contour detection for {subject}, Q{question_num+1}, {option}: {e}")
        return False


def _compute_roi_bounds(gray_image, subject: str, question_num: int, option: str):
    """
    Internal helper to compute ROI bounds (y1:y2, x1:x2) for a given subject/question/option.
    Returns tuple (y1, y2, x1, x2). Raises if subject invalid.
    """
    height, width = gray_image.shape

    subject_list = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    subject_idx = subject_list.index(subject)

    col_width = width // 5
    subject_x_start = subject_idx * col_width
    subject_x_end = (subject_idx + 1) * col_width

    row_height = height // 25
    question_y_start = int(row_height * (question_num + 2))
    question_y_end = question_y_start + row_height

    option_idx = ord(option) - ord('A')
    option_width = (subject_x_end - subject_x_start) // 4
    option_x_start = subject_x_start + (option_idx * option_width)
    option_x_end = option_x_start + option_width

    return question_y_start, question_y_end, option_x_start, option_x_end


def iter_bubble_rois(gray_image):
    """
    Generator yielding (subject, question_num, option, roi) over all bubbles using the
    same grid mapping as detection. Expects grayscale image.
    """
    if len(gray_image.shape) != 2:
        raise ValueError("iter_bubble_rois expects a grayscale image")

    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    options = ["A", "B", "C", "D"]
    for subject in subjects:
        for question_num in range(20):
            for option in options:
                y1, y2, x1, x2 = _compute_roi_bounds(gray_image, subject, question_num, option)
                roi = gray_image[y1:y2, x1:x2]
                yield subject, question_num, option, roi


def is_option_marked(gray_image, subject, question_num, option, cnn_model=None, threshold=0.35) -> bool:
    """
    Decide if an option bubble is marked using standard OMR evaluation conditions.
    
    Args:
        gray_image: Grayscale image
        subject: Subject name
        question_num: Question number (0-19)
        option: Option letter (A, B, C, D)
        cnn_model: Optional CNN model for classification
        threshold: Darkness threshold (30-40% standard)
    
    Returns:
        bool: True if bubble is marked according to OMR standards
    """
    y1, y2, x1, x2 = _compute_roi_bounds(gray_image, subject, question_num, option)
    roi = gray_image[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    if cnn_model is not None:
        try:
            # Use CNN if available and confidence is high
            resized = cv2.resize(roi, (28, 28))
            normalized = resized.astype('float32') / 255.0
            reshaped = normalized.reshape(1, 28, 28, 1)
            pred = cnn_model.predict(reshaped, verbose=0)
            
            if pred.shape[1] > 1:
                p_filled = float(pred[0][1])
            else:
                p_filled = float(pred[0][0])
            
            # Use CNN result if confidence is high, otherwise fall back to threshold
            if p_filled > 0.7 or p_filled < 0.3:  # High confidence
                return p_filled >= 0.5
        except Exception as e:
            print(f"CNN inference failed, falling back to threshold: {e}")

    # Standard OMR evaluation: darkness threshold method
    # Invert image so filled bubbles have high values
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    filled_ratio = np.sum(thresh == 255) / thresh.size
    
    # Standard OMR condition: 30-40% darkness threshold
    return filled_ratio >= threshold

def extract_responses(img):
    """
    Detect bubbles in each subject grid and classify as marked/unmarked.
    Uses improved grid detection and actual bubble analysis.
    Returns {subject: [answers]}.
    """
    # Handle both PIL Images and numpy arrays
    if isinstance(img, Image.Image):
        # Convert PIL Image to numpy array
        img = np.array(img)
        print("Converted PIL Image to numpy array for bubble detection")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"Processing image for bubble detection: shape={gray.shape}, dtype={gray.dtype}")
    
    # Detect actual bubble grid using the working implementation
    organized_bubbles = detect_actual_bubble_grid(gray)
    
    # Use actual subject names that match the answer key
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    
    print(f"Processing for subjects: {subjects}")
    
    responses = {}
    
    # Process each subject using the actual bubble detection
    for subject in subjects:
        responses[subject] = []
        
        print(f"Processing {subject} section...")
        
        # Process 20 questions per subject
        for question_num in range(20):
            response = get_question_responses_with_actual_bubbles(organized_bubbles, subject, question_num)
            responses[subject].append(response)
    
    return responses
def detect_actual_bubble_grid(gray_image):
    """
    Balanced high-precision bubble detection optimized for maximum accuracy.
    """
    height, width = gray_image.shape
    
    # Balanced preprocessing for accuracy without over-filtering
    processed_image = apply_balanced_precision_preprocessing(gray_image)
    
    # Multi-approach thresholding with balanced parameters
    # Approach 1: Primary adaptive threshold
    binary1 = cv2.adaptiveThreshold(
        processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Approach 2: OTSU threshold for contrast areas
    _, binary2 = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Approach 3: Conservative backup threshold
    binary3 = cv2.adaptiveThreshold(
        processed_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Smart combination based on image analysis
    image_mean = np.mean(processed_image)
    image_std = np.std(processed_image)
    
    if image_std > 40:  # High contrast
        combined_binary = cv2.bitwise_or(binary1, binary2)
    else:  # Lower contrast - use all methods
        combined_binary = cv2.bitwise_or(binary1, cv2.bitwise_or(binary2, binary3))
    
    # Gentle morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, kernel_clean)
    
    # Find all contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} total contours")
    
    bubble_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Balanced area range - not too strict
        if 30 < area < 1000:  
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Reasonable circularity threshold
                if circularity > 0.25:  
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Reasonable aspect ratio
                    if 0.4 < aspect_ratio < 2.5:  
                        roi = gray_image[y:y+h, x:x+w]
                        
                        if roi.size > 0:
                            mean_intensity = np.mean(roi)
                            roi_std = np.std(roi)
                            
                            # Simple quality score without over-filtering
                            quality_score = circularity * 0.7 + (roi_std / 255.0) * 0.3
                            
                            # More lenient quality threshold
                            if quality_score > 0.15:  
                                bubble_candidates.append({
                                    'center': (x + w//2, y + h//2),
                                    'bbox': (x, y, w, h),
                                    'area': area,
                                    'circularity': circularity,
                                    'aspect_ratio': aspect_ratio,
                                    'roi': roi,
                                    'mean_intensity': mean_intensity,
                                    'quality_score': quality_score
                                })
    
    print(f"Found {len(bubble_candidates)} balanced-quality bubble candidates")
    
    # Use adaptive grid with less strict requirements
    organized_bubbles = create_balanced_grid(bubble_candidates, height, width)
    
    return organized_bubbles


def apply_balanced_precision_preprocessing(gray_image):
    """
    Balanced preprocessing for accuracy without over-processing.
    """
    # Gentle noise reduction
    denoised = cv2.medianBlur(gray_image, 3)
    
    # Moderate contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Light gamma correction
    gamma = 1.1
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    return gamma_corrected


def create_balanced_grid(bubble_candidates, image_height, image_width):
    """
    CRITICAL FIX: Identify actual question rows to solve blank responses in Q14-20.
    """
    if len(bubble_candidates) < 100:  
        print(f"⚠️  WARNING: Only {len(bubble_candidates)} quality bubbles found, using fallback")
        return fallback_grid_organization(bubble_candidates)
    
    # Sort by position
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    
    # Detect rows with reasonable tolerance
    rows = detect_balanced_bubble_rows(bubble_candidates, tolerance=25)
    
    print(f"Detected {len(rows)} balanced bubble rows")
    
    if len(rows) < 15:  
        print(f"⚠️  WARNING: Only detected {len(rows)} rows, using fallback")
        return fallback_grid_organization(bubble_candidates)
    
    # CRITICAL FIX: Identify actual question rows instead of just taking best scored ones
    # Question rows should have ~20 bubbles (5 subjects × 4 options = 20 bubbles per row)
    question_rows = []
    
    print("Analyzing rows to find actual question rows...")
    for i, row in enumerate(rows):
        row_bubble_count = len(row)
        row_y = row[0]['center'][1]
        
        # Question rows typically have 18-25 bubbles (allowing some variance)
        if row_bubble_count >= 18 and row_bubble_count <= 25:
            question_rows.append((row_y, row))
            print(f"  Question row found: Y={row_y}, bubbles={row_bubble_count}")
    
    print(f"Found {len(question_rows)} potential question rows")
    
    # Sort question rows by Y position (top to bottom)
    question_rows.sort(key=lambda x: x[0])
    
    # Take the best 20 question rows, or all if less than 20
    if len(question_rows) >= 20:
        final_rows = [row for _, row in question_rows[:20]]
        print(f"Using top 20 question rows")
    else:
        final_rows = [row for _, row in question_rows]
        print(f"⚠️  Using all {len(final_rows)} available question rows")
        
        # Pad with empty rows if needed
        while len(final_rows) < 20:
            final_rows.append([])
    
    # Organize into subjects
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    organized_bubbles = {subject: {} for subject in subjects}
    
    for q_idx, row_bubbles in enumerate(final_rows):
        if not row_bubbles:  # Handle empty rows
            for subject in subjects:
                organized_bubbles[subject][q_idx] = [None, None, None, None]
            continue
            
        # Sort by X position (left to right)
        row_bubbles.sort(key=lambda b: b['center'][0])
        
        # Enhanced distribution to ensure all subjects get proper coverage
        total_bubbles = len(row_bubbles)
        bubbles_per_subject = total_bubbles // 5
        remainder = total_bubbles % 5
        
        current_idx = 0
        for s_idx, subject in enumerate(subjects):
            # Distribute remainder bubbles to first few subjects
            section_size = bubbles_per_subject + (1 if s_idx < remainder else 0)
            end_idx = min(current_idx + section_size, len(row_bubbles))
            section_bubbles = row_bubbles[current_idx:end_idx]
            current_idx = end_idx
            
            if len(section_bubbles) >= 4:
                organized_bubbles[subject][q_idx] = section_bubbles[:4]
            elif len(section_bubbles) > 0:
                padded = section_bubbles + [None] * (4 - len(section_bubbles))
                organized_bubbles[subject][q_idx] = padded
            else:
                organized_bubbles[subject][q_idx] = [None, None, None, None]
    
    # Enhanced reporting
    for subject in subjects:
        question_count = len(organized_bubbles[subject])
        filled_questions = len([q for q, bubbles in organized_bubbles[subject].items() 
                              if bubbles and bubbles[0] is not None])
        print(f"{subject}: {filled_questions}/{question_count} questions have bubbles")
    
    # Report success
    total_organized = sum(len(subject_data) for subject_data in organized_bubbles.values())
    print(f"FIXED organization: {total_organized} questions across {len(subjects)} subjects")
    
    return organized_bubbles
    print(f"Successfully organized {total_organized} balanced questions across {len(subjects)} subjects")
    
    return organized_bubbles


def detect_balanced_bubble_rows(bubbles, tolerance=25):
    """
    Enhanced row detection to capture all questions reliably.
    """
    if not bubbles:
        return []
    
    # Sort by Y coordinate first
    sorted_bubbles = sorted(bubbles, key=lambda b: b['center'][1])
    
    rows = []
    current_row = [sorted_bubbles[0]]
    
    # Use adaptive tolerance based on image characteristics
    base_tolerance = tolerance
    image_height = max(b['center'][1] for b in sorted_bubbles)
    adaptive_tolerance = max(base_tolerance, image_height // 50)  # Scale with image
    
    for bubble in sorted_bubbles[1:]:
        current_y = bubble['center'][1]
        row_y = current_row[0]['center'][1]
        
        # Check if bubble belongs to current row
        if abs(current_y - row_y) <= adaptive_tolerance:
            current_row.append(bubble)
        else:
            # Finalize current row with more lenient requirements
            if len(current_row) >= 8:  # Lower minimum for better coverage
                # Sort row by X coordinate
                current_row.sort(key=lambda b: b['center'][0])
                rows.append(current_row)
            
            # Start new row
            current_row = [bubble]
    
    # Add final row
    if len(current_row) >= 8:
        current_row.sort(key=lambda b: b['center'][0])
        rows.append(current_row)
    
    # Quality check: ensure we have enough rows for all questions
    if len(rows) < 20:
        print(f"Warning: Only {len(rows)} rows detected, retrying with more lenient settings...")
        # Retry with even more lenient settings
        return detect_balanced_bubble_rows_fallback(sorted_bubbles)
    
    print(f"Detected {len(rows)} balanced bubble rows with tolerance={adaptive_tolerance}")
    return rows


def detect_balanced_bubble_rows_fallback(sorted_bubbles):
    """
    Fallback row detection for challenging cases.
    """
    rows = []
    current_row = [sorted_bubbles[0]]
    
    # Very lenient tolerance for fallback
    tolerance = 35
    
    for bubble in sorted_bubbles[1:]:
        if abs(bubble['center'][1] - current_row[0]['center'][1]) <= tolerance:
            current_row.append(bubble)
        else:
            # Accept even smaller rows in fallback mode
            if len(current_row) >= 5:  # Very lenient minimum
                current_row.sort(key=lambda b: b['center'][0])
                rows.append(current_row)
            current_row = [bubble]
    
    # Add final row
    if len(current_row) >= 5:
        current_row.sort(key=lambda b: b['center'][0])
        rows.append(current_row)
    
    print(f"Fallback detected {len(rows)} rows with lenient tolerance={tolerance}")
    return rows


def create_adaptive_grid(bubble_candidates, image_height, image_width):
    """
    Enhanced adaptive grid structure to identify and use actual question rows.
    """
    if len(bubble_candidates) < 100:  # Need minimum bubbles for grid detection
        print(f"⚠️  WARNING: Only {len(bubble_candidates)} bubbles found, cannot form reliable grid")
        return fallback_grid_organization(bubble_candidates)
    
    # Sort all bubbles by position (top to bottom, left to right)
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    
    # Detect row structure using enhanced row detection
    rows = detect_balanced_bubble_rows(bubble_candidates, tolerance=25)
    
    print(f"Detected {len(rows)} potential bubble rows")
    
    if len(rows) < 20:  # Need at least 20 rows for 20 questions
        print(f"⚠️  WARNING: Only detected {len(rows)} rows, expected ~20")
        return fallback_grid_organization(bubble_candidates)
    
    # CRITICAL FIX: Identify actual question rows instead of just taking first 20
    # Question rows should have ~20 bubbles (5 subjects × 4 options)
    question_rows = []
    
    for i, row in enumerate(rows):
        row_bubble_count = len(row)
        row_y = row[0]['center'][1]
        
        # Question rows typically have 15-25 bubbles (allowing some variance)
        if row_bubble_count >= 15 and row_bubble_count <= 25:
            question_rows.append((row_y, row))
            print(f"Question row candidate {len(question_rows)}: Y={row_y}, bubbles={row_bubble_count}")
    
    # Sort question rows by Y position and take the best 20
    question_rows.sort(key=lambda x: x[0])  # Sort by Y coordinate
    
    if len(question_rows) < 20:
        print(f"⚠️  WARNING: Only found {len(question_rows)} question rows, need 20")
        # Fall back to using all available rows
        final_question_rows = [row for _, row in question_rows]
        # Pad with empty rows if needed
        while len(final_question_rows) < 20:
            final_question_rows.append([])
    else:
        # Take the 20 best question rows
        final_question_rows = [row for _, row in question_rows[:20]]
    
    print(f"Using {len(final_question_rows)} question rows for organization")
    
    # Organize into subject structure with enhanced coverage
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    organized_bubbles = {subject: {} for subject in subjects}
    
    for q_idx, row_bubbles in enumerate(final_question_rows):
        if not row_bubbles:  # Skip empty rows
            # Create empty entries for all subjects
            for subject in subjects:
                organized_bubbles[subject][q_idx] = [None, None, None, None]
            continue
            
        # Sort row bubbles by X position (left to right)
        row_bubbles.sort(key=lambda b: b['center'][0])
        
        # Enhanced distribution logic to ensure all subjects get bubbles
        total_bubbles = len(row_bubbles)
        base_per_subject = total_bubbles // 5
        remainder = total_bubbles % 5
        
        bubble_index = 0
        
        for s_idx, subject in enumerate(subjects):
            # Distribute remainder bubbles to first few subjects
            bubbles_for_subject = base_per_subject + (1 if s_idx < remainder else 0)
            
            if bubble_index >= len(row_bubbles):
                # If we run out of bubbles, create empty placeholders
                organized_bubbles[subject][q_idx] = [None, None, None, None]
                continue
                
            # Extract bubbles for this subject
            end_index = min(bubble_index + bubbles_for_subject, len(row_bubbles))
            subject_bubbles = row_bubbles[bubble_index:end_index]
            bubble_index = end_index
            
            if len(subject_bubbles) >= 4:
                # Take exactly 4 options (A, B, C, D)
                organized_bubbles[subject][q_idx] = subject_bubbles[:4]
            elif len(subject_bubbles) > 0:
                # Pad with None for missing options
                padded = subject_bubbles + [None] * (4 - len(subject_bubbles))
                organized_bubbles[subject][q_idx] = padded
            else:
                # No bubbles available for this subject in this row
                organized_bubbles[subject][q_idx] = [None, None, None, None]
    
    # Quality check and reporting
    for subject in subjects:
        question_count = len(organized_bubbles[subject])
        max_q = max(organized_bubbles[subject].keys()) if organized_bubbles[subject] else -1
        print(f"{subject}: Found bubbles for {question_count} questions (max Q{max_q})")
    
    # Report organization success
    total_organized = sum(len(subject_data) for subject_data in organized_bubbles.values())
    print(f"Enhanced organization: {total_organized} questions across {len(subjects)} subjects")
    
    return organized_bubbles


def detect_bubble_rows(bubbles, tolerance=25):
    """
    Group bubbles into rows based on Y coordinate proximity.
    """
    if not bubbles:
        return []
    
    rows = []
    current_row = [bubbles[0]]
    
    for bubble in bubbles[1:]:
        # If Y coordinate is within tolerance of current row, add to row
        if abs(bubble['center'][1] - current_row[0]['center'][1]) <= tolerance:
            current_row.append(bubble)
        else:
            # Start new row
            if len(current_row) >= 4:  # Only consider rows with enough bubbles
                rows.append(current_row)
            current_row = [bubble]
    
    # Add final row
    if len(current_row) >= 4:
        rows.append(current_row)
    
    return rows


def fallback_grid_organization(bubble_candidates):
    """
    Fallback organization when adaptive grid detection fails.
    """
    print("Using fallback grid organization...")
    
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    organized_bubbles = {subject: {} for subject in subjects}
    
    # Sort bubbles and distribute evenly
    bubble_candidates.sort(key=lambda b: (b['center'][1], b['center'][0]))
    
    # Distribute available bubbles across 20 questions × 5 subjects × 4 options
    bubbles_per_question = len(bubble_candidates) // 20 if len(bubble_candidates) >= 20 else 1
    
    for q_num in range(20):
        start_idx = q_num * bubbles_per_question
        end_idx = min(start_idx + bubbles_per_question, len(bubble_candidates))
        
        if start_idx < len(bubble_candidates):
            question_bubbles = bubble_candidates[start_idx:end_idx]
            question_bubbles.sort(key=lambda b: b['center'][0])  # Sort by X
            
            bubbles_per_subject = max(1, len(question_bubbles) // 5)
            
            for s_idx, subject in enumerate(subjects):
                subj_start = s_idx * bubbles_per_subject
                subj_end = min(subj_start + bubbles_per_subject, len(question_bubbles))
                
                if subj_start < len(question_bubbles):
                    subject_bubbles = question_bubbles[subj_start:subj_end]
                    
                    # Ensure exactly 4 bubbles per question (pad with None if needed)
                    if len(subject_bubbles) >= 4:
                        organized_bubbles[subject][q_num] = subject_bubbles[:4]
                    else:
                        padded = subject_bubbles + [None] * (4 - len(subject_bubbles))
                        organized_bubbles[subject][q_num] = padded
    
    return organized_bubbles


def apply_professional_image_enhancement(gray_image):
    """
    Apply balanced professional image enhancement for better bubble detection.
    """
    # More conservative CLAHE to avoid over-enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_image)
    
    # Lighter gamma correction
    gamma = 0.95  # Closer to 1.0 for less aggressive correction
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Normalize to full dynamic range
    normalized = cv2.normalize(gamma_corrected, None, 0, 255, cv2.NORM_MINMAX)
    
    # Light bilateral filter - reduce parameters to be less aggressive
    filtered = cv2.bilateralFilter(normalized, 5, 50, 50)
    
    return filtered


def get_question_responses_with_actual_bubbles(organized_bubbles, subject, question_num):
    """
    Professional-grade response detection with dynamic threshold calculation.
    """
    if subject not in organized_bubbles or question_num not in organized_bubbles[subject]:
        return ""
    
    bubbles = organized_bubbles[subject][question_num]
    if len(bubbles) < 4:
        return ""
    
    options = ["A", "B", "C", "D"]
    bubble_data = []
    
    # Collect comprehensive bubble analysis data
    for i, (option, bubble) in enumerate(zip(options, bubbles)):
        if bubble is None or bubble['roi'].size == 0:
            bubble_data.append({'option': option, 'fill_percent': 0.0, 'is_valid': False})
            continue
        
        roi = bubble['roi']
        
        # Professional fill percentage calculation inspired by the shared system
        fill_percent = calculate_bubble_fill_percentage(roi)
        
        bubble_data.append({
            'option': option, 
            'fill_percent': fill_percent,
            'mean_intensity': np.mean(roi),
            'std_dev': np.std(roi),
            'is_valid': True
        })
    
    # Dynamic threshold calculation like the professional system
    valid_bubbles = [b for b in bubble_data if b['is_valid']]
    if len(valid_bubbles) < 4:
        return ""
    
    fill_percentages = [b['fill_percent'] for b in valid_bubbles]
    threshold = calculate_dynamic_bubble_threshold(fill_percentages)
    
    # Detect marked bubbles using calculated threshold
    marked_options = []
    for bubble_info in valid_bubbles:
        if bubble_info['fill_percent'] >= threshold:
            marked_options.append(bubble_info['option'])
    
    # Apply professional OMR evaluation rules
    return apply_professional_omr_rules(marked_options, bubble_data, threshold)


def calculate_bubble_fill_percentage(roi):
    """
    Simplified and accurate fill percentage calculation matching manual analysis.
    Based on the discovery that manual analysis gives accurate results.
    """
    if roi.size == 0:
        return 0.0
    
    # Simple, reliable method that matches manual analysis
    # Count dark pixels (below 180 threshold)
    dark_pixels = np.sum(roi < 180)
    total_pixels = roi.size
    fill_ratio = dark_pixels / total_pixels
    
    # Apply light adjustment for consistency
    # Manual analysis showed C option should be ~0.5, algorithm was over-calculating
    
    # Add some refinement for edge cases
    mean_intensity = np.mean(roi)
    
    # If very light (mean > 220), likely unfilled
    if mean_intensity > 220:
        fill_ratio *= 0.5
    
    # If moderately dark (mean < 160), likely filled
    elif mean_intensity < 160:
        fill_ratio = min(1.0, fill_ratio * 1.2)
    
    return min(1.0, fill_ratio)  # Cap at 100%


def calculate_dynamic_bubble_threshold(fill_percentages):
    """
    Ultra-sensitive threshold calculation for light pencil marks (0.24-0.27 range).
    This OMR sheet has very light fills that require aggressive detection.
    """
    if len(fill_percentages) < 2:
        return 0.18  # Much more aggressive default for light marks
    
    # Sort fill percentages
    sorted_fills = sorted(fill_percentages)
    
    # Enhanced statistical analysis optimized for light marks
    mean_fill = np.mean(sorted_fills)
    std_fill = np.std(sorted_fills)
    median_fill = np.median(sorted_fills)
    q75 = np.percentile(sorted_fills, 75)
    q25 = np.percentile(sorted_fills, 25)
    max_fill = max(sorted_fills)
    min_fill = min(sorted_fills)
    
    print(f"  Fill statistics: min={min_fill:.3f}, max={max_fill:.3f}, mean={mean_fill:.3f}, std={std_fill:.3f}")
    
    # Method 1: Gap detection optimized for light marks
    max_gap = 0.0
    gap_threshold = 0.18
    
    for i in range(1, len(sorted_fills)):
        gap = sorted_fills[i] - sorted_fills[i-1]
        if gap > max_gap and gap > 0.02:  # Much lower gap requirement (2% instead of 6%)
            max_gap = gap
            gap_threshold = sorted_fills[i-1] + gap * 0.2  # Place threshold at 20% of gap
    
    print(f"  Gap analysis: max_gap={max_gap:.3f}, gap_threshold={gap_threshold:.3f}")
    
    # Method 2: Optimized for light pencil marks (0.24-0.27 range)
    range_fill = max_fill - min_fill
    
    if range_fill > 0.12:  # Good variation - likely has marks
        if max_fill > 0.3:  # Even modest marks detected
            # Use ultra-sensitive threshold for light marks
            threshold = 0.20  # Much lower threshold for light detection
        elif max_gap > 0.03:  # Small gap found
            threshold = gap_threshold
        else:
            # Use statistical approach with extreme sensitivity
            threshold = min(0.22, mean_fill + std_fill * 0.5)
    
    elif range_fill > 0.06:  # Medium variation (smaller range for light marks)
        # Ultra-sensitive threshold for medium variation
        threshold = max(0.18, min(0.22, q75 * 0.8))
    
    else:  # Low variation - all bubbles similar
        # Very aggressive threshold for light marks
        threshold = max(0.18, max_fill * 0.75)  # 75% of maximum for light marks
    
    # Optimized bounds for light pencil marks
    if max_fill > 0.4:  # Moderate marks present
        threshold = max(threshold, 0.22)  # Don't go too low with moderate marks
    elif max_fill < 0.25:  # All marks are very light
        threshold = max_fill * 0.85  # 85% of maximum for very light marks
    else:
        # Standard bounds optimized for light marks (0.24-0.27 range)
        threshold = max(0.16, min(0.24, threshold))
    
    print(f"  Final threshold: {threshold:.3f}")
    return threshold


def apply_professional_omr_rules(marked_options, bubble_data, threshold):
    """
    Ultra-aggressive OMR rules optimized for light pencil marks (0.24-0.27 range).
    """
    if len(marked_options) == 0:
        return ""
    
    elif len(marked_options) == 1:
        # Single mark - be very lenient for light marks
        marked_bubble = next(b for b in bubble_data if b['option'] == marked_options[0] and b['is_valid'])
        
        # Extremely lenient for single marks in light pencil scenario
        if marked_bubble['fill_percent'] > threshold * 0.95:  # Just 5% buffer above threshold
            return marked_options[0]
        else:
            return ""  # Mark too weak
    
    elif len(marked_options) > 1:
        # MULTIPLE marks - apply ultra-aggressive disambiguation for light marks
        marked_bubbles = [b for b in bubble_data if b['option'] in marked_options and b['is_valid']]
        
        if len(marked_bubbles) > 1:
            # Sort by fill percentage (highest first)
            marked_bubbles.sort(key=lambda x: x['fill_percent'], reverse=True)
            
            highest_fill = marked_bubbles[0]['fill_percent']
            second_fill = marked_bubbles[1]['fill_percent']
            
            # Ultra-aggressive disambiguation optimized for light marks
            fill_gap = highest_fill - second_fill
            relative_difference = fill_gap / highest_fill if highest_fill > 0 else 0
            
            print(f"    Disambiguation: highest={highest_fill:.3f}, second={second_fill:.3f}, gap={fill_gap:.3f}")
            
            # Much more lenient criteria for light marks
            tiny_gap = fill_gap > 0.02   # Just 2% absolute difference (was 5%)
            any_relative = relative_difference > 0.05  # Just 5% relative difference (was 8%)
            light_mark = highest_fill > 0.20  # Much lower threshold (20% vs 35%)
            weaker_secondary = second_fill < 0.25   # Lower secondary threshold
            minimal_contrast = fill_gap > 0.015 and highest_fill > 0.22  # Minimal contrast for light marks
            
            # Count satisfied criteria - need just 1 for ultra-aggressive mode
            disambiguation_score = sum([
                tiny_gap,
                any_relative,
                light_mark,
                weaker_secondary,
                minimal_contrast
            ])
            
            print(f"    Criteria: gap={tiny_gap}, rel={any_relative}, light={light_mark}, weak2nd={weaker_secondary}, contrast={minimal_contrast}")
            print(f"    Score: {disambiguation_score}/5")
            
            # Ultra-lenient acceptance - just need 1 criteria
            if disambiguation_score >= 1:
                return marked_bubbles[0]['option']
            
            # Fallback cases for light marks - always pick the highest if there's ANY difference
            if highest_fill > second_fill:  # Any difference at all
                return marked_bubbles[0]['option']
        
        # If somehow still undecided, pick the first marked option
        return marked_options[0] if marked_options else "MULTIPLE"
    
    return "UNCLEAR"


def calculate_adaptive_threshold(intensities, min_gap=20):
    """
    Calculate adaptive threshold using professional gap analysis method.
    """
    if len(intensities) < 3:
        return 150  # Default threshold
    
    sorted_intensities = sorted(intensities)
    
    # Find the largest gap between intensity values
    max_gap = min_gap
    threshold = 150  # Default
    
    for i in range(1, len(sorted_intensities) - 1):
        # Calculate gap using a small window
        gap = sorted_intensities[i + 1] - sorted_intensities[i - 1]
        if gap > max_gap:
            max_gap = gap
            threshold = sorted_intensities[i - 1] + gap / 2
    
    # Ensure threshold is within reasonable bounds
    threshold = max(100, min(200, threshold))
    
    return threshold

def extract_responses(img):
    """
    Extract responses from OMR sheet image using actual bubble detection.
    Returns {subject: [answers]}.
    """
    # Handle both PIL Images and numpy arrays
    if isinstance(img, Image.Image):
        img = np.array(img)
        print("Converted PIL Image to numpy array for bubble detection")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    print(f"Processing image for bubble detection: shape={gray.shape}, dtype={gray.dtype}")
    
    # Detect actual bubble layout
    organized_bubbles = detect_actual_bubble_grid(gray)
    
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    responses = {}
    
    for subject in subjects:
        responses[subject] = []
        
        if subject in organized_bubbles:
            # Get responses for questions we found
            max_question = max(organized_bubbles[subject].keys()) if organized_bubbles[subject] else -1
            print(f"{subject}: Found bubbles for {len(organized_bubbles[subject])} questions (max Q{max_question})")
            
            for question_num in range(20):  # 20 questions per subject
                if question_num in organized_bubbles[subject]:
                    response = get_question_responses_with_actual_bubbles(organized_bubbles, subject, question_num)
                else:
                    response = ""  # No bubbles found for this question
                responses[subject].append(response)
        else:
            # No bubbles found for this subject
            print(f"{subject}: No bubbles detected")
            responses[subject] = [""] * 20
    
    return responses
    
    # Load CNN model if available
    cnn_model = None
    model_path = "Models/cnn_modal.h5"
    if os.path.exists(model_path):
        try:
            cnn_model = load_model(model_path)
        except Exception as e:
            print(f"Could not load CNN model: {e}")
    
    # Use actual subject names that match the answer key
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    print(f"Processing for subjects: {subjects}")
    
    # Apply OMR evaluation conditions for each subject
    for i, subject in enumerate(subjects):
        responses[subject] = []
        print(f"Processing {subject} section...")
        
        # For each subject, process 20 questions using OMR rules
        for question_num in range(20):
            # Apply standard OMR evaluation conditions
            response = get_question_responses_with_omr_rules(gray, subject, question_num, cnn_model)
            responses[subject].append(response)
        
        print(f"Completed {subject}: {len(responses[subject])} responses")
    
    print(f"Bubble detection completed. Results: {responses}")
    return responses

def detect_bubble_grid(img):
    """
    Detect bubble positions using contour analysis and geometric patterns.
    Returns list of bubble regions for each subject.
    """
    # Ensure we have a grayscale numpy array
    if not isinstance(img, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply preprocessing for better bubble detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use adaptive thresholding for better bubble detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter bubble-like contours
    bubble_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 1000:  # Filter by area
            # Check if contour is roughly circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Reasonably circular
                    bubble_contours.append(contour)
    
    # Sort bubbles by position (top to bottom, left to right)
    bubble_centers = []
    for contour in bubble_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            bubble_centers.append((cx, cy, contour))
    
    # Sort by y-coordinate first, then x-coordinate
    bubble_centers.sort(key=lambda x: (x[1], x[0]))
    
    # Group bubbles into subjects (assuming 5 subjects)
    height, width = img.shape
    section_height = height // 5
    subject_bubbles = [[] for _ in range(5)]
    
    for cx, cy, contour in bubble_centers:
        subject_idx = min(cy // section_height, 4)  # Ensure within bounds
        
        # Extract bubble region
        x, y, w, h = cv2.boundingRect(contour)
        bubble_region = img[y:y+h, x:x+w]
        subject_bubbles[subject_idx].append(bubble_region)
    
    return subject_bubbles

def classify_bubble_advanced(bubble_region, cnn_model=None):
    """
    Advanced bubble classification using multiple techniques.
    """
    if bubble_region.size == 0:
        return False
    
    # Method 1: Pixel density analysis
    total_pixels = bubble_region.size
    filled_pixels = cv2.countNonZero(bubble_region)
    fill_ratio = filled_pixels / total_pixels
    
    # Method 2: Edge detection analysis
    edges = cv2.Canny(bubble_region, 50, 150)
    edge_pixels = cv2.countNonZero(edges)
    edge_ratio = edge_pixels / total_pixels
    
    # Method 3: CNN classification if model is available
    if cnn_model is not None:
        try:
            # Resize bubble region to model input size
            resized = cv2.resize(bubble_region, (28, 28))
            normalized = resized.astype('float32') / 255.0
            reshaped = normalized.reshape(1, 28, 28, 1)
            
            prediction = cnn_model.predict(reshaped, verbose=0)
            cnn_confidence = prediction[0][1] if prediction.shape[1] > 1 else prediction[0][0]
            
            # Use CNN prediction if confidence is high
            if cnn_confidence > 0.7:
                return cnn_confidence > 0.5
        except Exception as e:
            print(f"CNN classification failed: {e}")
    
    # Fallback to traditional methods
    # Combine fill ratio and edge analysis
    if fill_ratio > 0.4:  # High fill ratio indicates marking
        return True
    elif fill_ratio > 0.2 and edge_ratio < 0.3:  # Medium fill with low edges
        return True
    else:
        return False

def classify_bubble(region):
    """
    Legacy method - kept for compatibility
    """
    return classify_bubble_advanced(region)


def score_omr_image(
    img_or_path: Union[str, Image.Image, np.ndarray],
    answer_key_path: Optional[str] = None,
    set_name: str = "Set - A"
) -> Dict[str, Any]:
    """
    Convenience helper: detect answers from an OMR image and return the score.

    Inputs:
    - img_or_path: path to image, PIL Image, or numpy array
    - answer_key_path: path to the Excel answer key (optional). If None, uses default.
    - set_name: sheet name in the Excel answer key (e.g., "Set - A")

    Output dict contains keys from utils.scoring.evaluate:
    { sections, total, max_possible, percentage, detailed_analysis, csv_data }

    Error modes:
    - Raises FileNotFoundError if answer key file not found
    - Raises ValueError if image path is invalid
    """
    # Resolve default answer key path if not provided
    if answer_key_path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # .../OMR-Scorer
        answer_key_path = os.path.join(base_dir, "sampledata", "answer_key.xlsx.xlsx")

    if isinstance(img_or_path, str):
        if not os.path.exists(img_or_path):
            raise ValueError(f"Image file not found: {img_or_path}")
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    # Detect responses
    responses = extract_responses(img)

    # Load answer key and evaluate
    from utils import answerkey  # local import to avoid circulars
    from utils import scoring

    if not os.path.exists(answer_key_path):
        raise FileNotFoundError(f"Answer key not found at: {answer_key_path}")

    key = answerkey.load_answer_key(answer_key_path, set_name)
    result = scoring.evaluate(responses, key)

    # Attach raw responses for optional inspection
    result["responses"] = responses
    result["answer_key_path"] = answer_key_path
    result["set_name"] = set_name
    return result
