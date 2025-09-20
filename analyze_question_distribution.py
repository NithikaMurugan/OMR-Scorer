#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image

def analyze_question_distribution(image_path):
    """Analyze how questions are distributed across the OMR sheet"""
    
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
    
    total_bubbles_for_questions = 0
    for i, row in enumerate(rows):
        print(f"  Row {i}: {len(row)} bubbles")
        if len(row) > 0:
            y_positions = [b['center'][1] for b in row]
            x_positions = [b['center'][0] for b in row]
            print(f"    Y range: {min(y_positions)}-{max(y_positions)}")
            print(f"    X range: {min(x_positions)}-{max(x_positions)}")
            
            # Estimate how many questions this row could contain
            if len(row) >= 20:  # Could contain question bubbles
                bubbles_per_question = 20  # 5 subjects × 4 options
                estimated_questions = len(row) // bubbles_per_question
                print(f"    Estimated questions: {estimated_questions}")
                total_bubbles_for_questions += len(row)
    
    print(f"\nTotal bubbles that could be questions: {total_bubbles_for_questions}")
    print(f"Expected bubbles for 20 questions × 5 subjects × 4 options = {20 * 5 * 4} = 400")
    
    # Calculate how questions might be distributed
    if total_bubbles_for_questions >= 400:
        print(f"We have {total_bubbles_for_questions} bubbles, enough for 20 questions")
        questions_per_bubble_count = total_bubbles_for_questions // 400
        print(f"Questions might be distributed across multiple rows")
    else:
        print(f"We have {total_bubbles_for_questions} bubbles, not enough for all 20 questions")
        print("Some questions might be in non-bubble format or missing")
    
    # Analyze specific large rows
    print(f"\nDetailed analysis of large rows:")
    for i, row in enumerate(rows):
        if len(row) >= 50:  # Focus on larger rows
            print(f"\nRow {i} with {len(row)} bubbles:")
            
            # Check if this could be multiple questions
            x_positions = [b['center'][0] for b in row]
            gaps = [x_positions[j+1] - x_positions[j] for j in range(len(x_positions)-1)]
            
            if gaps:
                avg_gap = np.mean(gaps)
                large_gaps = [j for j, gap in enumerate(gaps) if gap > avg_gap * 2]
                print(f"    Average gap: {avg_gap:.1f}")
                print(f"    Large gaps at positions: {large_gaps}")
                
                # If we find multiple large gaps, this could be multiple questions
                if len(large_gaps) >= 3:
                    estimated_questions_in_row = len(large_gaps) + 1
                    print(f"    Could contain {estimated_questions_in_row} questions")
                    
                    # Split by large gaps and analyze each segment
                    segments = []
                    start = 0
                    for gap_pos in large_gaps:
                        segments.append(row[start:gap_pos+1])
                        start = gap_pos + 1
                    segments.append(row[start:])
                    
                    for seg_idx, segment in enumerate(segments):
                        if len(segment) >= 15:  # Reasonable question size
                            print(f"      Segment {seg_idx}: {len(segment)} bubbles (likely 1 question)")

if __name__ == "__main__":
    analyze_question_distribution("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")