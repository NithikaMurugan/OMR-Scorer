#!/usr/bin/env python3
"""
Response Extraction Debug
Debug the extract_responses function to see where the discrepancy occurs
between detailed analysis and final results.
"""

import cv2
import numpy as np
from utils.bubbledetection import extract_responses

def debug_response_extraction():
    """Debug the extract_responses function to find the discrepancy."""
    
    # Load image
    image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    image = cv2.imread(image_path)
    
    print("=== RESPONSE EXTRACTION DEBUG ===")
    print("Comparing detailed analysis vs final extracted responses...\n")
    
    # Extract responses using the main function
    responses = extract_responses(image)
    
    print("Final extracted responses from extract_responses():")
    for subject, answers in responses.items():
        print(f"{subject}: {answers}")
    
    print("\nExpected responses for comparison:")
    expected = {
        "Python": ["A", "C", "C", "C", "C", "A"],
        "EDA": ["A", "D", "B", "A", "C", "B"],  
        "SQL": ["C", "C", "C", "B", "B", "A"],
        "POWER BI": ["B", "C", "A", "B", "C", "B"],
        "Satistics": ["A", "B", "C", "B", "C", "B"]
    }
    
    for subject in expected:
        print(f"{subject}: {expected[subject]}")
    
    print("\n=== DETAILED COMPARISON ===")
    for subject in expected:
        print(f"\n{subject}:")
        actual = responses.get(subject, [])
        expected_subj = expected[subject]
        
        for i in range(6):  # We know there are 6 questions
            actual_ans = actual[i] if i < len(actual) else "?"
            expected_ans = expected_subj[i] if i < len(expected_subj) else "?"
            
            status = "✅" if actual_ans == expected_ans else "❌"
            print(f"  Q{i+1}: Detected='{actual_ans}' | Expected='{expected_ans}' {status}")

if __name__ == "__main__":
    debug_response_extraction()