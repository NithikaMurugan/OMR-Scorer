"""
Test the bubble detection function with the generated test OMR sheet
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bubbledetection import extract_responses
from PIL import Image
import numpy as np

def test_bubble_detection():
    # Load the test OMR image
    if os.path.exists("sampledata/test_omr_perfect.png"):
        print("Testing with perfect test OMR sheet...")
        img = Image.open("sampledata/test_omr_perfect.png")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        # Extract responses
        responses = extract_responses(img)
        
        print("Extracted responses:")
        for subject, subject_responses in responses.items():
            print(f"{subject}: {subject_responses[:5]}... (showing first 5)")
            
            # Count response types
            answered = len([r for r in subject_responses if r and r != "MULTIPLE"])
            multiple = len([r for r in subject_responses if r == "MULTIPLE"])
            unanswered = len([r for r in subject_responses if not r])
            
            print(f"  Answered: {answered}, Multiple: {multiple}, Unanswered: {unanswered}")
        
    else:
        print("Test OMR image not found. Please run utils/create_test_omr.py first")

if __name__ == "__main__":
    test_bubble_detection()