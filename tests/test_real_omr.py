"""
Test the bubble detection function with real OMR sheets from Set A
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bubbledetection import extract_responses
from utils.preprocess import correct_orientation_and_perspective
from PIL import Image
import numpy as np

def test_real_omr():
    # Test with a real OMR sheet
    omr_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if os.path.exists(omr_path):
        print(f"Testing with real OMR sheet: {omr_path}")
        img = Image.open(omr_path)
        
        print(f"Original image size: {img.size}")
        print(f"Original image mode: {img.mode}")
        
        # Preprocess the image first
        print("Preprocessing image...")
        processed_img = correct_orientation_and_perspective(img)
        
        print(f"Processed image size: {processed_img.size}")
        
        # Extract responses
        print("Extracting responses...")
        responses = extract_responses(processed_img)
        
        print("\nExtracted responses from real OMR:")
        for subject, subject_responses in responses.items():
            print(f"\n{subject}:")
            print(f"  First 10: {subject_responses[:10]}")
            print(f"  Last 10:  {subject_responses[10:20]}")
            
            # Count response types
            answered = len([r for r in subject_responses if r and r != "MULTIPLE"])
            multiple = len([r for r in subject_responses if r == "MULTIPLE"])
            unanswered = len([r for r in subject_responses if not r])
            
            print(f"  Stats: Answered={answered}, Multiple={multiple}, Unanswered={unanswered}")
            
            # Calculate detection rate
            detection_rate = (answered / 20) * 100
            print(f"  Detection rate: {detection_rate:.1f}%")
        
    else:
        print(f"Real OMR image not found at: {omr_path}")

if __name__ == "__main__":
    test_real_omr()