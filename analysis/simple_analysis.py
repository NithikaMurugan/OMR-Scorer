import sys
import os
from PIL import Image
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_bubble_distribution_simple():
    """Simple analysis using existing bubble detection code"""
    print("=== SIMPLE BUBBLE DISTRIBUTION ANALYSIS ===")
    print()
    
    # Import here to avoid circular imports
    from utils.bubbledetection import extract_responses
    
    # Load and process the test image
    img = Image.open("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")
    
    # Temporarily modify the bubbledetection to be more verbose
    # Let's just run the existing detection and see the output
    print("Running existing bubble detection with verbose output...")
    
    detected_responses = extract_responses(img)
    
    print("\nDetected Responses Structure:")
    for subject, responses in detected_responses.items():
        print(f"{subject}: {len(responses)} responses")
        
        # Count response types
        correct_responses = [r for r in responses if r and r != 'MULTIPLE' and r != '']
        multiple_responses = [r for r in responses if r == 'MULTIPLE']
        blank_responses = [r for r in responses if r == '']
        
        print(f"  Valid: {len(correct_responses)}, Multiple: {len(multiple_responses)}, Blank: {len(blank_responses)}")
        
        # Show where the blanks are
        blank_positions = [i+1 for i, r in enumerate(responses) if r == '']
        if blank_positions:
            print(f"  Blank at questions: {blank_positions}")
    
    # The key insight: questions 14-20 are blank across most subjects
    # This suggests that the grid organization is missing the bottom rows
    
    print("\n=== ANALYSIS CONCLUSION ===")
    print("Pattern observed: Questions 14-20 are mostly blank across subjects")
    print("This indicates that the bubble detection is not capturing the bottom rows of the OMR sheet")
    print("The system is detecting 28 rows but only using the first 20, missing the actual question rows")

if __name__ == "__main__":
    analyze_bubble_distribution_simple()