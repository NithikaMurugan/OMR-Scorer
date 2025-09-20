"""
Create a test OMR sheet image for testing bubble detection
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_omr_sheet(width=2100, height=1500):
    """
    Create a simple test OMR sheet with known filled bubbles
    """
    # Create white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Define subjects
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    options = ["A", "B", "C", "D"]
    
    # Calculate dimensions
    col_width = width // 5
    row_height = height // 25
    bubble_radius = 15
    
    # Draw title
    try:
        # Try to use a font
        font = ImageFont.truetype("arial.ttf", 20)
        title_font = ImageFont.truetype("arial.ttf", 30)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    draw.text((width//2 - 100, 20), "OMR Test Sheet - Set A", fill='black', font=title_font)
    
    # Test pattern: Known answers for verification
    test_answers = {
        "Python": ['A', 'C', 'C', 'C', 'C', 'A', 'C', 'C', 'B', 'C', 'A', 'A', 'D', 'A', 'B', 'A', 'C', 'D', 'A', 'B'],
        "EDA": ['A', 'D', 'B', 'A', 'C', 'B', 'A', 'B', 'D', 'C', 'C', 'A', 'B', 'C', 'A', 'B', 'D', 'B', 'A', 'B'],
        "SQL": ['C', 'C', 'C', 'B', 'B', 'A', 'C', 'B', 'D', 'A', 'C', 'B', 'C', 'C', 'A', 'B', 'B', 'A', 'A', 'B'],
        "POWER BI": ['B', 'C', 'A', 'B', 'C', 'B', 'B', 'C', 'C', 'B', 'B', 'B', 'D', 'B', 'A', 'B', 'B', 'B', 'B', 'B'],
        "Satistics": ['A', 'B', 'C', 'B', 'C', 'B', 'B', 'B', 'A', 'B', 'C', 'B', 'C', 'B', 'B', 'B', 'C', 'A', 'B', 'C']
    }
    
    # Draw subjects and bubbles
    for subject_idx, subject in enumerate(subjects):
        subject_x = subject_idx * col_width
        
        # Draw subject header
        header_y = 60
        draw.text((subject_x + 20, header_y), subject, fill='black', font=font)
        
        # Draw 20 questions for each subject
        for question_num in range(20):
            question_y = int(row_height * (question_num + 3))
            
            # Draw question number
            draw.text((subject_x + 10, question_y), f"{question_num + 1}.", fill='black', font=font)
            
            # Draw option bubbles
            for option_idx, option in enumerate(options):
                bubble_x = subject_x + 50 + (option_idx * 40)
                bubble_y = question_y + 10
                
                # Draw option label
                draw.text((bubble_x - 5, bubble_y - 30), option, fill='black', font=font)
                
                # Draw bubble circle
                bubble_bbox = [
                    bubble_x - bubble_radius, bubble_y - bubble_radius,
                    bubble_x + bubble_radius, bubble_y + bubble_radius
                ]
                
                # Check if this bubble should be filled based on test answers
                correct_answer = test_answers[subject][question_num]
                if option == correct_answer:
                    # Fill the bubble (simulate student marking)
                    draw.ellipse(bubble_bbox, fill='black', outline='black')
                else:
                    # Empty bubble
                    draw.ellipse(bubble_bbox, fill='white', outline='black', width=2)
    
    return img

def create_test_omr_with_errors():
    """
    Create a test OMR with some intentional errors (multiple marks, missed answers)
    """
    img = create_test_omr_sheet()
    draw = ImageDraw.Draw(img)
    
    # Add some intentional errors for testing
    # Multiple marks on Python Q1 (A and B both filled)
    subject_x = 0  # Python is first column
    question_y = int((1500 // 25) * 3)  # First question
    bubble_radius = 15
    
    # Fill both A and B for Python Q1
    for option_idx in [0, 1]:  # A and B
        bubble_x = 50 + (option_idx * 40)
        bubble_y = question_y + 10
        bubble_bbox = [
            bubble_x - bubble_radius, bubble_y - bubble_radius,
            bubble_x + bubble_radius, bubble_y + bubble_radius
        ]
        draw.ellipse(bubble_bbox, fill='black', outline='black')
    
    return img

if __name__ == "__main__":
    # Create test images
    test_omr = create_test_omr_sheet()
    test_omr.save("sampledata/test_omr_perfect.png")
    print("Created test_omr_perfect.png")
    
    test_omr_errors = create_test_omr_with_errors()
    test_omr_errors.save("sampledata/test_omr_with_errors.png")
    print("Created test_omr_with_errors.png")