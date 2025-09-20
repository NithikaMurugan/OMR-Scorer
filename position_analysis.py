import sys
import os
from PIL import Image
import numpy as np
import cv2

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.answerkey import load_answer_key

def visualize_bubble_positions():
    """Create a detailed visualization of bubble positions vs correct answers"""
    print("=== BUBBLE POSITION ANALYSIS ===")
    print()
    
    # Load answer key for reference
    answer_key = load_answer_key("sampledata/answer_key.xlsx.xlsx", "Set - A")
    
    # Load the test image
    img = Image.open("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")
    img_array = np.array(img)
    
    # Convert to RGB for visualization
    if len(img_array.shape) == 3:
        vis_img = img_array.copy()
    else:
        vis_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Import bubble detection here to avoid circular imports
    from utils.bubbledetection import detect_actual_bubble_grid
    
    # Convert to grayscale for processing
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    # Get organized bubbles
    organized_bubbles = detect_actual_bubble_grid(gray)
    
    print("Current Detection vs Expected Answers:")
    print("=" * 80)
    
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # RGB colors
    
    for s_idx, subject in enumerate(subjects):
        if subject not in organized_bubbles:
            continue
            
        print(f"\n{subject}:")
        print("Q# | Options Positions (A,B,C,D) | Detected | Expected | Match")
        print("-" * 75)
        
        color = colors[s_idx % len(colors)]
        
        for q_idx in range(6):  # Only first 6 questions
            if q_idx not in organized_bubbles[subject]:
                continue
                
            bubbles = organized_bubbles[subject][q_idx]
            expected = answer_key[subject][q_idx] if q_idx < len(answer_key[subject]) else "?"
            
            # Get bubble positions
            positions = []
            for i, bubble in enumerate(bubbles):
                option_letter = ['A', 'B', 'C', 'D'][i]
                if bubble and bubble['center']:
                    x, y = bubble['center']
                    positions.append(f"{option_letter}:({x},{y})")
                    
                    # Draw on visualization
                    cv2.circle(vis_img, (x, y), 15, color, 2)
                    cv2.putText(vis_img, f"{subject[0]}{q_idx+1}{option_letter}", 
                              (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    positions.append(f"{option_letter}:None")
            
            # Calculate which bubble is most filled (simulating current detection)
            if bubbles and all(b is not None for b in bubbles):
                from utils.bubbledetection import calculate_bubble_fill_percentage
                fill_percentages = []
                for bubble in bubbles:
                    if bubble and 'roi' in bubble and bubble['roi'] is not None:
                        fill_pct = calculate_bubble_fill_percentage(bubble['roi'])
                        fill_percentages.append(fill_pct)
                    else:
                        fill_percentages.append(0.0)
                
                if fill_percentages:
                    max_idx = fill_percentages.index(max(fill_percentages))
                    detected = ['A', 'B', 'C', 'D'][max_idx]
                    match = "✅" if detected == expected else "❌"
                    
                    print(f"Q{q_idx+1} | {' | '.join(positions[:4])} | {detected} | {expected} | {match}")
                else:
                    print(f"Q{q_idx+1} | {' | '.join(positions[:4])} | ? | {expected} | ?")
            else:
                print(f"Q{q_idx+1} | {' | '.join(positions[:4])} | ? | {expected} | ?")
    
    # Save visualization
    vis_img_pil = Image.fromarray(vis_img)
    vis_img_pil.save("bubble_positions_debug.jpg")
    print(f"\n📊 Visualization saved to: bubble_positions_debug.jpg")
    print("\nKey insights:")
    print("- Each subject has a different color")
    print("- Labels show SubjectQuestion Option (e.g., P1A = Python Question 1 Option A)")
    print("- Compare bubble positions with the actual marked answers on the OMR sheet")

if __name__ == "__main__":
    visualize_bubble_positions()