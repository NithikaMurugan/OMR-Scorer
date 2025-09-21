import sys
import os
from PIL import Image
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bubbledetection import extract_responses
from utils.answerkey import load_answer_key

def test_actual_questions_only():
    """Test accuracy for the actual 6 questions that exist on the OMR sheet"""
    print("=== TESTING ACTUAL QUESTIONS ONLY (Q1-Q6) ===")
    print()
    
    # Load answer key
    answer_key = load_answer_key("sampledata/answer_key.xlsx.xlsx", "Set - A")
    
    # Load and process the test image
    img = Image.open("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")
    detected_responses = extract_responses(img)
    
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    
    total_correct = 0
    total_questions = 0
    
    print("Results for ACTUAL QUESTIONS (Q1-Q6):")
    print("=" * 60)
    
    for subject in subjects:
        if subject not in detected_responses:
            print(f"{subject}: No responses detected")
            continue
            
        detected = detected_responses[subject][:6]  # Only first 6 questions
        correct_answers = answer_key[subject][:6]   # Only first 6 answers
        
        correct_count = 0
        wrong_count = 0
        multiple_count = 0
        blank_count = 0
        
        print(f"\n{subject}:")
        print("Q# | Detected | Correct | Status")
        print("-" * 35)
        
        for i, (det, cor) in enumerate(zip(detected, correct_answers)):
            q_num = i + 1
            status = ""
            
            if det == "":
                status = "BLANK"
                blank_count += 1
            elif det == "MULTIPLE":
                status = "MULTIPLE"
                multiple_count += 1
            elif det.upper() == cor.upper():
                status = "CORRECT ✅"
                correct_count += 1
            else:
                status = "WRONG ❌"
                wrong_count += 1
            
            print(f"Q{q_num} | {det:8s} | {cor:7s} | {status}")
        
        subject_accuracy = (correct_count / 6 * 100) if 6 > 0 else 0
        print(f"\n{subject} Summary: {correct_count}/6 ({subject_accuracy:.1f}%)")
        
        total_correct += correct_count
        total_questions += 6
    
    # Overall accuracy for actual questions
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"ACTUAL QUESTIONS ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy >= 90:
        print("🎉 EXCELLENT! Very close to 100% on actual questions!")
    elif overall_accuracy >= 70:
        print("🔥 GREAT PROGRESS! Getting close to target!")
    elif overall_accuracy >= 50:
        print("📈 GOOD IMPROVEMENT! Continue optimizing!")
    else:
        print("⚡ MORE WORK NEEDED for 100% accuracy target")
    
    # Show remaining issues
    print(f"\nNext optimization targets:")
    print(f"- Reduce MULTIPLE responses for clearer detection")
    print(f"- Fix bubble positioning for wrong answers")
    print(f"- Fine-tune threshold calculation")
    
    return overall_accuracy

if __name__ == "__main__":
    test_actual_questions_only()