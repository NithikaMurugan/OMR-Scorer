import sys
import os
from PIL import Image
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bubbledetection import extract_responses
from utils.answerkey import load_answer_key

def analyze_specific_errors():
    """Analyze specific detection errors to optimize further"""
    print("=== DETAILED ERROR ANALYSIS FOR 100% ACCURACY ===")
    print()
    
    # Load answer key
    answer_key = load_answer_key("sampledata/answer_key.xlsx.xlsx", "Set - A")
    
    # Load and process the test image
    img = Image.open("sampledata/omr_sheets/set_A/Set A/Img1.jpeg")
    detected_responses = extract_responses(img)
    
    # Analyze errors by subject
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Satistics"]
    
    total_correct = 0
    total_wrong = 0
    total_multiple = 0
    total_blank = 0
    
    for subject in subjects:
        print(f"=== {subject.upper()} ANALYSIS ===")
        if subject not in detected_responses:
            print(f"Subject {subject} not detected!")
            continue
            
        detected = detected_responses[subject]
        correct_answers = answer_key[subject]
        
        correct_count = 0
        wrong_count = 0
        multiple_count = 0
        blank_count = 0
        
        print("Question | Detected | Correct | Status | Issue")
        print("-" * 50)
        
        for i, (det, cor) in enumerate(zip(detected, correct_answers)):
            q_num = i + 1
            status = ""
            issue = ""
            
            if det == "":
                status = "BLANK"
                blank_count += 1
                issue = "No bubble detected"
            elif det == "MULTIPLE":
                status = "MULTIPLE"
                multiple_count += 1
                issue = "Multiple bubbles detected"
            elif det.upper() == cor.upper():
                status = "CORRECT ✅"
                correct_count += 1
            else:
                status = "WRONG ❌"
                wrong_count += 1
                issue = f"Detected {det} instead of {cor}"
            
            print(f"Q{q_num:2d}      | {det:8s} | {cor:7s} | {status:10s} | {issue}")
        
        accuracy = (correct_count / len(detected) * 100) if len(detected) > 0 else 0
        print(f"\n{subject} Summary: {correct_count}/20 ({accuracy:.1f}%)")
        print(f"  Correct: {correct_count}, Wrong: {wrong_count}, Multiple: {multiple_count}, Blank: {blank_count}")
        print()
        
        total_correct += correct_count
        total_wrong += wrong_count
        total_multiple += multiple_count
        total_blank += blank_count
    
    # Overall analysis
    total_questions = 100
    overall_accuracy = (total_correct / total_questions * 100)
    
    print("=== OVERALL ANALYSIS ===")
    print(f"Total Accuracy: {total_correct}/100 ({overall_accuracy:.1f}%)")
    print(f"Breakdown:")
    print(f"  ✅ Correct: {total_correct} ({total_correct}%)")
    print(f"  ❌ Wrong: {total_wrong} ({total_wrong}%)")
    print(f"  🔄 Multiple: {total_multiple} ({total_multiple}%)")
    print(f"  ⭕ Blank: {total_blank} ({total_blank}%)")
    print()
    
    # Identify primary issues
    print("=== KEY OPTIMIZATION TARGETS ===")
    if total_multiple > 15:
        print(f"1. 🔄 MULTIPLE responses ({total_multiple}%): Threshold disambiguation needs improvement")
    if total_blank > 15:
        print(f"2. ⭕ BLANK responses ({total_blank}%): Bubble detection sensitivity too low")
    if total_wrong > 15:
        print(f"3. ❌ WRONG responses ({total_wrong}%): Grid organization or bubble detection errors")
    
    # Progress tracking
    if overall_accuracy >= 20:
        print(f"\n🎯 PROGRESS: {overall_accuracy:.1f}% - GOOD IMPROVEMENT! Getting closer to 100%")
    elif overall_accuracy >= 15:
        print(f"\n📈 PROGRESS: {overall_accuracy:.1f}% - SOLID IMPROVEMENT! Continue optimizing")
    else:
        print(f"\n⚡ PROGRESS: {overall_accuracy:.1f}% - IMPROVEMENT NEEDED for 100% target")
    
    return overall_accuracy

if __name__ == "__main__":
    analyze_specific_errors()