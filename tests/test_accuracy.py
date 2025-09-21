import os
import sys
from PIL import Image
import numpy as np

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bubbledetection import extract_responses
from utils.scoring import evaluate
from utils.answerkey import load_answer_key

def test_omr_accuracy():
    """Test the complete OMR system accuracy"""
    print("Testing Complete OMR System Accuracy")
    print("=" * 50)
    
    # Load answer key
    answer_key_path = "sampledata/answer_key.xlsx.xlsx"
    try:
        answer_key = load_answer_key(answer_key_path, "Set - A")  # Correct sheet name
        print(f"Loaded answer key with {len(answer_key)} subjects")
        for subject, answers in answer_key.items():
            print(f"  {subject}: {len(answers)} answers")
    except Exception as e:
        print(f"Error loading answer key: {e}")
        return
    
    # Test image path
    test_image_path = "sampledata/omr_sheets/set_A/Set A/Img1.jpeg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    try:
        # Load and process the test image
        print(f"\nProcessing test image: {test_image_path}")
        img = Image.open(test_image_path)
        
        # Extract responses using bubble detection
        print("\n=== Bubble Detection Results ===")
        detected_responses = extract_responses(img)
        
        if not detected_responses:
            print("No responses detected!")
            return
        
        print(f"Detected responses for {len(detected_responses)} subjects:")
        for subject, responses in detected_responses.items():
            print(f"  {subject}: {len(responses)} responses")
            print(f"    Sample: {responses[:5]}...")  # Show first 5 responses
        
        # Calculate scores using the evaluate function
        print("\n=== Scoring Results ===")
        evaluation_result = evaluate(detected_responses, answer_key)
        
        if 'sections' in evaluation_result:
            total_correct = evaluation_result.get('total', 0)
            total_questions = evaluation_result.get('max_possible', 0)
            overall_accuracy = evaluation_result.get('percentage', 0)
            
            # Show per-subject results
            for subject, score in evaluation_result['sections'].items():
                subject_total = len(answer_key.get(subject, []))
                subject_accuracy = (score / subject_total * 100) if subject_total > 0 else 0
                print(f"{subject}: {score}/{subject_total} ({subject_accuracy:.1f}%)")
            
            # Overall accuracy
            print(f"\nOVERALL ACCURACY: {total_correct}/{total_questions} ({overall_accuracy:.1f}%)")
            
            if overall_accuracy >= 100:
                print("🎉 PERFECT ACCURACY ACHIEVED! 🎉")
            elif overall_accuracy >= 90:
                print("🔥 EXCELLENT ACCURACY! Very close to 100%!")
            elif overall_accuracy >= 75:
                print("✅ GOOD ACCURACY! Significant improvement!")
            elif overall_accuracy >= 50:
                print("📈 MODERATE ACCURACY. Continue optimizing...")
            else:
                print("⚠️ LOW ACCURACY. Major optimization needed...")
            
            # Show detailed analysis if available
            if 'detailed_analysis' in evaluation_result:
                print("\n=== Detailed Analysis ===")
                for subject, analysis in evaluation_result['detailed_analysis'].items():
                    correct = analysis.get('correct', 0)
                    incorrect = analysis.get('incorrect', 0)
                    multiple = analysis.get('multiple_selections', 0)
                    unanswered = analysis.get('unanswered', 0)
                    print(f"{subject}: Correct={correct}, Wrong={incorrect}, Multiple={multiple}, Blank={unanswered}")
            
            return overall_accuracy
        else:
            print("Error in evaluation results")
            return 0
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_omr_accuracy()