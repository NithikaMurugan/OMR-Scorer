"""
Test scoring system with real bubble detection results
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scoring import evaluate
from utils.answerkey import load_answer_key

def test_scoring_with_real_data():
    """Test scoring with realistic bubble detection results"""
    
    # Load the actual answer key
    answer_key_path = "sampledata/answer_key.xlsx.xlsx"
    answer_key = load_answer_key(answer_key_path, "Set - A")
    
    print("=== TESTING SCORING SYSTEM ===")
    print(f"Answer key loaded: {list(answer_key.keys())}")
    
    # Test Case 1: Perfect responses
    print("\n--- Test Case 1: Perfect Responses ---")
    perfect_responses = {
        "Python": answer_key["Python"],
        "EDA": answer_key["EDA"], 
        "SQL": answer_key["SQL"],
        "POWER BI": answer_key["POWER BI"],
        "Satistics": answer_key["Satistics"]
    }
    
    result1 = evaluate(perfect_responses, answer_key)
    print(f"Perfect score result:")
    print(f"  Total: {result1['total']}/{result1.get('max_possible', 100)}")
    print(f"  Percentage: {result1.get('percentage', 0):.1f}%")
    print(f"  Sections: {result1['sections']}")
    
    # Test Case 2: Realistic bubble detection with MULTIPLE and empty responses
    print("\n--- Test Case 2: Realistic Bubble Detection ---")
    realistic_responses = {
        "Python": ['A', '', 'MULTIPLE', 'C', 'C'] + answer_key["Python"][5:15] + ['B', 'MULTIPLE', '', 'A', 'B'],
        "EDA": ['', 'D', 'B', 'MULTIPLE', 'C'] + answer_key["EDA"][5:12] + ['', 'MULTIPLE', 'A', 'B', 'D', 'B', 'A', 'B'],
        "SQL": answer_key["SQL"][:10] + ['MULTIPLE', '', 'C', 'C', 'A'] + ['B'] * 5,
        "POWER BI": ['B', 'MULTIPLE', '', 'B'] + answer_key["POWER BI"][4:16] + ['', 'B', 'B', 'MULTIPLE'],
        "Satistics": ['MULTIPLE'] * 5 + answer_key["Satistics"][5:15] + ['', 'B', 'C', 'A', 'B']
    }
    
    result2 = evaluate(realistic_responses, answer_key)
    print(f"Realistic detection result:")
    print(f"  Total: {result2['total']}/{result2.get('max_possible', 100)}")
    print(f"  Percentage: {result2.get('percentage', 0):.1f}%")
    print(f"  Sections: {result2['sections']}")
    
    if 'detailed_analysis' in result2:
        print(f"  Detailed Analysis:")
        for subject, analysis in result2['detailed_analysis'].items():
            print(f"    {subject}: Correct={analysis.get('correct', 0)}, "
                  f"Multiple={analysis.get('multiple_selections', 0)}, "
                  f"Unanswered={analysis.get('unanswered', 0)}")
    
    # Test Case 3: All wrong answers
    print("\n--- Test Case 3: All Wrong Answers ---")
    wrong_responses = {}
    for subject, correct_answers in answer_key.items():
        wrong_responses[subject] = []
        for correct in correct_answers:
            # Choose a wrong answer
            options = ['A', 'B', 'C', 'D']
            wrong_options = [opt for opt in options if opt != correct]
            wrong_responses[subject].append(wrong_options[0] if wrong_options else 'A')
    
    result3 = evaluate(wrong_responses, answer_key)
    print(f"All wrong result:")
    print(f"  Total: {result3['total']}/{result3.get('max_possible', 100)}")
    print(f"  Percentage: {result3.get('percentage', 0):.1f}%")
    print(f"  Sections: {result3['sections']}")
    
    # Test Case 4: Empty responses (no answers detected)
    print("\n--- Test Case 4: No Answers Detected ---")
    empty_responses = {
        "Python": [''] * 20,
        "EDA": [''] * 20,
        "SQL": [''] * 20, 
        "POWER BI": [''] * 20,
        "Satistics": [''] * 20
    }
    
    result4 = evaluate(empty_responses, answer_key)
    print(f"Empty responses result:")
    print(f"  Total: {result4['total']}/{result4.get('max_possible', 100)}")
    print(f"  Percentage: {result4.get('percentage', 0):.1f}%")
    print(f"  Sections: {result4['sections']}")
    
    print("\n=== SCORING TESTS COMPLETED ===")

if __name__ == "__main__":
    test_scoring_with_real_data()