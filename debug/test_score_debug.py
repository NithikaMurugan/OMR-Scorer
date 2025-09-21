"""
Test to verify score calculation with real bubble detection data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scoring import evaluate
from utils.answerkey import load_answer_key

def test_score_calculation_debug():
    """Test score calculation with debug output"""
    
    print("=== DEBUGGING SCORE CALCULATION ===")
    
    # Load answer key
    answer_key_path = "sampledata/answer_key.xlsx.xlsx"
    set_choice = "Set - A"
    
    if not os.path.exists(answer_key_path):
        print(f"ERROR: Answer key not found at {answer_key_path}")
        return
    
    answer_key = load_answer_key(answer_key_path, set_choice)
    print(f"Answer key loaded for {set_choice}")
    print(f"Subjects: {list(answer_key.keys())}")
    print(f"Questions per subject: {[len(answers) for answers in answer_key.values()]}")
    
    # Simulate realistic bubble detection with lots of MULTIPLE responses
    # This matches what we're seeing from the real bubble detection
    bubble_responses = {
        "Python": ['MULTIPLE', 'A', 'MULTIPLE', 'C', 'MULTIPLE', 'A', 'MULTIPLE', 'C', 'B', 'MULTIPLE', 
                   'A', 'MULTIPLE', 'A', 'MULTIPLE', 'B', 'MULTIPLE', 'C', 'MULTIPLE', 'A', 'MULTIPLE'],
        "EDA": ['A', 'MULTIPLE', 'B', '', 'MULTIPLE', 'B', 'MULTIPLE', '', 'C', 'C', 
                'MULTIPLE', 'A', '', 'MULTIPLE', 'A', 'B', 'MULTIPLE', 'B', 'MULTIPLE', 'B'],
        "SQL": ['', 'C', 'MULTIPLE', 'B', 'MULTIPLE', 'A', 'MULTIPLE', 'B', '', 'A', 
                'C', 'MULTIPLE', 'C', 'MULTIPLE', 'A', 'B', 'MULTIPLE', 'A', 'MULTIPLE', 'B'],
        "POWER BI": ['MULTIPLE', 'C', '', 'B', 'MULTIPLE', 'B', 'MULTIPLE', 'C', 'C', 'MULTIPLE', 
                     'B', 'B', 'MULTIPLE', 'B', 'MULTIPLE', 'B', 'B', '', 'B', 'MULTIPLE'],
        "Satistics": ['MULTIPLE', 'MULTIPLE', 'MULTIPLE', 'MULTIPLE', 'MULTIPLE', 'B', 'B', 'B', 'A', 'B', 
                      'C', 'B', 'MULTIPLE', 'B', 'MULTIPLE', 'B', 'MULTIPLE', 'A', 'MULTIPLE', 'C']
    }
    
    print(f"\nBubble responses loaded:")
    for subject, responses in bubble_responses.items():
        correct_count = len([r for r in responses if r and r != 'MULTIPLE'])
        multiple_count = len([r for r in responses if r == 'MULTIPLE'])
        empty_count = len([r for r in responses if not r])
        print(f"  {subject}: {correct_count} answered, {multiple_count} multiple, {empty_count} empty")
    
    # Evaluate scores
    print(f"\n=== CALLING SCORING.EVALUATE ===")
    result = evaluate(bubble_responses, answer_key)
    
    print(f"\n=== SCORE CALCULATION RESULTS ===")
    print(f"Full result object: {result}")
    print(f"")
    print(f"Key metrics:")
    print(f"  Total score: {result.get('total', 'MISSING')}")
    print(f"  Max possible: {result.get('max_possible', 'MISSING')}")
    print(f"  Percentage: {result.get('percentage', 'MISSING')}")
    print(f"  Sections: {result.get('sections', 'MISSING')}")
    
    if 'detailed_analysis' in result:
        print(f"\nDetailed analysis:")
        for subject, analysis in result['detailed_analysis'].items():
            print(f"  {subject}:")
            print(f"    Correct: {analysis.get('correct', 0)}")
            print(f"    Incorrect: {analysis.get('incorrect', 0)}")
            print(f"    Unanswered: {analysis.get('unanswered', 0)}")
            print(f"    Multiple: {analysis.get('multiple_selections', 0)}")
            print(f"    Invalid: {analysis.get('invalid', 0)}")
    
    # Verify calculation manually
    print(f"\n=== MANUAL VERIFICATION ===")
    expected_total = sum(result.get('sections', {}).values())
    actual_total = result.get('total', 0)
    print(f"Expected total (sum of sections): {expected_total}")
    print(f"Actual total from result: {actual_total}")
    print(f"Match: {expected_total == actual_total}")
    
    if result.get('max_possible', 0) > 0:
        expected_percentage = (actual_total / result['max_possible']) * 100
        actual_percentage = result.get('percentage', 0)
        print(f"Expected percentage: {expected_percentage:.1f}%")
        print(f"Actual percentage: {actual_percentage:.1f}%")
        print(f"Match: {abs(expected_percentage - actual_percentage) < 0.1}")

if __name__ == "__main__":
    test_score_calculation_debug()