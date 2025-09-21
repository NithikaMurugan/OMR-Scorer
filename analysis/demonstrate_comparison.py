"""
Demonstrate how OMR answer comparison works with Excel answer key
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.answerkey import load_answer_key
from utils.scoring import evaluate
import pandas as pd

def demonstrate_answer_comparison():
    """Show step-by-step how answers are compared"""
    
    print("=" * 60)
    print("OMR ANSWER COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Load Excel answer key
    print("\n1. LOADING ANSWER KEY FROM EXCEL FILE")
    print("-" * 40)
    
    answer_key_path = "sampledata/answer_key.xlsx.xlsx"
    set_choice = "Set - A"
    
    if not os.path.exists(answer_key_path):
        print(f"ERROR: Answer key file not found at {answer_key_path}")
        return
    
    # Show Excel file contents
    print(f"Reading Excel file: {answer_key_path}")
    try:
        df = pd.read_excel(answer_key_path, sheet_name=set_choice)
        print(f"Excel sheet '{set_choice}' loaded successfully")
        print(f"Columns found: {list(df.columns)}")
        print(f"Number of questions: {len(df)}")
        print("\nFirst 5 rows of Excel data:")
        print(df.head())
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return
    
    # Load structured answer key
    answer_key = load_answer_key(answer_key_path, set_choice)
    print(f"\nStructured answer key loaded:")
    for subject, answers in answer_key.items():
        print(f"  {subject}: {answers[:5]}... (first 5 answers)")
    
    # Step 2: Simulate OMR detected responses
    print("\n2. SIMULATED OMR DETECTED RESPONSES")
    print("-" * 40)
    
    # Create realistic OMR responses with some correct, some wrong, some multiple
    omr_responses = {
        "Python": ['A', 'C', 'B', 'C', 'C', 'A', 'MULTIPLE', 'C', 'B', 'C', 
                   'A', 'A', 'D', '', 'B', 'A', 'C', 'D', 'MULTIPLE', 'B'],
        "EDA": ['A', 'D', 'A', 'A', 'C', 'B', 'A', 'MULTIPLE', 'D', 'C', 
                'C', 'A', 'B', '', 'A', 'B', 'D', 'B', 'A', 'B'],
        "SQL": ['C', 'C', 'D', 'B', 'B', 'A', 'C', 'B', 'D', 'A', 
                'MULTIPLE', 'B', 'C', 'C', '', 'B', 'B', 'A', 'A', 'B'],
        "POWER BI": ['B', 'C', 'A', 'B', 'C', 'B', 'B', 'C', 'C', 'B', 
                     'B', 'B', 'D', 'B', 'A', '', 'B', 'B', 'B', 'B'],
        "Satistics": ['A', 'B', 'C', 'B', 'C', 'B', 'B', 'B', 'A', 'B', 
                      'C', 'B', 'C', 'B', 'B', 'B', 'MULTIPLE', 'A', 'B', 'C']
    }
    
    print("OMR detected responses (first 10 questions):")
    for subject, responses in omr_responses.items():
        print(f"  {subject}: {responses[:10]}")
    
    # Step 3: Detailed comparison for one subject
    print("\n3. DETAILED ANSWER COMPARISON - PYTHON SUBJECT")
    print("-" * 50)
    
    python_correct = answer_key["Python"]
    python_student = omr_responses["Python"]
    
    print("Question-by-question comparison:")
    print("Q# | Correct | Student | Result")
    print("---|---------|---------|--------")
    
    correct_count = 0
    for i in range(len(python_correct)):
        correct_ans = python_correct[i]
        student_ans = python_student[i]
        
        if not student_ans or student_ans == '':
            result = "UNANSWERED"
        elif student_ans == 'MULTIPLE':
            result = "MULTIPLE_MARKS"
        elif student_ans.upper() == correct_ans.upper():
            result = "✓ CORRECT"
            correct_count += 1
        else:
            result = "✗ WRONG"
        
        print(f"{i+1:2d} |    {correct_ans}    |    {student_ans}    | {result}")
    
    print(f"\nPython Summary: {correct_count}/20 correct answers")
    
    # Step 4: Full evaluation
    print("\n4. COMPLETE EVALUATION RESULTS")
    print("-" * 40)
    
    evaluation_result = evaluate(omr_responses, answer_key)
    
    print("Overall Results:")
    print(f"  Total Score: {evaluation_result['total']}")
    print(f"  Maximum Possible: {evaluation_result['max_possible']}")
    print(f"  Percentage: {evaluation_result['percentage']:.1f}%")
    
    print("\nSubject-wise Results:")
    for subject, score in evaluation_result['sections'].items():
        max_score = len(answer_key[subject])
        percentage = (score / max_score) * 100
        print(f"  {subject}: {score}/{max_score} ({percentage:.1f}%)")
    
    print("\nDetailed Analysis:")
    for subject, analysis in evaluation_result['detailed_analysis'].items():
        print(f"  {subject}:")
        print(f"    Correct answers: {analysis['correct']}")
        print(f"    Wrong answers: {analysis['incorrect']}")
        print(f"    Unanswered: {analysis['unanswered']}")
        print(f"    Multiple marks: {analysis['multiple_selections']}")
    
    # Step 5: Show how scoring logic works
    print("\n5. SCORING LOGIC EXPLANATION")
    print("-" * 40)
    print("How the system compares answers:")
    print("1. Load correct answers from Excel file")
    print("2. For each OMR detected response:")
    print("   - If response matches correct answer → +1 point")
    print("   - If response is wrong → 0 points")
    print("   - If response is empty ('') → 0 points (unanswered)")
    print("   - If response is 'MULTIPLE' → 0 points (invalid)")
    print("3. Calculate percentage: (correct_answers / total_questions) × 100")
    print("4. Provide detailed breakdown by subject")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_answer_comparison()