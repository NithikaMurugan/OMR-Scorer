import pandas as pd
import json

def evaluate(student_responses, answer_key_dict):
    """
    Enhanced evaluation with comprehensive scoring and validation.
    
    Args:
        student_responses: Dict of student responses by subject
        answer_key_dict: Dict of correct answers by subject
        
    Returns:
        Dict containing scores, sections, and detailed analysis
    """
    print(f"Starting evaluation with:")
    print(f"Student responses: {student_responses}")
    print(f"Answer key: {answer_key_dict}")
    
    sections = {}
    total_score = 0
    total_possible = 0
    detailed_analysis = {}
    
    # Check if we have valid inputs
    if not student_responses:
        print("WARNING: No student responses found!")
        return {
            'total_score': 0,
            'total_possible': 0,
            'percentage': 0,
            'sections': {},
            'detailed_analysis': {},
            'status': 'error',
            'message': 'No student responses detected'
        }
    
    if not answer_key_dict:
        print("WARNING: No answer key found!")
        return {
            'total_score': 0,
            'total_possible': 0,
            'percentage': 0,
            'sections': {},
            'detailed_analysis': {},
            'status': 'error',
            'message': 'No answer key provided'
        }
    
    # Process each subject
    for subject, correct_answers in answer_key_dict.items():
        print(f"Processing subject: {subject}")
        if subject not in student_responses:
            # Subject not found in responses - assign zero scores
            print(f"Subject {subject} not found in student responses")
            sections[subject] = 0
            total_possible += len(correct_answers)
            detailed_analysis[subject] = {
                'correct': 0,
                'incorrect': 0,
                'unanswered': len(correct_answers),
                'multiple_selections': 0,
                'invalid': 0
            }
            continue
        
        student_answers = student_responses[subject]
        
        # Ensure lists are same length
        max_questions = max(len(correct_answers), len(student_answers))
        
        # Pad shorter list with empty strings
        correct_answers_padded = (correct_answers + [''] * max_questions)[:max_questions]
        student_answers_padded = (student_answers + [''] * max_questions)[:max_questions]
        
        # Calculate scores for this subject
        subject_score = 0
        analysis = {
            'correct': 0,
            'incorrect': 0,
            'unanswered': 0,
            'multiple_selections': 0,
            'invalid': 0
        }
        
        for student_ans, correct_ans in zip(student_answers_padded, correct_answers_padded):
            # Apply standard OMR evaluation rules
            if not student_ans or student_ans == '':
                # OMR Rule: No marks = unanswered/skipped (0 points)
                analysis['unanswered'] += 1
            elif student_ans == 'MULTIPLE':
                # OMR Rule: Multiple marks = invalid (0 points)
                analysis['multiple_selections'] += 1
            elif student_ans == 'UNCLEAR':
                # OMR Rule: Unclear marks = invalid (0 points)
                analysis['invalid'] += 1
            elif student_ans.upper() not in ['A', 'B', 'C', 'D']:
                # OMR Rule: Invalid marks = invalid (0 points)
                analysis['invalid'] += 1
            elif student_ans.upper() == correct_ans.upper():
                # OMR Rule: Correct single mark = +1 point
                subject_score += 1
                analysis['correct'] += 1
            else:
                # OMR Rule: Wrong single mark = 0 points
                analysis['incorrect'] += 1
        
        sections[subject] = subject_score
        total_score += subject_score
        total_possible += len(correct_answers_padded)
        detailed_analysis[subject] = analysis
    
    # Create CSV data
    csv_data = create_csv_data(sections, total_score, total_possible, detailed_analysis)
    
    # Calculate percentage
    percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
    
    return {
        "sections": sections,
        "total": total_score,
        "max_possible": total_possible,
        "percentage": percentage,
        "detailed_analysis": detailed_analysis,
        "csv_data": csv_data
    }

def create_csv_data(sections, total_score, total_possible, detailed_analysis):
    """
    Create comprehensive CSV data for export.
    
    Args:
        sections: Dict of section scores
        total_score: Total score achieved
        total_possible: Maximum possible score
        detailed_analysis: Detailed breakdown by subject
        
    Returns:
        str: CSV formatted string
    """
    csv_lines = ["Subject,Score,Max_Score,Percentage,Correct,Incorrect,Unanswered,Multiple,Invalid"]
    
    for subject, score in sections.items():
        max_score = len([k for k in detailed_analysis[subject].values()])
        if max_score == 0:
            max_score = 20  # Default assumption
        
        analysis = detailed_analysis.get(subject, {})
        percentage = (score / max_score * 100) if max_score > 0 else 0
        
        csv_line = f"{subject},{score},{max_score},{percentage:.1f}," \
                  f"{analysis.get('correct', 0)}," \
                  f"{analysis.get('incorrect', 0)}," \
                  f"{analysis.get('unanswered', 0)}," \
                  f"{analysis.get('multiple_selections', 0)}," \
                  f"{analysis.get('invalid', 0)}"
        
        csv_lines.append(csv_line)
    
    # Add total row
    total_percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
    total_correct = sum(analysis.get('correct', 0) for analysis in detailed_analysis.values())
    total_incorrect = sum(analysis.get('incorrect', 0) for analysis in detailed_analysis.values())
    total_unanswered = sum(analysis.get('unanswered', 0) for analysis in detailed_analysis.values())
    total_multiple = sum(analysis.get('multiple_selections', 0) for analysis in detailed_analysis.values())
    total_invalid = sum(analysis.get('invalid', 0) for analysis in detailed_analysis.values())
    
    csv_lines.append(f"TOTAL,{total_score},{total_possible},{total_percentage:.1f},"
                    f"{total_correct},{total_incorrect},{total_unanswered},"
                    f"{total_multiple},{total_invalid}")
    
    return "\n".join(csv_lines)

def calculate_confidence_score(bubble_responses, confidence_data=None):
    """
    Calculate confidence scores for responses based on bubble detection quality.
    
    Args:
        bubble_responses: Student responses
        confidence_data: Optional confidence data from bubble detection
        
    Returns:
        Dict of confidence scores by subject and question
    """
    confidence_scores = {}
    
    for subject, responses in bubble_responses.items():
        subject_confidence = {}
        
        for i, response in enumerate(responses):
            if confidence_data and subject in confidence_data and i in confidence_data[subject]:
                # Use provided confidence data
                confidence = confidence_data[subject][i]
            else:
                # Default confidence based on response type
                if response == 'MULTIPLE':
                    confidence = 0.3  # Low confidence for multiple selections
                elif response == '':
                    confidence = 0.8  # High confidence for clearly empty bubbles
                elif response in ['A', 'B', 'C', 'D']:
                    confidence = 0.9  # High confidence for clear single selections
                else:
                    confidence = 0.1  # Very low confidence for invalid responses
            
            subject_confidence[i] = confidence
        
        confidence_scores[subject] = subject_confidence
    
    return confidence_scores

def validate_responses(student_responses, answer_key_dict):
    """
    Validate student responses against answer key structure.
    
    Args:
        student_responses: Student responses to validate
        answer_key_dict: Answer key for validation
        
    Returns:
        Dict containing validation results and suggestions
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'suggestions': []
    }
    
    # Check if subjects match
    response_subjects = set(student_responses.keys())
    answer_key_subjects = set(answer_key_dict.keys())
    
    missing_subjects = answer_key_subjects - response_subjects
    extra_subjects = response_subjects - answer_key_subjects
    
    if missing_subjects:
        validation_results['warnings'].append(f"Missing subjects in responses: {missing_subjects}")
    
    if extra_subjects:
        validation_results['warnings'].append(f"Extra subjects in responses: {extra_subjects}")
    
    # Check question counts
    for subject in response_subjects.intersection(answer_key_subjects):
        response_count = len(student_responses[subject])
        answer_key_count = len(answer_key_dict[subject])
        
        if response_count != answer_key_count:
            validation_results['warnings'].append(
                f"Question count mismatch in {subject}: "
                f"responses={response_count}, answer_key={answer_key_count}"
            )
    
    # Check for invalid responses
    valid_responses = {'A', 'B', 'C', 'D', '', 'MULTIPLE'}
    
    for subject, responses in student_responses.items():
        invalid_responses = [resp for resp in responses if resp not in valid_responses]
        if invalid_responses:
            validation_results['errors'].append(
                f"Invalid responses in {subject}: {set(invalid_responses)}"
            )
            validation_results['is_valid'] = False
    
    # Generate suggestions
    if validation_results['warnings'] or validation_results['errors']:
        validation_results['suggestions'].extend([
            "Check bubble detection parameters if many responses are invalid",
            "Verify answer key format and completeness",
            "Ensure image quality is sufficient for accurate detection"
        ])
    
    return validation_results
