#!/usr/bin/env python3

# Test scoring calculation
from utils.bubbledetection import extract_responses
from utils.answerkey import load_answer_key
from utils.scoring import evaluate
from PIL import Image
import numpy as np

# Create a dummy image for testing
dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

# Get bubble responses (will use our mock data)
print("Getting bubble responses...")
bubble_responses = extract_responses(dummy_img)

# Load answer key
print("Loading answer key...")
answer_key = load_answer_key('sampledata/answer_key.xlsx.xlsx', 'Set - A')

# Evaluate
print("Evaluating...")
results = evaluate(bubble_responses, answer_key)

print('\n=== SCORING RESULTS ===')
print(f'Total Score: {results["total"]}/{results["max_possible"]}')
print(f'Percentage: {results["percentage"]:.1f}%')
print(f'Sections: {results["sections"]}')

print('\n=== DETAILED ANALYSIS ===')
for subject, analysis in results['detailed_analysis'].items():
    print(f'{subject}: {analysis}')

print('\n=== SAMPLE COMPARISON ===')
for subject in list(bubble_responses.keys())[:2]:  # Show first 2 subjects
    print(f'\n{subject}:')
    print(f'  Expected (first 5): {answer_key[subject][:5]}')
    print(f'  Student (first 5):  {bubble_responses[subject][:5]}')
    print(f'  Score: {results["sections"][subject]}/20')