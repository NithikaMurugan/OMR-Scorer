import pandas as pd
import os

# Check answer key file structure
answer_key_path = "sampledata/answer_key.xlsx.xlsx"

if os.path.exists(answer_key_path):
    try:
        # Read all sheet names
        xlsx_file = pd.ExcelFile(answer_key_path)
        sheet_names = xlsx_file.sheet_names
        print(f"Available sheets in {answer_key_path}:")
        for i, sheet in enumerate(sheet_names):
            print(f"  {i+1}. {sheet}")
        
        # Try to read the first sheet
        if sheet_names:
            print(f"\nReading first sheet: '{sheet_names[0]}'")
            df = pd.read_excel(answer_key_path, sheet_name=sheet_names[0])
            print(f"Columns: {list(df.columns)}")
            print(f"Shape: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File not found: {answer_key_path}")