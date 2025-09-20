import pandas as pd
import os
import re

def load_answer_key(path, set_choice):
    """
    Load answer key from Excel file and parse the format properly.
    Handles formats like "1 - a", "21 - b", etc.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Answer key not found: {path}")

    df_keys = pd.read_excel(path, sheet_name=None)
    if set_choice not in df_keys:
        raise ValueError(f"{set_choice} sheet not found in Excel file.")

    df = df_keys[set_choice]
    answer_key_dict = {}
    
    for col in df.columns:
        # Clean column name (remove leading/trailing spaces)
        clean_col_name = col.strip()
        answers = []
        
        for value in df[col]:
            if pd.isna(value):
                answers.append('')
                continue
                
            # Convert to string and clean
            value_str = str(value).strip()
            
            # Extract answer letter from formats like "1 - a", "21 - b", "81. a"
            # Look for the last letter in the string
            match = re.search(r'[a-dA-D](?![a-zA-Z])', value_str)
            if match:
                answer = match.group().upper()
                answers.append(answer)
            else:
                # If no valid answer found, treat as empty
                answers.append('')
        
        answer_key_dict[clean_col_name] = answers
    
    return answer_key_dict
