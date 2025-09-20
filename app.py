import streamlit as st
from PIL import Image
import pandas as pd
import os

# Import helper functions from utils
from utils import preprocess, bubble_detection, scoring

st.set_page_config(page_title="Automated OMR Evaluation System", layout="wide")
st.title("Automated OMR Evaluation System")

# --- Upload OMR sheet ---
uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpg","jpeg","png","pdf"])
if not uploaded_file:
    st.warning("Please upload an OMR sheet to continue.")
    st.stop()

# --- Select OMR Set ---
set_choice = st.selectbox("Select OMR Set", ["Set-A", "Set-B"])

# --- Load Answer Key ---
answer_key_path = os.path.join("sample_data", "answer_key.xlsx")
try:
    # Load all sheets from Excel file
    df_keys = pd.read_excel(answer_key_path, sheet_name=None)  # sheet_name=None loads all sheets
except Exception as e:
    st.error(f"Error loading answer key: {e}")
    st.stop()

# Check if selected set exists in Excel
if set_choice not in df_keys:
    st.error(f"Answer key for {set_choice} not found in Excel file.")
    st.stop()

# Extract the correct sheet for scoring
answer_key_df = df_keys[set_choice]

# Convert DataFrame to dictionary: {subject_name: [answers]}
answer_key_dict = {col: answer_key_df[col].tolist() for col in answer_key_df.columns}

# --- Convert PDF to Image if needed ---
if uploaded_file.type == "application/pdf":
    from pdf2image import convert_from_bytes
    images = convert_from_bytes(uploaded_file.read())
    omr_image = images[0]  # use first page
else:
    omr_image = Image.open(uploaded_file)

st.image(omr_image, caption="Uploaded OMR Sheet", use_column_width=True)

# --- Preprocess Image ---
processed_img = preprocess.correct_orientation_and_threshold(omr_image)

# --- Bubble Detection ---
bubble_responses = bubble_detection.extract_responses(processed_img)

# --- Evaluate Scores ---
student_scores = scoring.evaluate(bubble_responses, answer_key_dict)

# --- Display Section-wise Scores ---
st.subheader("Section-wise Scores:")
for section, score in student_scores['sections'].items():
    st.write(f"{section}: {score}/20")

st.subheader(f"Total Score: {student_scores['total']}/100")

# --- Download CSV ---
st.download_button(
    "Download Results as CSV",
    student_scores['csv_data'],
    file_name="results.csv",
    mime="text/csv"
)
