import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('omr_processing.log'),
        logging.StreamHandler()
    ]
)

class OMRValidationError(Exception):
    """Custom exception for OMR validation errors"""
    pass

class OMRProcessingError(Exception):
    """Custom exception for OMR processing errors"""
    pass

def validate_uploaded_file(uploaded_file):
    """
    Validate uploaded file format, size, and basic properties.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        bool: True if valid, raises exception if invalid
        
    Raises:
        OMRValidationError: If file validation fails
    """
    if uploaded_file is None:
        raise OMRValidationError("No file uploaded")
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        raise OMRValidationError("File size too large. Maximum allowed size is 10MB")
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf']
    if uploaded_file.type not in allowed_types:
        raise OMRValidationError(f"Invalid file type: {uploaded_file.type}. Allowed types: JPG, PNG, PDF")
    
    # Check if file is corrupted by trying to open it
    try:
        if uploaded_file.type == 'application/pdf':
            # Basic PDF validation
            file_content = uploaded_file.read()
            if not file_content.startswith(b'%PDF'):
                raise OMRValidationError("Invalid PDF file format")
            uploaded_file.seek(0)  # Reset file pointer
        else:
            # Basic image validation
            image = Image.open(uploaded_file)
            image.verify()  # Verify image integrity
            uploaded_file.seek(0)  # Reset file pointer
    except Exception as e:
        raise OMRValidationError(f"File appears to be corrupted: {str(e)}")
    
    logging.info(f"File validation successful: {uploaded_file.name}")
    return True

def validate_image_quality(image_array):
    """
    Validate image quality for OMR processing.
    
    Args:
        image_array: numpy array of the image
        
    Returns:
        dict: Quality metrics and validation results
        
    Raises:
        OMRValidationError: If image quality is too poor
    """
    results = {
        'resolution': None,
        'contrast': None,
        'blur_score': None,
        'brightness': None,
        'warnings': [],
        'is_valid': True
    }
    
    try:
        height, width = image_array.shape[:2]
        results['resolution'] = (width, height)
        
        # Check minimum resolution
        if width < 800 or height < 1000:
            results['warnings'].append("Low resolution detected. Minimum recommended: 800x1000")
            if width < 400 or height < 500:
                results['is_valid'] = False
                raise OMRValidationError("Image resolution too low for reliable processing")
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Check contrast
        contrast = gray.std()
        results['contrast'] = contrast
        if contrast < 30:
            results['warnings'].append("Low contrast detected")
            if contrast < 15:
                results['is_valid'] = False
                raise OMRValidationError("Image contrast too low for reliable bubble detection")
        
        # Check blur using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        results['blur_score'] = blur_score
        if blur_score < 100:
            results['warnings'].append("Image appears blurry")
            if blur_score < 50:
                results['is_valid'] = False
                raise OMRValidationError("Image too blurry for reliable processing")
        
        # Check brightness
        brightness = np.mean(gray)
        results['brightness'] = brightness
        if brightness < 50 or brightness > 200:
            results['warnings'].append("Suboptimal lighting conditions detected")
        
        logging.info(f"Image quality validation: {results}")
        return results
        
    except OMRValidationError:
        raise
    except Exception as e:
        raise OMRValidationError(f"Error during image quality validation: {str(e)}")

def validate_answer_key(answer_key_path, set_name):
    """
    Validate answer key file and structure.
    
    Args:
        answer_key_path: Path to answer key Excel file
        set_name: Name of the answer key set
        
    Returns:
        bool: True if valid
        
    Raises:
        OMRValidationError: If answer key validation fails
    """
    try:
        if not os.path.exists(answer_key_path):
            raise OMRValidationError(f"Answer key file not found: {answer_key_path}")
        
        import pandas as pd
        
        # Try to read the Excel file
        try:
            df_keys = pd.read_excel(answer_key_path, sheet_name=None)
        except Exception as e:
            raise OMRValidationError(f"Error reading answer key Excel file: {str(e)}")
        
        # Check if the specified set exists
        if set_name not in df_keys:
            available_sets = list(df_keys.keys())
            raise OMRValidationError(f"Set '{set_name}' not found. Available sets: {available_sets}")
        
        df = df_keys[set_name]
        
        # Validate structure
        if df.empty:
            raise OMRValidationError(f"Answer key set '{set_name}' is empty")
        
        # Check for valid answer options (now only check final parsed answers)
        valid_options = {'A', 'B', 'C', 'D', ''}
        for col in df.columns:
            # Load the answers using the same parsing logic as answerkey.py
            import re
            parsed_answers = []
            for value in df[col]:
                if pd.isna(value):
                    parsed_answers.append('')
                    continue
                    
                value_str = str(value).strip()
                match = re.search(r'[a-dA-D](?![a-zA-Z])', value_str)
                if match:
                    answer = match.group().upper()
                    parsed_answers.append(answer)
                else:
                    parsed_answers.append('')
            
            invalid_answers = set(parsed_answers) - valid_options
            if invalid_answers:
                raise OMRValidationError(f"Invalid answers found in {col}: {invalid_answers}")
        
        logging.info(f"Answer key validation successful for set: {set_name}")
        return True
        
    except OMRValidationError:
        raise
    except Exception as e:
        raise OMRValidationError(f"Unexpected error during answer key validation: {str(e)}")

def handle_processing_error(error, context=""):
    """
    Handle and display processing errors to the user.
    
    Args:
        error: Exception object
        context: Additional context about where the error occurred
    """
    error_message = str(error)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log the error
    logging.error(f"Processing error in {context}: {error_message}")
    
    # Display user-friendly error messages
    if isinstance(error, OMRValidationError):
        st.error(f"❌ Validation Error: {error_message}")
        
        # Provide suggestions based on error type
        if "resolution" in error_message.lower():
            st.info("💡 **Suggestion:** Try uploading a higher resolution image (minimum 800x1000 pixels)")
        elif "contrast" in error_message.lower():
            st.info("💡 **Suggestion:** Ensure good lighting and contrast when capturing the OMR sheet")
        elif "blur" in error_message.lower():
            st.info("💡 **Suggestion:** Take a clear, focused image without camera shake")
        elif "answer key" in error_message.lower():
            st.info("💡 **Suggestion:** Check that the answer key file exists and contains the correct set")
        elif "file" in error_message.lower():
            st.info("💡 **Suggestion:** Ensure the file is not corrupted and is in a supported format")
    
    elif isinstance(error, OMRProcessingError):
        st.error(f"❌ Processing Error: {error_message}")
        st.info("💡 **Suggestion:** Try adjusting the image or contact support if the issue persists")
    
    else:
        st.error(f"❌ Unexpected Error: {error_message}")
        st.info("💡 **Suggestion:** Please try again or contact support if the problem continues")
    
    # Show troubleshooting tips
    with st.expander("🔧 Troubleshooting Tips"):
        st.markdown("""
        **Common Issues and Solutions:**
        
        1. **Low Image Quality:**
           - Ensure good lighting conditions
           - Hold the camera steady
           - Capture from directly above the sheet
        
        2. **File Format Issues:**
           - Use JPG, PNG, or PDF formats only
           - Ensure files are not corrupted
           - Keep file size under 10MB
        
        3. **Processing Errors:**
           - Check that the OMR sheet is properly aligned
           - Ensure the sheet is flat without folds or tears
           - Verify the correct answer key set is selected
        
        4. **Bubble Detection Issues:**
           - Use a dark pen/pencil for marking
           - Fill bubbles completely
           - Avoid stray marks on the sheet
        """)

def safe_image_processing(func, *args, **kwargs):
    """
    Wrapper function for safe image processing with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function or None if error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {str(e)}")
        raise OMRProcessingError(f"Error in {func.__name__}: {str(e)}")

def display_processing_warnings(warnings):
    """
    Display processing warnings to the user.
    
    Args:
        warnings: List of warning messages
    """
    if warnings:
        st.warning("⚠️ **Processing Warnings:**")
        for warning in warnings:
            st.write(f"• {warning}")
        
        st.info("💡 These warnings may affect processing accuracy. Consider retaking the image if possible.")

def create_error_report(error, file_info, processing_context):
    """
    Create a detailed error report for debugging.
    
    Args:
        error: Exception object
        file_info: Information about the processed file
        processing_context: Context about the processing step
        
    Returns:
        dict: Error report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'file_info': file_info,
        'processing_context': processing_context,
        'system_info': {
            'opencv_version': cv2.__version__,
            'numpy_version': np.__version__
        }
    }
    
    logging.error(f"Error report created: {report}")
    return report