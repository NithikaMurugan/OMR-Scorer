import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Import utility modules
from utils import preprocess, bubbledetection, scoring, answerkey
from utils.database import OMRDatabase
from utils.validation import (validate_uploaded_file, validate_image_quality, 
                            validate_answer_key, handle_processing_error,
                            safe_image_processing, display_processing_warnings,
                            OMRValidationError, OMRProcessingError)
from utils.export import OMRExporter

# Page configuration
st.set_page_config(
    page_title="Automated OMR Evaluation System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database and exporter
@st.cache_resource
def init_database():
    return OMRDatabase()

@st.cache_resource  
def init_exporter():
    return OMRExporter(init_database())

@st.cache_data
def get_answer_key_path():
    """Get the absolute path to the answer key file."""
    return os.path.join(os.path.dirname(__file__), "sampledata", "answer_key.xlsx.xlsx")

db = init_database()
exporter = init_exporter()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", [
    "OMR Processing", 
    "Batch Processing", 
    "Dashboard", 
    "Student Analysis", 
    "Exam Management"
])

if page == "OMR Processing":
    st.title("🎯 Automated OMR Evaluation System")
    st.markdown("Upload OMR sheets for automated evaluation and scoring")
    
    # Add information about real bubble detection
    st.info("🔍 **Real Bubble Detection Active**: The system now uses actual computer vision to detect filled bubbles in your OMR sheets for Set A!")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload OMR Sheet")
        uploaded_file = st.file_uploader(
            "Choose an OMR sheet", 
            type=["jpg", "jpeg", "png", "pdf"],
            help="Supported formats: JPG, JPEG, PNG, PDF"
        )
    
    with col2:
        st.subheader("⚙️ Configuration")
        
        # Exam selection
        exam_name = st.text_input("Exam Name", value="Sample Exam")
        exam_date = st.date_input("Exam Date", value=datetime.now().date())
        subject = st.text_input("Subject", value="General")
        
        # Set selection
        set_choice = st.selectbox("Select OMR Set", ["Set - A", "Set - B"])
        
        # Answer key upload/selection
        st.subheader("📄 Answer Key")
        answer_key_option = st.radio(
            "Choose answer key source:",
            ["Use default answer key", "Upload custom answer key"],
            index=0
        )
        
        answer_key_path = None
        custom_answer_key = None
        
        if answer_key_option == "Use default answer key":
            answer_key_path = get_answer_key_path()
            if os.path.exists(answer_key_path):
                st.success(f"✅ Default answer key found: {os.path.basename(answer_key_path)}")
            else:
                st.error(f"❌ Default answer key not found at: {answer_key_path}")
        else:
            custom_answer_key = st.file_uploader(
                "Upload Answer Key Excel File",
                type=["xlsx", "xls"],
                help="Upload an Excel file with answer keys for each set"
            )
            if custom_answer_key:
                st.success(f"✅ Custom answer key uploaded: {custom_answer_key.name}")
        
        # Debug mode toggle
        debug_mode = st.checkbox("Debug Mode", help="Show detailed bubble detection information")
        
        # Student information
        student_id = st.text_input("Student ID (Optional)")
        student_name = st.text_input("Student Name (Optional)")
    
    if uploaded_file is not None:
        try:
            # Validate uploaded file
            validate_uploaded_file(uploaded_file)
            
            start_time = time.time()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load and validate image
            status_text.text("Loading and validating image...")
            progress_bar.progress(10)
            
            if uploaded_file.type == "application/pdf":
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(uploaded_file.read())
                omr_image = images[0]
            else:
                omr_image = Image.open(uploaded_file)
            
            # Validate image quality
            image_array = np.array(omr_image)
            quality_results = validate_image_quality(image_array)
            
            # Display quality warnings if any
            if quality_results['warnings']:
                display_processing_warnings(quality_results['warnings'])
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📋 Original OMR Sheet")
                st.image(omr_image, caption="Uploaded OMR Sheet", use_column_width=True)
                
                # Display quality metrics
                with st.expander("📊 Image Quality Metrics"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Resolution", f"{quality_results['resolution'][0]}x{quality_results['resolution'][1]}")
                        st.metric("Contrast", f"{quality_results['contrast']:.1f}")
                    with col_b:
                        st.metric("Blur Score", f"{quality_results['blur_score']:.1f}")
                        st.metric("Brightness", f"{quality_results['brightness']:.1f}")
            
            # Step 2: Load and validate answer key
            status_text.text("Loading and validating answer key...")
            progress_bar.progress(20)
            
            # Handle answer key loading
            if answer_key_option == "Use default answer key":
                if not answer_key_path or not os.path.exists(answer_key_path):
                    st.error("❌ Default answer key not found! Please upload a custom answer key.")
                    st.stop()
                    
                validate_answer_key(answer_key_path, set_choice)
                answer_key_dict = answerkey.load_answer_key(answer_key_path, set_choice)
                st.info(f"📄 Using default answer key: {os.path.basename(answer_key_path)}")
                
            else:  # Custom answer key
                if not custom_answer_key:
                    st.error("❌ Please upload a custom answer key file!")
                    st.stop()
                
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                    tmp_file.write(custom_answer_key.read())
                    temp_answer_key_path = tmp_file.name
                
                try:
                    validate_answer_key(temp_answer_key_path, set_choice)
                    answer_key_dict = answerkey.load_answer_key(temp_answer_key_path, set_choice)
                    st.info(f"📄 Using custom answer key: {custom_answer_key.name}")
                except Exception as e:
                    st.error(f"❌ Error loading custom answer key: {str(e)}")
                    st.stop()
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_answer_key_path)
                    except:
                        pass
            
            # Step 3: Preprocess image with error handling
            status_text.text("Correcting orientation and perspective...")
            progress_bar.progress(40)
            
            processed_img = safe_image_processing(
                preprocess.correct_orientation_and_perspective, 
                omr_image
            )
            
            with col2:
                st.subheader("🔧 Processed Sheet")
                st.image(processed_img, caption="Processed OMR Sheet", use_column_width=True, channels="GRAY")
            
            # Step 4: Extract responses with error handling
            status_text.text("Detecting and classifying bubbles...")
            progress_bar.progress(70)
            
            bubble_responses = safe_image_processing(
                bubbledetection.extract_responses, 
                processed_img
            )
            
            # Show debug information if enabled
            if debug_mode:
                st.subheader("🔍 Debug: Detected Responses")
                for subject, responses in bubble_responses.items():
                    with st.expander(f"{subject} - Detected Responses"):
                        # Show first 10 responses for brevity
                        response_display = []
                        for i, resp in enumerate(responses[:10]):
                            response_display.append(f"Q{i+1}: {resp if resp else 'No answer'}")
                        st.write(" | ".join(response_display))
                        if len(responses) > 10:
                            st.write(f"... and {len(responses)-10} more responses")
                        st.write(f"Total responses detected: {len(responses)}")
                        
                        # Count different response types
                        answered = len([r for r in responses if r and r != "MULTIPLE"])
                        multiple = len([r for r in responses if r == "MULTIPLE"])
                        unanswered = len([r for r in responses if not r])
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Answered", answered)
                        with col_b:
                            st.metric("Multiple", multiple) 
                        with col_c:
                            st.metric("Unanswered", unanswered)
            
            # Step 5: Calculate scores
            status_text.text("Calculating scores...")
            progress_bar.progress(90)
            
            student_scores = scoring.evaluate(bubble_responses, answer_key_dict)
            
            # Debug output for score calculation
            print(f"DEBUG - Score calculation results:")
            print(f"  student_scores = {student_scores}")
            print(f"  Total score: {student_scores.get('total', 'MISSING')}")
            print(f"  Max possible: {student_scores.get('max_possible', 'MISSING')}")
            print(f"  Percentage: {student_scores.get('percentage', 'MISSING')}")
            print(f"  Sections: {student_scores.get('sections', 'MISSING')}")
            
            processing_time = time.time() - start_time
            
            # Step 6: Save to database with error handling
            try:
                # Create exam entry
                exam_id = db.add_exam(exam_name, exam_date, subject, set_choice, 
                                    total_questions=100, max_score=100)
                
                # Save student if provided
                if student_id:
                    db.add_student(student_id, student_name)
                
                # Save results
                result_id = db.save_result(
                    student_id or f"temp_{int(time.time())}", 
                    exam_id, 
                    set_choice,
                    bubble_responses,
                    answer_key_dict,
                    student_scores['sections'],
                    student_scores['total'],
                    100,  # max_score
                    processing_time,
                    uploaded_file.name
                )
                
                # Log successful processing
                db.log_processing(uploaded_file.name, "SUCCESS", None, processing_time)
                
            except Exception as e:
                st.warning(f"Results processed but database save failed: {str(e)}")
                result_id = "temp"
                db.log_processing(uploaded_file.name, "PARTIAL_SUCCESS", str(e), processing_time)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing completed!")
            
            # Display results
            st.success(f"Processing completed in {processing_time:.2f} seconds")
            
            # Results display
            st.subheader("📊 Results Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_possible = student_scores.get('max_possible', 100)
                percentage = student_scores.get('percentage', 0)
                st.metric("Total Score", f"{student_scores['total']}/{total_possible}", 
                         f"{percentage:.1f}%")
            
            with col2:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            with col3:
                st.metric("Result ID", result_id)
            
            # Section-wise scores
            st.subheader("📈 Section-wise Performance")
            
            sections_df = pd.DataFrame([
                {"Subject": subject, "Score": score, "Total": 20, "Percentage": score/20*100}
                for subject, score in student_scores['sections'].items()
            ])
            
            fig = px.bar(sections_df, x="Subject", y="Percentage", 
                        title="Subject-wise Performance (%)",
                        color="Percentage", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed responses
            with st.expander("🔍 View Detailed Responses"):
                for subject, responses in bubble_responses.items():
                    st.write(f"**{subject}:**")
                    response_text = " | ".join([f"Q{i+1}: {resp if resp else 'No Answer'}" 
                                              for i, resp in enumerate(responses[:10])])  # Show first 10
                    st.write(response_text)
            
            # Enhanced Export options
            st.subheader("💾 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Download CSV"):
                    try:
                        csv_data = exporter.export_basic_csv()
                        st.download_button(
                            "Download Results as CSV",
                            csv_data,
                            file_name=f"omr_results_{result_id}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"CSV export failed: {str(e)}")
            
            with col2:
                if st.button("📊 Download Excel Report"):
                    try:
                        excel_data = exporter.export_detailed_excel()
                        st.download_button(
                            "Download Detailed Excel Report",
                            excel_data,
                            file_name=f"detailed_report_{result_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {str(e)}")
            
            with col3:
                if st.button("📋 Download JSON Report"):
                    try:
                        json_data = exporter.export_json_report(include_responses=True)
                        st.download_button(
                            "Download JSON Report",
                            json_data,
                            file_name=f"json_report_{result_id}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"JSON export failed: {str(e)}")
        
        except (OMRValidationError, OMRProcessingError) as e:
            handle_processing_error(e, "OMR Processing")
        except Exception as e:
            handle_processing_error(e, "Unexpected Error")
            if uploaded_file:
                db.log_processing(uploaded_file.name, "ERROR", str(e), None)
    
    else:
        st.info("👆 Please upload an OMR sheet to begin processing")
        
        # Debug information
        with st.expander("🔧 System Information"):
            answer_key_path = get_answer_key_path()
            st.write(f"**Current Working Directory:** {os.getcwd()}")
            st.write(f"**Answer Key Path:** {answer_key_path}")
            st.write(f"**Answer Key Exists:** {os.path.exists(answer_key_path)}")
            if os.path.exists(answer_key_path):
                try:
                    import pandas as pd
                    df_keys = pd.read_excel(answer_key_path, sheet_name=None)
                    st.write(f"**Available Sets:** {list(df_keys.keys())}")
                except Exception as e:
                    st.write(f"**Error reading answer key:** {str(e)}")

elif page == "Batch Processing":
    st.title("📦 Batch Processing")
    st.markdown("Process multiple OMR sheets simultaneously")
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        exam_name = st.text_input("Batch Exam Name", value="Batch Exam")
        set_choice = st.selectbox("OMR Set", ["Set - A", "Set - B"])
    
    with col2:
        exam_date = st.date_input("Batch Exam Date")
        subject = st.text_input("Batch Subject", value="General")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose multiple OMR sheets", 
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        help="Select multiple files for batch processing"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files for processing")
        
        if st.button("🚀 Start Batch Processing"):
            # Create exam entry
            exam_id = db.add_exam(exam_name, exam_date, subject, set_choice)
            
            # Load answer key
            try:
                # Use absolute path for answer key
                answer_key_path = get_answer_key_path()
                answer_key_dict = answerkey.load_answer_key(answer_key_path, set_choice)
            except Exception as e:
                st.error(f"Error loading answer key: {str(e)}")
                st.stop()
            
            # Progress tracking
            progress_bar = st.progress(0)
            results_container = st.container()
            
            batch_results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    start_time = time.time()
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Process file
                    if uploaded_file.type == "application/pdf":
                        from pdf2image import convert_from_bytes
                        images = convert_from_bytes(uploaded_file.read())
                        omr_image = images[0]
                    else:
                        omr_image = Image.open(uploaded_file)
                    
                    # Process image
                    processed_img = preprocess.correct_orientation_and_perspective(omr_image)
                    bubble_responses = bubbledetection.extract_responses(processed_img)
                    student_scores = scoring.evaluate(bubble_responses, answer_key_dict)
                    
                    processing_time = time.time() - start_time
                    
                    # Save result
                    student_id = f"batch_{i+1}"
                    result_id = db.save_result(
                        student_id, exam_id, set_choice,
                        bubble_responses, answer_key_dict,
                        student_scores['sections'], student_scores['total'],
                        100, processing_time, uploaded_file.name
                    )
                    
                    batch_results.append({
                        "File": uploaded_file.name,
                        "Student ID": student_id,
                        "Total Score": student_scores['total'],
                        "Percentage": f"{student_scores.get('percentage', 0):.1f}%",
                        "Processing Time": f"{processing_time:.2f}s",
                        "Status": "✅ Success"
                    })
                    
                    db.log_processing(uploaded_file.name, "SUCCESS", None, processing_time)
                
                except Exception as e:
                    batch_results.append({
                        "File": uploaded_file.name,
                        "Student ID": f"batch_{i+1}",
                        "Total Score": "Error",
                        "Percentage": "Error",
                        "Processing Time": "Error",
                        "Status": f"❌ Error: {str(e)[:50]}..."
                    })
                    
                    db.log_processing(uploaded_file.name, "ERROR", str(e), None)
            
            # Display batch results
            with results_container:
                st.subheader("📋 Batch Processing Results")
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df, use_container_width=True)
                
                # Summary statistics
                successful = len([r for r in batch_results if r["Status"].startswith("✅")])
                failed = len(batch_results) - successful
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", len(batch_results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)

elif page == "Dashboard":
    st.title("📊 Evaluator Dashboard")
    st.markdown("Comprehensive analytics and performance insights")
    
    # Get all results
    all_results = db.get_all_results()
    
    if all_results.empty:
        st.info("No results found. Please process some OMR sheets first.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(all_results))
        
        with col2:
            avg_score = all_results['percentage'].mean()
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col3:
            highest_score = all_results['percentage'].max()
            st.metric("Highest Score", f"{highest_score:.1f}%")
        
        with col4:
            total_exams = all_results['exam_id'].nunique()
            st.metric("Total Exams", total_exams)
        
        # Performance distribution
        st.subheader("📈 Score Distribution")
        fig = px.histogram(all_results, x="percentage", nbins=20, 
                          title="Score Distribution (%)",
                          labels={"percentage": "Score (%)", "count": "Number of Students"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent results
        st.subheader("🕒 Recent Results")
        recent_results = all_results.head(10)[['student_id', 'student_name', 'exam_name', 
                                              'total_score', 'percentage', 'created_at']]
        st.dataframe(recent_results, use_container_width=True)
        
        # Set-wise performance
        if 'set_name' in all_results.columns:
            st.subheader("📊 Set-wise Performance Comparison")
            set_performance = all_results.groupby('set_name')['percentage'].agg(['mean', 'count']).reset_index()
            set_performance.columns = ['Set', 'Average Score (%)', 'Number of Students']
            
            fig = px.bar(set_performance, x='Set', y='Average Score (%)', 
                        text='Number of Students',
                        title="Average Performance by Set")
            fig.update_traces(texttemplate='%{text} students', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

elif page == "Student Analysis":
    st.title("👨‍🎓 Student Performance Analysis")
    
    # Student selector
    all_results = db.get_all_results()
    
    if all_results.empty:
        st.info("No results found. Please process some OMR sheets first.")
    else:
        unique_students = all_results['student_id'].unique()
        selected_student = st.selectbox("Select Student", unique_students)
        
        if selected_student:
            student_results = db.get_student_results(selected_student)
            
            if not student_results.empty:
                # Student summary
                st.subheader(f"📊 Performance Summary for {selected_student}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Exams", len(student_results))
                
                with col2:
                    avg_score = student_results['percentage'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
                
                with col3:
                    best_score = student_results['percentage'].max()
                    st.metric("Best Score", f"{best_score:.1f}%")
                
                # Performance trend
                st.subheader("📈 Performance Trend")
                fig = px.line(student_results, x='created_at', y='percentage',
                             title="Performance Over Time",
                             labels={"created_at": "Date", "percentage": "Score (%)"})
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                display_columns = ['exam_name', 'exam_date', 'subject', 'set_name', 
                                 'total_score', 'percentage', 'created_at']
                st.dataframe(student_results[display_columns], use_container_width=True)

elif page == "Exam Management":
    st.title("🎓 Exam Management")
    
    tab1, tab2, tab3 = st.tabs(["📊 Exam Statistics", "📈 Subject Analysis", "📤 Export Data"])
    
    with tab1:
        st.subheader("Exam Performance Statistics")
        exam_stats = db.get_exam_statistics()
        
        if not exam_stats.empty:
            st.dataframe(exam_stats, use_container_width=True)
            
            # Visualization
            fig = px.bar(exam_stats, x='set_name', y='avg_percentage',
                        title="Average Performance by Exam Set",
                        labels={"avg_percentage": "Average Score (%)", "set_name": "Set"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No exam statistics available yet.")
    
    with tab2:
        st.subheader("Subject-wise Analysis")
        subject_analysis = db.get_subject_wise_analysis()
        
        if not subject_analysis.empty:
            st.dataframe(subject_analysis, use_container_width=True)
            
            # Subject performance chart
            fig = px.bar(subject_analysis, x='subject', y='accuracy_percentage',
                        title="Subject-wise Accuracy",
                        labels={"accuracy_percentage": "Accuracy (%)", "subject": "Subject"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No subject analysis available yet.")
    
    with tab3:
        st.subheader("Export Data")
        
        export_format = st.selectbox("Select Export Format", ["CSV", "Excel"])
        
        if st.button("📥 Export All Results"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"omr_results_export_{timestamp}.csv"
            
            try:
                exported_file = db.export_results_to_csv(filename)
                
                with open(exported_file, 'r') as f:
                    csv_data = f.read()
                
                st.download_button(
                    "Download Exported Data",
                    csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
                
                st.success(f"Data exported successfully! Click above to download.")
            
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**OMR Evaluation System v2.0**")
st.sidebar.markdown("Built with ❤️ using Streamlit")
