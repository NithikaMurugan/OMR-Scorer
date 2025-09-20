import pandas as pd
import json
import os
from datetime import datetime
import xlsxwriter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import io
import base64

class OMRExporter:
    """Comprehensive export functionality for OMR results"""
    
    def __init__(self, database):
        self.db = database
    
    def export_basic_csv(self, exam_id=None, filename=None):
        """
        Export basic results to CSV format.
        
        Args:
            exam_id: Optional exam ID to filter results
            filename: Optional custom filename
            
        Returns:
            str: CSV data as string
        """
        # Get results from database
        with self.db.db_path as conn:
            query = '''
                SELECT 
                    r.student_id,
                    s.name as student_name,
                    s.class as student_class,
                    s.roll_number,
                    e.exam_name,
                    e.exam_date,
                    e.subject,
                    r.set_name,
                    r.total_score,
                    r.max_score,
                    r.percentage,
                    r.created_at
                FROM results r
                LEFT JOIN students s ON r.student_id = s.student_id
                LEFT JOIN exams e ON r.exam_id = e.id
            '''
            params = []
            
            if exam_id:
                query += ' WHERE r.exam_id = ?'
                params.append(exam_id)
            
            query += ' ORDER BY r.percentage DESC'
            
            df = pd.read_sql_query(query, conn, params=params)
        
        # Format the data
        df['exam_date'] = pd.to_datetime(df['exam_date']).dt.strftime('%Y-%m-%d')
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df['percentage'] = df['percentage'].round(2)
        
        return df.to_csv(index=False)
    
    def export_detailed_excel(self, exam_id=None, filename=None):
        """
        Export detailed results to Excel with multiple sheets and formatting.
        
        Args:
            exam_id: Optional exam ID to filter results
            filename: Optional custom filename
            
        Returns:
            bytes: Excel file content
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"omr_detailed_results_{timestamp}.xlsx"
        
        # Create Excel workbook
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        data_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        percentage_format = workbook.add_format({
            'num_format': '0.00%',
            'align': 'center',
            'border': 1
        })
        
        # Sheet 1: Summary Results
        self._create_summary_sheet(workbook, header_format, data_format, percentage_format, exam_id)
        
        # Sheet 2: Detailed Responses
        self._create_detailed_responses_sheet(workbook, header_format, data_format, exam_id)
        
        # Sheet 3: Statistics
        self._create_statistics_sheet(workbook, header_format, data_format, percentage_format, exam_id)
        
        # Sheet 4: Subject Analysis
        self._create_subject_analysis_sheet(workbook, header_format, data_format, percentage_format, exam_id)
        
        workbook.close()
        output.seek(0)
        
        return output.getvalue()
    
    def _create_summary_sheet(self, workbook, header_format, data_format, percentage_format, exam_id):
        """Create summary results sheet"""
        worksheet = workbook.add_worksheet('Summary Results')
        
        # Get summary data
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            query = '''
                SELECT 
                    r.student_id,
                    COALESCE(s.name, 'N/A') as student_name,
                    COALESCE(s.class, 'N/A') as student_class,
                    COALESCE(s.roll_number, 'N/A') as roll_number,
                    e.exam_name,
                    e.exam_date,
                    e.subject,
                    r.set_name,
                    r.total_score,
                    r.max_score,
                    r.percentage / 100.0 as percentage,
                    r.processing_time,
                    r.created_at
                FROM results r
                LEFT JOIN students s ON r.student_id = s.student_id
                LEFT JOIN exams e ON r.exam_id = e.id
            '''
            params = []
            
            if exam_id:
                query += ' WHERE r.exam_id = ?'
                params.append(exam_id)
            
            query += ' ORDER BY r.percentage DESC'
            
            df = pd.read_sql_query(query, conn, params=params)
        
        # Write headers
        headers = ['Student ID', 'Name', 'Class', 'Roll Number', 'Exam', 'Date', 'Subject', 
                  'Set', 'Score', 'Max Score', 'Percentage', 'Processing Time', 'Submitted']
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Write data
        for row, (_, record) in enumerate(df.iterrows(), start=1):
            worksheet.write(row, 0, record['student_id'], data_format)
            worksheet.write(row, 1, record['student_name'], data_format)
            worksheet.write(row, 2, record['student_class'], data_format)
            worksheet.write(row, 3, record['roll_number'], data_format)
            worksheet.write(row, 4, record['exam_name'], data_format)
            worksheet.write(row, 5, record['exam_date'], data_format)
            worksheet.write(row, 6, record['subject'], data_format)
            worksheet.write(row, 7, record['set_name'], data_format)
            worksheet.write(row, 8, record['total_score'], data_format)
            worksheet.write(row, 9, record['max_score'], data_format)
            worksheet.write(row, 10, record['percentage'], percentage_format)
            worksheet.write(row, 11, f"{record['processing_time']:.2f}s" if record['processing_time'] else 'N/A', data_format)
            worksheet.write(row, 12, record['created_at'], data_format)
        
        # Auto-adjust column widths
        for col in range(len(headers)):
            worksheet.set_column(col, col, 15)
    
    def _create_detailed_responses_sheet(self, workbook, header_format, data_format, exam_id):
        """Create detailed responses sheet"""
        worksheet = workbook.add_worksheet('Detailed Responses')
        
        # Get detailed responses
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            query = '''
                SELECT 
                    r.student_id,
                    dr.subject,
                    dr.question_number,
                    dr.student_answer,
                    dr.correct_answer,
                    dr.is_correct,
                    dr.confidence_score
                FROM detailed_responses dr
                JOIN results r ON dr.result_id = r.id
            '''
            params = []
            
            if exam_id:
                query += ' WHERE r.exam_id = ?'
                params.append(exam_id)
            
            query += ' ORDER BY r.student_id, dr.subject, dr.question_number'
            
            df = pd.read_sql_query(query, conn, params=params)
        
        # Write headers
        headers = ['Student ID', 'Subject', 'Question', 'Student Answer', 'Correct Answer', 
                  'Correct', 'Confidence']
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Write data
        for row, (_, record) in enumerate(df.iterrows(), start=1):
            worksheet.write(row, 0, record['student_id'], data_format)
            worksheet.write(row, 1, record['subject'], data_format)
            worksheet.write(row, 2, record['question_number'], data_format)
            worksheet.write(row, 3, record['student_answer'] or 'No Answer', data_format)
            worksheet.write(row, 4, record['correct_answer'], data_format)
            worksheet.write(row, 5, 'Yes' if record['is_correct'] else 'No', data_format)
            worksheet.write(row, 6, f"{record['confidence_score']:.2f}" if record['confidence_score'] else 'N/A', data_format)
        
        # Auto-adjust column widths
        for col in range(len(headers)):
            worksheet.set_column(col, col, 12)
    
    def _create_statistics_sheet(self, workbook, header_format, data_format, percentage_format, exam_id):
        """Create statistics sheet"""
        worksheet = workbook.add_worksheet('Statistics')
        
        # Get statistics
        stats_data = self.db.get_exam_statistics(exam_id)
        
        row = 0
        # Title
        worksheet.write(row, 0, 'Exam Statistics Summary', header_format)
        worksheet.merge_range(row, 0, row, 5, 'Exam Statistics Summary', header_format)
        row += 2
        
        # Overall statistics
        if not stats_data.empty:
            # Headers
            headers = ['Set Name', 'Total Students', 'Average Score', 'Max Score', 'Min Score', 'Average %']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            row += 1
            
            # Data
            for _, record in stats_data.iterrows():
                worksheet.write(row, 0, record['set_name'], data_format)
                worksheet.write(row, 1, record['total_students'], data_format)
                worksheet.write(row, 2, f"{record['avg_score']:.1f}", data_format)
                worksheet.write(row, 3, record['max_score'], data_format)
                worksheet.write(row, 4, record['min_score'], data_format)
                worksheet.write(row, 5, record['avg_percentage'] / 100.0, percentage_format)
                row += 1
        
        # Auto-adjust column widths
        for col in range(6):
            worksheet.set_column(col, col, 15)
    
    def _create_subject_analysis_sheet(self, workbook, header_format, data_format, percentage_format, exam_id):
        """Create subject analysis sheet"""
        worksheet = workbook.add_worksheet('Subject Analysis')
        
        # Get subject analysis
        subject_data = self.db.get_subject_wise_analysis(exam_id)
        
        row = 0
        # Title
        worksheet.write(row, 0, 'Subject-wise Performance Analysis', header_format)
        worksheet.merge_range(row, 0, row, 3, 'Subject-wise Performance Analysis', header_format)
        row += 2
        
        if not subject_data.empty:
            # Headers
            headers = ['Subject', 'Accuracy %', 'Total Questions', 'Correct Answers']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            row += 1
            
            # Data
            for _, record in subject_data.iterrows():
                worksheet.write(row, 0, record['subject'], data_format)
                worksheet.write(row, 1, record['accuracy_percentage'] / 100.0, percentage_format)
                worksheet.write(row, 2, record['total_questions'], data_format)
                worksheet.write(row, 3, record['correct_answers'], data_format)
                row += 1
        
        # Auto-adjust column widths
        for col in range(4):
            worksheet.set_column(col, col, 20)
    
    def export_json_report(self, exam_id=None, include_responses=True):
        """
        Export comprehensive JSON report.
        
        Args:
            exam_id: Optional exam ID to filter results
            include_responses: Whether to include detailed responses
            
        Returns:
            str: JSON report
        """
        report = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'exam_id': exam_id,
                'include_responses': include_responses
            },
            'summary': {},
            'results': [],
            'statistics': {},
            'subject_analysis': []
        }
        
        # Get summary statistics
        stats = self.db.get_exam_statistics(exam_id)
        if not stats.empty:
            report['statistics'] = stats.to_dict('records')
        
        # Get subject analysis
        subject_analysis = self.db.get_subject_wise_analysis(exam_id)
        if not subject_analysis.empty:
            report['subject_analysis'] = subject_analysis.to_dict('records')
        
        # Get detailed results
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            query = '''
                SELECT 
                    r.*,
                    s.name as student_name,
                    s.class as student_class,
                    e.exam_name,
                    e.exam_date,
                    e.subject as exam_subject
                FROM results r
                LEFT JOIN students s ON r.student_id = s.student_id
                LEFT JOIN exams e ON r.exam_id = e.id
            '''
            params = []
            
            if exam_id:
                query += ' WHERE r.exam_id = ?'
                params.append(exam_id)
            
            results_df = pd.read_sql_query(query, conn, params=params)
        
        # Process results
        for _, result in results_df.iterrows():
            result_data = {
                'student_id': result['student_id'],
                'student_name': result['student_name'],
                'exam_name': result['exam_name'],
                'exam_date': result['exam_date'],
                'set_name': result['set_name'],
                'total_score': result['total_score'],
                'max_score': result['max_score'],
                'percentage': result['percentage'],
                'section_scores': json.loads(result['section_scores_json']) if result['section_scores_json'] else {},
                'processing_time': result['processing_time'],
                'created_at': result['created_at']
            }
            
            if include_responses and result['responses_json']:
                result_data['responses'] = json.loads(result['responses_json'])
            
            report['results'].append(result_data)
        
        # Summary statistics
        if report['results']:
            scores = [r['percentage'] for r in report['results']]
            report['summary'] = {
                'total_students': len(report['results']),
                'average_score': sum(scores) / len(scores),
                'highest_score': max(scores),
                'lowest_score': min(scores),
                'median_score': sorted(scores)[len(scores)//2]
            }
        
        return json.dumps(report, indent=2, default=str)
    
    def create_student_report_card(self, student_id, exam_id=None):
        """
        Create individual student report card.
        
        Args:
            student_id: Student ID
            exam_id: Optional exam ID
            
        Returns:
            dict: Student report data
        """
        student_results = self.db.get_student_results(student_id)
        
        if exam_id:
            student_results = student_results[student_results['exam_id'] == exam_id]
        
        if student_results.empty:
            return None
        
        # Get detailed responses for the student
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            query = '''
                SELECT 
                    dr.*,
                    r.set_name,
                    e.exam_name
                FROM detailed_responses dr
                JOIN results r ON dr.result_id = r.id
                JOIN exams e ON r.exam_id = e.id
                WHERE r.student_id = ?
            '''
            params = [student_id]
            
            if exam_id:
                query += ' AND r.exam_id = ?'
                params.append(exam_id)
            
            detailed_responses = pd.read_sql_query(query, conn, params=params)
        
        # Create report card
        report_card = {
            'student_info': {
                'student_id': student_id,
                'name': student_results.iloc[0]['student_name'] if 'student_name' in student_results.columns else 'N/A',
                'total_exams': len(student_results)
            },
            'performance_summary': {
                'average_score': student_results['percentage'].mean(),
                'best_score': student_results['percentage'].max(),
                'worst_score': student_results['percentage'].min(),
                'total_score': student_results['total_score'].sum(),
                'max_possible': student_results['max_score'].sum()
            },
            'exam_details': student_results.to_dict('records'),
            'subject_performance': {}
        }
        
        # Subject-wise performance
        if not detailed_responses.empty:
            subject_perf = detailed_responses.groupby('subject').agg({
                'is_correct': ['count', 'sum', 'mean'],
                'confidence_score': 'mean'
            }).round(2)
            
            subject_perf.columns = ['total_questions', 'correct_answers', 'accuracy', 'avg_confidence']
            report_card['subject_performance'] = subject_perf.to_dict('index')
        
        return report_card