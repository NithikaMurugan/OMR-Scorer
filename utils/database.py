import sqlite3
import json
import pandas as pd
from datetime import datetime
import os

class OMRDatabase:
    def __init__(self, db_path="database/results.db"):
        self.db_path = db_path
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Students table
            c.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE,
                    name TEXT,
                    class TEXT,
                    roll_number TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Exams table
            c.execute('''
                CREATE TABLE IF NOT EXISTS exams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_name TEXT NOT NULL,
                    exam_date DATE,
                    subject TEXT,
                    set_name TEXT,
                    total_questions INTEGER,
                    max_score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Enhanced results table
            c.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    exam_id INTEGER,
                    set_name TEXT,
                    total_score INTEGER,
                    max_score INTEGER,
                    percentage REAL,
                    processing_time REAL,
                    image_path TEXT,
                    responses_json TEXT,
                    section_scores_json TEXT,
                    confidence_scores_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (exam_id) REFERENCES exams (id)
                )
            ''')
            
            # Detailed responses table
            c.execute('''
                CREATE TABLE IF NOT EXISTS detailed_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER,
                    subject TEXT,
                    question_number INTEGER,
                    student_answer TEXT,
                    correct_answer TEXT,
                    is_correct BOOLEAN,
                    confidence_score REAL,
                    FOREIGN KEY (result_id) REFERENCES results (id)
                )
            ''')
            
            # Processing logs table
            c.execute('''
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    processing_status TEXT,
                    error_message TEXT,
                    processing_time REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            c.execute('CREATE INDEX IF NOT EXISTS idx_results_student_id ON results(student_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_results_exam_id ON results(exam_id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_detailed_responses_result_id ON detailed_responses(result_id)')
            
            conn.commit()
    
    def add_student(self, student_id, name=None, class_name=None, roll_number=None):
        """Add or update student information."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO students (student_id, name, class, roll_number)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, class_name, roll_number))
            conn.commit()
    
    def add_exam(self, exam_name, exam_date=None, subject=None, set_name=None, 
                 total_questions=100, max_score=100):
        """Add exam information."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO exams (exam_name, exam_date, subject, set_name, total_questions, max_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (exam_name, exam_date, subject, set_name, total_questions, max_score))
            conn.commit()
            return c.lastrowid
    
    def save_result(self, student_id, exam_id, set_name, student_responses, 
                   answer_key, section_scores, total_score, max_score, 
                   processing_time=None, image_path=None, confidence_scores=None):
        """Save comprehensive result data."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Calculate percentage
            percentage = (total_score / max_score * 100) if max_score > 0 else 0
            
            # Convert data to JSON strings
            responses_json = json.dumps(student_responses)
            section_scores_json = json.dumps(section_scores)
            confidence_json = json.dumps(confidence_scores) if confidence_scores else None
            
            # Insert main result
            c.execute('''
                INSERT INTO results (
                    student_id, exam_id, set_name, total_score, max_score, percentage,
                    processing_time, image_path, responses_json, section_scores_json,
                    confidence_scores_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, exam_id, set_name, total_score, max_score, percentage,
                  processing_time, image_path, responses_json, section_scores_json,
                  confidence_json))
            
            result_id = c.lastrowid
            
            # Insert detailed responses
            for subject, responses in student_responses.items():
                correct_answers = answer_key.get(subject, [])
                for i, (student_answer, correct_answer) in enumerate(zip(responses, correct_answers)):
                    is_correct = student_answer == correct_answer
                    confidence = confidence_scores.get(subject, {}).get(i, 0.0) if confidence_scores else 0.0
                    
                    c.execute('''
                        INSERT INTO detailed_responses (
                            result_id, subject, question_number, student_answer,
                            correct_answer, is_correct, confidence_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (result_id, subject, i+1, student_answer, correct_answer, is_correct, confidence))
            
            conn.commit()
            return result_id
    
    def log_processing(self, image_path, status, error_message=None, processing_time=None):
        """Log processing attempts."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO processing_logs (image_path, processing_status, error_message, processing_time)
                VALUES (?, ?, ?, ?)
            ''', (image_path, status, error_message, processing_time))
            conn.commit()
    
    def get_student_results(self, student_id):
        """Get all results for a specific student."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('''
                SELECT r.*, e.exam_name, e.exam_date, e.subject
                FROM results r
                LEFT JOIN exams e ON r.exam_id = e.id
                WHERE r.student_id = ?
                ORDER BY r.created_at DESC
            ''', conn, params=[student_id])
    
    def get_exam_statistics(self, exam_id=None):
        """Get comprehensive exam statistics."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    COUNT(*) as total_students,
                    AVG(total_score) as avg_score,
                    MAX(total_score) as max_score,
                    MIN(total_score) as min_score,
                    AVG(percentage) as avg_percentage,
                    set_name
                FROM results
            '''
            params = []
            
            if exam_id:
                query += ' WHERE exam_id = ?'
                params.append(exam_id)
            
            query += ' GROUP BY set_name'
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_subject_wise_analysis(self, exam_id=None):
        """Get subject-wise performance analysis."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    subject,
                    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_percentage,
                    COUNT(*) as total_questions,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers
                FROM detailed_responses dr
                JOIN results r ON dr.result_id = r.id
            '''
            params = []
            
            if exam_id:
                query += ' WHERE r.exam_id = ?'
                params.append(exam_id)
            
            query += ' GROUP BY subject ORDER BY accuracy_percentage DESC'
            
            return pd.read_sql_query(query, conn, params=params)
    
    def get_question_wise_analysis(self, subject, exam_id=None):
        """Get question-wise analysis for a specific subject."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    question_number,
                    correct_answer,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_count,
                    AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100 as accuracy_percentage,
                    AVG(confidence_score) as avg_confidence
                FROM detailed_responses dr
                JOIN results r ON dr.result_id = r.id
                WHERE dr.subject = ?
            '''
            params = [subject]
            
            if exam_id:
                query += ' AND r.exam_id = ?'
                params.append(exam_id)
            
            query += ' GROUP BY question_number, correct_answer ORDER BY question_number'
            
            return pd.read_sql_query(query, conn, params=params)
    
    def export_results_to_csv(self, filename, exam_id=None):
        """Export results to CSV file."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    r.student_id,
                    s.name as student_name,
                    s.class,
                    s.roll_number,
                    e.exam_name,
                    e.exam_date,
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
            df.to_csv(filename, index=False)
            return filename
    
    def get_all_results(self):
        """Get all results for dashboard display."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query('''
                SELECT 
                    r.*,
                    e.exam_name,
                    e.exam_date,
                    e.subject as exam_subject,
                    s.name as student_name,
                    s.class as student_class
                FROM results r
                LEFT JOIN exams e ON r.exam_id = e.id
                LEFT JOIN students s ON r.student_id = s.student_id
                ORDER BY r.created_at DESC
            ''', conn)