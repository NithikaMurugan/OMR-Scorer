# 🎯 OMR-Scorer: Automated OMR Evaluation System

A comprehensive, AI-powered Optical Mark Recognition (OMR) system for automated evaluation of answer sheets. This system provides end-to-end processing from image capture to detailed analytics with a user-friendly web interface.

## 🚀 Features

### Core Functionality
- **📱 Mobile-Friendly Upload**: Process OMR sheets captured via mobile devices
- **🔧 Advanced Image Processing**: Automatic orientation detection and perspective correction
- **🎯 Intelligent Bubble Detection**: AI-powered bubble classification using computer vision and CNN models
- **📊 Multi-Set Support**: Handle different question paper sets (Set-A, Set-B, etc.)
- **📦 Batch Processing**: Process multiple OMR sheets simultaneously
- **💾 Secure Database Storage**: Comprehensive data storage with SQLite backend

### Advanced Processing Pipeline
1. **Sheet Orientation Detection**: Automatic detection and correction of sheet rotation
2. **Perspective Rectification**: Correction of camera angle distortions
3. **Registration Marker Detection**: Precise alignment using corner markers
4. **Grid Identification**: Intelligent bubble grid detection and extraction
5. **Bubble Classification**: Multi-method bubble state detection (filled/unfilled)
6. **Answer Validation**: Support for multiple selections and invalid responses

### Dashboard & Analytics
- **📈 Real-time Dashboard**: Comprehensive evaluator dashboard with performance insights
- **👨‍🎓 Student Analytics**: Individual student performance tracking
- **📊 Subject-wise Analysis**: Detailed breakdown by subjects and topics
- **📉 Statistical Reports**: Aggregate statistics and performance distributions
- **🔍 Question-wise Analysis**: Identify difficult questions and common mistakes

### Export & Reporting
- **📄 CSV/Excel Export**: Multiple export formats with detailed formatting
- **📊 Comprehensive Reports**: Multi-sheet Excel reports with statistics
- **📋 Individual Report Cards**: Personalized student performance reports
- **📈 Visual Analytics**: Interactive charts and performance visualizations

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV, PIL
- **Machine Learning**: TensorFlow/Keras (for CNN-based classification)
- **Database**: SQLite
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Export**: xlsxwriter, openpyxl

## 📋 System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space
- Windows/Linux/macOS

### Recommended Requirements
- Python 3.9+
- 8GB RAM
- 5GB free disk space
- Dedicated GPU (for CNN processing)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/NithikaMurugan/OMR-Scorer.git
   cd OMR-Scorer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv cv_env
   # Windows
   cv_env\Scripts\activate
   # Linux/macOS
   source cv_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**
   ```bash
   python -c "from utils.database import OMRDatabase; OMRDatabase().init_database()"
   ```

## 🎮 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Processing OMR Sheets

#### Single Sheet Processing
1. Navigate to **OMR Processing** page
2. Upload an OMR sheet (JPG, PNG, or PDF)
3. Configure exam details and select answer key set
4. Click **Process** and view results
5. Export results in various formats

#### Batch Processing
1. Navigate to **Batch Processing** page
2. Upload multiple OMR sheets
3. Configure batch settings
4. Start batch processing and monitor progress
5. Download batch results

### Dashboard Navigation

#### 📊 Main Dashboard
- View overall performance statistics
- Monitor score distributions
- Access recent results
- Compare set-wise performance

#### 👨‍🎓 Student Analysis
- Select individual students
- View performance trends
- Access detailed score history
- Generate student report cards

#### 🎓 Exam Management
- View exam statistics
- Analyze subject-wise performance
- Export comprehensive reports
- Manage answer keys

## 📁 Project Structure

```
OMR-Scorer/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── analysis/                   # Analysis and visualization scripts
│   ├── README.md              # Analysis scripts documentation
│   ├── analyze_omr_structure.py
│   ├── bubble_analysis.py
│   └── ...                    # Other analysis tools
├── database/
│   └── results.db             # SQLite database
├── debug/                      # Debug and diagnostic scripts
│   ├── README.md              # Debug scripts documentation
│   ├── bubble_position_debug.py
│   ├── debug_bubble_recognition.py
│   └── ...                    # Other debug tools
├── images/                     # Generated images and visualizations
│   ├── README.md              # Images directory documentation
│   └── ...                    # Debug/analysis output images
├── logs/                       # Application log files
│   ├── README.md              # Logs directory documentation
│   └── omr_processing.log     # Main processing logs
├── Models/
│   └── cnn_modal.h5           # Pre-trained CNN model
├── sampledata/
│   ├── answer_key.xlsx.xlsx   # Answer key templates
│   └── omr_sheets/            # Sample OMR sheets
│       ├── set_A/
│       └── set_B/
├── tests/                      # Test scripts
│   ├── README.md              # Test scripts documentation
│   ├── test_accuracy.py
│   ├── test_bubble_detection.py
│   └── ...                    # Other test files
├── train/                      # Training and evaluation scripts
│   ├── evaluate_scoring.py
│   ├── train_cnn.py
│   └── prepare_bubble_dataset.py
└── utils/                      # Core utility modules
    ├── preprocess.py          # Image preprocessing
    ├── bubbledetection.py     # Bubble detection & classification
    ├── scoring.py             # Score calculation
    ├── answerkey.py           # Answer key management
    ├── database.py            # Database operations
    ├── validation.py          # Input validation & error handling
    ├── export.py              # Export functionality
    └── ...                    # Other utility scripts
```

## 📊 Database Schema

### Tables Overview
- **students**: Student information and metadata
- **exams**: Exam configuration and details
- **results**: Main results with scores and performance
- **detailed_responses**: Question-wise response tracking
- **processing_logs**: System processing logs and errors

## 🎯 Image Quality Requirements

### Optimal Conditions
- **Resolution**: Minimum 800x1000 pixels (recommended: 1200x1600+)
- **Lighting**: Even, bright lighting without shadows
- **Angle**: Direct overhead capture (perpendicular to sheet)
- **Focus**: Sharp, clear image without blur
- **Format**: JPG, PNG, or PDF

### Acceptable Variations
- Slight rotation (±15 degrees) - automatically corrected
- Minor perspective distortion - automatically rectified
- Moderate lighting variations - enhanced during processing

## 🔧 Configuration

### Answer Key Setup
1. Create Excel file with answer keys
2. Use separate sheets for different sets (Set-A, Set-B)
3. Format: Subject columns with answers (A, B, C, D)
4. Place in `sampledata/` directory

### CNN Model (Optional)
- Place trained CNN model as `Models/cnn_modal.h5`
- Model should accept 28x28 grayscale images
- Output: Binary classification (filled/unfilled)

## 🚨 Troubleshooting

### Common Issues

#### Low Processing Accuracy
- **Cause**: Poor image quality or lighting
- **Solution**: Retake image with better lighting and focus

#### File Upload Errors
- **Cause**: Unsupported format or corrupted file
- **Solution**: Use JPG/PNG format, ensure file integrity

#### Database Connection Issues
- **Cause**: Permissions or disk space
- **Solution**: Check file permissions and available storage

#### Performance Issues
- **Cause**: Large file sizes or insufficient memory
- **Solution**: Reduce image size or close other applications

### Error Messages
The system provides detailed error messages with suggestions:
- **Validation Errors**: Input format and quality issues
- **Processing Errors**: Image processing and analysis problems
- **System Errors**: Database and file system issues

## 📈 Performance Optimization

### Image Processing
- Resize large images to optimal dimensions
- Use appropriate compression for uploads
- Process in grayscale when possible

### Batch Processing
- Process files in smaller batches for large datasets
- Monitor system resources during processing
- Use background processing for large operations

## 🔒 Security Considerations

- Local data storage (no cloud dependency)
- Input validation and sanitization
- Error handling without data exposure
- Secure file processing pipeline

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Nithika Murugan** - *Initial work* - [NithikaMurugan](https://github.com/NithikaMurugan)

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Streamlit team for the excellent web framework
- TensorFlow team for machine learning capabilities
- Contributors and testers who helped improve the system

## 📞 Support

For support, email support@omr-scorer.com or create an issue on GitHub.

## 🔄 Version History

- **v2.0.0** - Complete system overhaul with enhanced features
- **v1.0.0** - Initial release with basic OMR processing

---

**Built with ❤️ for educational institutions worldwide**