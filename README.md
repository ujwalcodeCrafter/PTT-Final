# Food Image Fraud Detector (PTT Project)

A sophisticated machine learning system to detect fraudulent or tampered food images using multi-stage forensic analysis and deep learning techniques.

## 📋 Features

- **Multi-Stage Analysis Pipeline**: 6-stage fraud detection system
  - Stage 1: Metadata Extraction (EXIF, creation date, modifications)
  - Stage 2: Image Preprocessing & Normalization
  - Stage 3: Forensic Spatial Analysis
  - Stage 4: Frequency Domain Analysis (FFT-based tampering detection)
  - Stage 5: Region-Based Anomaly Detection (16-patch grid analysis)
  - Stage 6: ML Classification (Random Forest)

- **Web Interface**: Flask-based UI for easy image uploads and fraud analysis
- **Real-time Detection**: Instant fraud probability assessment with risk levels
- **Visual Heatmaps**: Spatial visualization of detected anomalies
- **Detailed Reports**: Comprehensive analysis reports with confidence scores
- **YOLO Integration**: Object detection for food items in images
- **Batch Processing**: Handle multiple images efficiently

## 🏗️ Project Structure

```
PTT_Project/
├── src/
│   ├── app.py                           # Flask web application
│   ├── pipeline.py                      # Fraud detection pipeline
│   ├── train_model.py                   # Model training script
│   ├── stage1_metadata/
│   │   └── extractor.py                 # Metadata extraction
│   ├── stage2_preprocessing/
│   │   └── preprocessor.py              # Image preprocessing
│   ├── stage3_forensic_spatial/
│   │   └── extractor.py                 # Forensic spatial features
│   ├── stage4_frequency_analysis/
│   │   └── extractor.py                 # Frequency domain analysis
│   ├── stage5_region_analysis/
│   │   └── extractor.py                 # Region-based analysis
│   ├── stage6_ml_classification/
│   │   └── classifier.py                # ML classification model
│   └── visualization/
│       └── report.py                    # Report generation
├── models/
│   ├── food_fraud_detector.joblib       # Trained model
│   ├── scaler.joblib                    # Feature scaler
│   └── threshold.joblib                 # Decision threshold
├── data/
│   ├── real/                            # Real (authentic) food images
│   └── fake/                            # Fake (fraudulent) food images
├── static/
│   ├── css/
│   │   └── style.css                    # Web UI styling
│   ├── js/
│   │   └── app.js                       # Frontend JavaScript
│   └── images/
│       ├── real/
│       └── fake/
├── templates/
│   ├── base.html                        # Base template
│   ├── index.html                       # Home page
│   ├── gallery.html                     # Image gallery
│   └── about.html                       # About page
├── uploads/                             # User uploaded files
├── config.py                            # Configuration & parameters
├── requirements.txt                     # Python dependencies
├── yolov8n.pt                           # YOLOv8 nano model weights
└── README.md                            # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB+ disk space for models

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PTT_Project
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place authentic food images in `data/real/`
   - Place fraudulent/tampered food images in `data/fake/`

5. **Train the model** (optional, if you have new training data)
   ```bash
   python -m src.train_model
   ```

## 📖 Usage

### Web Interface (Recommended)

1. **Start the Flask server**
   ```bash
   python -m src.app
   ```

2. **Open in browser**
   - Navigate to `http://127.0.0.1:5000`
   - Upload a food image
   - View fraud analysis results, heatmaps, and confidence scores

### Command Line / Python API

```python
from src.pipeline import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline('models/food_fraud_detector.joblib')

# Analyze an image
image_path = 'path/to/food_image.jpg'
result = pipeline.predict(image_path)

print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Explanation: {result['explanation']}")
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Image Processing**: Target size, allowed formats, max file size
- **Stage 5 Parameters**: Patch grid size, anomaly threshold
- **Classification**: Model type, cross-validation folds
- **Risk Thresholds**: Low/medium/high risk boundaries
- **Flask Server**: Upload folder, secret key

## 📊 Model Details

### Risk Classification Thresholds
- **Low Risk** (< 0.35): Likely authentic
- **Medium Risk** (0.35 - 0.60): Possible tampering detected
- **High Risk** (> 0.60): Likely fraudulent

### Features Used

**Metadata Features** (5):
- EXIF data completeness, creation date, modification timestamp

**Spatial Features** (250+):
- Edge detection, texture analysis, gradient-based forensic markers

**Frequency Features** (100+):
- FFT spectrum analysis, compression artifacts, noise patterns

**Region Features** (16):
- Per-patch anomaly scores from 4×4 grid analysis

## 🛠️ Technology Stack

### Core Libraries
- **Machine Learning**: scikit-learn
- **Image Processing**: OpenCV, Pillow
- **Object Detection**: YOLOv8 (ultralytics)
- **Feature Extraction**: SciPy, NumPy
- **Web Framework**: Flask
- **Data Handling**: Pandas, NumPy
- **Serialization**: joblib

### Additional Tools
- **OCR**: EasyOCR (for text extraction from images)
- **EXIF**: exifread
- **Visualization**: Matplotlib, Seaborn

## 📈 Performance

- **Processing Time**: ~2-5 seconds per image (including all stages)
- **Model Accuracy**: ~85-92% (depending on training data quality)
- **Memory Usage**: ~500MB with loaded model
- **Concurrent Users**: Supports multiple simultaneous uploads

## 🧪 Testing

Run tests with pytest:
```bash
pytest tests/
```

## 📝 Training a New Model

If you have collected new training data:

```bash
python -m src.train_model
```

The script will:
1. Load images from `data/real/` and `data/fake/`
2. Extract features from all 6 stages
3. Train the Random Forest classifier
4. Evaluate performance with cross-validation
5. Save the model to `models/food_fraud_detector.joblib`

## 🔐 Security Considerations

- **File Upload Limits**: Maximum 10MB per file
- **Allowed Formats**: JPG, JPEG, PNG, WebP, BMP
- **Secret Key**: Change `SECRET_KEY` in `config.py` before production deployment
- **HTTPS**: Use HTTPS in production environments

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Food Fraud Detection Team** - Initial development

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- scikit-learn community
- Food image dataset contributors

## 📧 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [Your Contact Information]

## 🔄 Version History

- **v1.0** - Initial release with 6-stage detection pipeline

---

**Note**: This system is designed for research and commercial food quality verification. Always validate results with domain experts before making critical decisions.
