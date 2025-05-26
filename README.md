# BrailleCraft: Sign-Language to Braille 
A real-time American Sign Language (ASL) recognition system that uses computer vision and machine learning to translate hand gestures into text. The system features a modern GUI interface, dual-model architecture for improved accuracy, and comprehensive data augmentation capabilities.

## 🌟 Features

- **Real-time Recognition**: Live hand gesture recognition using webcam input
- **Dual Model Architecture**: Separate models for A-L and M-Z letters for optimized accuracy
- **Modern GUI Interface**: User-friendly Tkinter-based interface with confidence indicators
- **Data Augmentation**: Advanced landmark augmentation for improved model training
- **Comprehensive Reporting**: Detailed accuracy reports and performance visualization
- **Text Formation**: Build words and sentences from recognized letters
- **Arduino Integration**: Support for hardware integration (expandable)
- **Export Capabilities**: Save recognized text and generate performance reports


## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera for real-time recognition
- At least 4GB RAM recommended


### Required Dependencies

```
opencv-python>=4.5.0
mediapipe>=0.8.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
joblib>=1.1.0
Pillow>=8.3.0
```

## 🎯 Quick Start

1. **Prepare Your Models**: Ensure you have trained models in the specified directory:
   - `random_forest_model.joblib` (A-L letters)
   - `random_forest_model_m_to_z.joblib` (M-Z letters)

2. **Run the Application**:
```bash
python app.py
```

3. **Start Recognition**:
   - Position your hand in front of the camera
   - Select the sign you want to test from the dropdown
   - Watch real-time predictions with confidence scores
   - Use keyboard shortcuts for quick actions

### Model Architecture

- **Model 1 (A-L)**: Random Forest classifier trained on letters A through L
- **Model 2 (M-Z)**: Random Forest classifier trained on letters M through Z
- **Feature Extraction**: 63-dimensional feature vectors from MediaPipe hand landmarks
- **Confidence Scoring**: Probability-based confidence for each prediction

## 💻 Usage

### GUI Interface

The application provides several interface elements:

#### Main Display
- **Video Feed**: Real-time camera input with hand landmark visualization
- **Predicted Character**: Large display of the recognized letter
- **Confidence Score**: Percentage confidence of the prediction
- **Formed Text**: Current word/sentence being built

#### Controls
- **Model Selection**: Choose between Model 1 (A-L), Model 2 (M-Z), or Both
- **Test Sign**: Select which sign you're testing for accuracy measurement
- **Text Controls**: Add letters, spaces, delete, and clear text

#### Keyboard Shortcuts
- `Space`: Add current predicted letter to text
- `Enter`: Add space to text
- `Backspace`: Delete last character
- `Ctrl+S`: Save current text to file


## 🔄 Data Augmentation

The system includes comprehensive data augmentation capabilities:

### Augmentation Techniques

1. **Rotation**: Rotate landmarks around center point (-30° to +30°)
2. **Scaling**: Scale landmarks (0.8x to 1.2x)
3. **Translation**: Translate landmarks (-0.1 to +0.1 units)
4. **Noise Addition**: Add Gaussian noise (σ = 0.01)
5. **Horizontal Flipping**: Mirror landmarks horizontally


### Feature Extraction

The system extracts 63 features from MediaPipe hand landmarks:
- 21 hand landmarks × 3 coordinates (x, y, z) = 63 features
- Normalized coordinates relative to hand bounding box
- Additional angle calculations between finger joints

### Training Process

1. **Data Collection**: Collect hand landmark data for each letter
2. **Data Augmentation**: Apply transformations to increase dataset size
3. **Feature Processing**: Extract and normalize landmark features
4. **Model Training**: Train Random Forest classifiers
5. **Model Evaluation**: Test accuracy and generate performance reports

## 🎨 GUI Features

### Visual Indicators

- **Confidence-based Coloring**:
  - Green: High confidence (≥80%)
  - Orange: Medium confidence (50-79%)
  - Red: Low confidence (<50%)

### Real-time Feedback

- **Correctness Indicator**: Shows if prediction matches selected test sign
- **Performance Tracking**: Collects accuracy data during testing
- **Live Statistics**: Real-time confidence and prediction updates

### Report Generation

Generate comprehensive reports including:
- Overall accuracy statistics
- Per-letter accuracy breakdown
- Performance visualizations
- Detailed CSV data export


**Made with ❤️ for the deaf and hard-of-hearing community**
