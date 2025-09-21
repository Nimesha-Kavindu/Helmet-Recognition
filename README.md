# Motorcycle Helmet Detection System

A real-time desktop application for detecting motorcycle helmets using YOLOv8 and Tkinter.

## Features
- Real-time helmet detection from webcam
- Desktop GUI with Tkinter
- Optimized for CPU inference
- Detection visualization with bounding boxes
- Detection statistics

## Setup Instructions

### 1. Install Python 3.11+
Make sure you have Python 3.11 or higher installed.

### 2. Install Required Packages
Due to network issues, install packages individually:

```bash
# Core packages
pip install opencv-python
pip install pillow
pip install numpy

# ML packages (these might need special handling)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

### 3. Download YOLO Model
The app will automatically download YOLOv8n model on first run, or you can download manually:

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # This will download the model
```

### 4. Run the Application
```bash
python src/helmet_detection_app.py
```

## Project Structure
```
Helmet Recognition/
├── src/
│   └── helmet_detection_app.py    # Main application
├── models/                        # YOLO models storage
├── data/                         # Dataset storage
├── utils/                        # Utility functions
├── config.py                     # Configuration settings
├── requirements.txt              # Package dependencies
└── README.md                     # This file
```

## Troubleshooting

### Network Issues
If you encounter network errors during package installation:
1. Try installing packages one by one
2. Use offline installers if available
3. Check your internet connection

### Camera Issues
- Make sure your webcam is connected
- Grant camera permissions to the application
- Try different camera indices (0, 1, 2) if default doesn't work

### Performance Issues
- The app is optimized for CPU inference
- Lower the FPS if needed (modify config.py)
- Consider using lighter YOLO models (YOLOv8n is the lightest)

## Next Steps
1. Add custom helmet detection training
2. Improve detection accuracy
3. Add detection logging
4. Export detection results