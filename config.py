# Configuration file for Helmet Detection App

# Model settings
MODEL_PATH = "models/yolov8n.pt"  # Path to YOLO model
CONFIDENCE_THRESHOLD = 0.5        # Minimum confidence for detection
NMS_THRESHOLD = 0.4              # Non-maximum suppression threshold

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Detection classes (COCO dataset indices)
PERSON_CLASS = 0
HELMET_CLASSES = {
    'helmet': 1,
    'motorcycle': 3,
    'bicycle': 1
}

# GUI settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
VIDEO_DISPLAY_WIDTH = 640
VIDEO_DISPLAY_HEIGHT = 480

# Colors for bounding boxes (BGR format)
COLORS = {
    'helmet_detected': (0, 255, 0),      # Green
    'no_helmet': (0, 0, 255),            # Red
    'person': (255, 0, 0),               # Blue
    'motorcycle': (255, 255, 0)          # Cyan
}