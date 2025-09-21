"""
Motorcycle Helmet Detection Desktop Application
Real-time helmet detection using YOLOv8 and Tkinter GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import sys
from ultralytics import YOLO
import numpy as np

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils'))
from helmet_detector import HelmetDetector

class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motorcycle Helmet Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.cap = None
        self.is_detecting = False
        self.model = None
        self.model_loaded = False
        self.detection_count = 0
        
        # Initialize helmet detector
        self.helmet_detector = HelmetDetector()
        
        # Load models
        self.load_model()
        
        # Create GUI elements
        self.setup_gui()
        
    def load_model(self):
        """Load the helmet detection models"""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            print(f"Loading models from: {models_dir}")
            
            # Load models using helmet detector
            if self.helmet_detector.load_models(models_dir):
                self.model_loaded = True
                print("✓ Helmet detection system loaded successfully!")
            else:
                print("✗ Failed to load helmet detection models!")
                self.model_loaded = False
                
        except Exception as e:
            print(f"Error loading helmet detection system: {str(e)}")
            self.model_loaded = False
        
    def setup_gui(self):
        """Setup the main GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Motorcycle Helmet Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        self.video_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display label
        self.video_label = ttk.Label(self.video_frame, text="Camera feed will appear here")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(control_frame, text="Start Detection", 
                                   command=self.start_detection)
        self.start_btn.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Detection", 
                                  command=self.stop_detection, state="disabled")
        self.stop_btn.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=2, column=0, pady=(20, 5))
        
        # Detection info
        self.detection_info = ttk.Label(control_frame, text="Helmets: 0 | No Helmets: 0")
        self.detection_info.grid(row=3, column=0, pady=5)
        
        # Additional detection info
        self.extra_info = ttk.Label(control_frame, text="Persons: 0 | Motorcycles: 0")
        self.extra_info.grid(row=4, column=0, pady=5)
        
        # Model status
        model_status = "Model: Loaded ✓" if self.model_loaded else "Model: Not loaded ✗"
        self.model_status_label = ttk.Label(control_frame, text=model_status)
        self.model_status_label.grid(row=5, column=0, pady=5)
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=6, column=0, pady=(20, 0))
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, 
                                         variable=self.confidence_var, orient="horizontal")
        self.confidence_scale.grid(row=7, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Confidence value display
        self.confidence_label = ttk.Label(control_frame, text="0.5")
        self.confidence_label.grid(row=8, column=0)
        
        # Update confidence display
        self.confidence_var.trace('w', self.update_confidence_display)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def update_confidence_display(self, *args):
        """Update confidence threshold display"""
        confidence = self.confidence_var.get()
        self.confidence_label.config(text=f"{confidence:.1f}")
    
    def start_detection(self):
        """Start the helmet detection"""
        if not self.model_loaded:
            messagebox.showerror("Error", "YOLO model not loaded. Please restart the application.")
            return
            
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
                
            self.is_detecting = True
            self.detection_count = 0
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_label.config(text="Status: Detecting...")
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def stop_detection(self):
        """Stop the helmet detection"""
        self.is_detecting = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Status: Stopped")
        self.video_label.config(image="", text="Camera feed will appear here")
        
    def detection_loop(self):
        """Main detection loop with helmet detection"""
        while self.is_detecting:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            try:
                # Run helmet detection
                confidence_threshold = self.confidence_var.get()
                detected_frame, detections = self.helmet_detector.detect_helmets_in_frame(
                    frame, confidence_threshold)
                
                # Update detection counts
                self.root.after(0, self.update_detection_counts, detections)
                
            except Exception as e:
                print(f"Detection error: {e}")
                detected_frame = frame
                detections = {'helmets': 0, 'no_helmets': 0, 'persons': 0, 'motorcycles': 0}
                
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update GUI in main thread
            self.root.after(0, self.update_video_display, photo)
            
            time.sleep(0.03)  # ~30 FPS
    
    def update_detection_counts(self, detections):
        """Update detection counts in GUI"""
        helmet_text = f"Helmets: {detections['helmets']} | No Helmets: {detections['no_helmets']}"
        extra_text = f"Persons: {detections['persons']} | Motorcycles: {detections['motorcycles']}"
        
        self.detection_info.config(text=helmet_text)
        self.extra_info.config(text=extra_text)
            
    def update_video_display(self, photo):
        """Update the video display in GUI"""
        self.video_label.config(image=photo, text="")
        self.video_label.photo = photo  # Keep a reference
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    print("Starting Motorcycle Helmet Detection System...")
    print("Loading YOLO model, please wait...")
    
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    if app.model_loaded:
        print("✓ Model loaded successfully!")
        print("✓ Ready for helmet detection!")
    else:
        print("✗ Model loading failed!")
        
    root.mainloop()