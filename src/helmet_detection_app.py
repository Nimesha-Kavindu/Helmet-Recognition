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

class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motorcycle Helmet Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.cap = None
        self.is_detecting = False
        self.model = None
        
        # Create GUI elements
        self.setup_gui()
        
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
        self.detection_info = ttk.Label(control_frame, text="Helmets detected: 0")
        self.detection_info.grid(row=3, column=0, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def start_detection(self):
        """Start the helmet detection"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
                
            self.is_detecting = True
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
        """Main detection loop"""
        while self.is_detecting:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # TODO: Add helmet detection here
            # For now, just display the frame
            
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update GUI in main thread
            self.root.after(0, self.update_video_display, photo)
            
            time.sleep(0.03)  # ~30 FPS
            
    def update_video_display(self, photo):
        """Update the video display in GUI"""
        self.video_label.config(image=photo, text="")
        self.video_label.photo = photo  # Keep a reference
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()