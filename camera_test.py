"""
Simple Camera Test for Helmet Detection App
This script tests basic camera functionality before adding ML detection
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time

class CameraTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Test - Helmet Detection System")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.cap = None
        self.is_running = False
        
        # Create GUI elements
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the test GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Camera Test - Basic Functionality", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        self.video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display label
        self.video_label = ttk.Label(self.video_frame, text="Click 'Start Camera' to begin")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(control_frame, text="Start Camera", 
                                   command=self.start_camera)
        self.start_btn.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=2, column=0, pady=(20, 5))
        
        # Camera info
        self.camera_info = ttk.Label(control_frame, text="Camera: Not connected")
        self.camera_info.grid(row=3, column=0, pady=5)
        
        # Test different cameras button
        self.test_cameras_btn = ttk.Button(control_frame, text="Test Cameras", 
                                          command=self.test_available_cameras)
        self.test_cameras_btn.grid(row=4, column=0, pady=10, sticky=tk.W+tk.E)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def test_available_cameras(self):
        """Test which camera indices are available"""
        available_cameras = []
        
        # Test camera indices 0-3
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            message = f"Available cameras: {available_cameras}"
            self.camera_info.config(text=f"Found cameras: {available_cameras}")
        else:
            message = "No cameras found!"
            self.camera_info.config(text="No cameras detected")
            
        messagebox.showinfo("Camera Test Results", message)
        
    def start_camera(self):
        """Start the camera feed"""
        try:
            # Try camera index 0 first, then 1, then 2
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.camera_info.config(text=f"Camera {camera_index}: Connected")
                        break
                    else:
                        self.cap.release()
                else:
                    if self.cap:
                        self.cap.release()
            else:
                messagebox.showerror("Error", "Could not open any camera")
                return
                
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.status_label.config(text="Status: Starting camera...")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Status: Stopped")
        self.camera_info.config(text="Camera: Disconnected")
        self.video_label.config(image="", text="Click 'Start Camera' to begin")
        
    def camera_loop(self):
        """Main camera loop"""
        frame_count = 0
        
        # Enable stop button after camera starts
        self.root.after(100, lambda: self.stop_btn.config(state="normal"))
        self.root.after(100, lambda: self.status_label.config(text="Status: Camera running"))
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to read from camera"))
                break
                
            frame_count += 1
            
            # Add frame counter overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert frame for tkinter display
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
        if self.is_running:  # Only update if still running
            self.video_label.config(image=photo, text="")
            self.video_label.photo = photo  # Keep a reference
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    print("Starting Camera Test Application...")
    print("This will test basic camera functionality before adding helmet detection.")
    
    root = tk.Tk()
    app = CameraTestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()