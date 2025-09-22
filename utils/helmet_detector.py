"""
Helmet Detection Utilities
Functions to add helmet-specific detection capabilities
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class HelmetDetector:
    def __init__(self):
        self.helmet_model = None
        self.general_model = None
        self.helmet_model_loaded = False
        
    def load_models(self, models_dir):
        """Load both general YOLO and helmet-specific models"""
        try:
            # Load general YOLO model
            general_model_path = os.path.join(models_dir, "yolov8n.pt")
            if os.path.exists(general_model_path):
                self.general_model = YOLO(general_model_path)
                print("✓ General YOLO model loaded")
            
            # Try to load helmet-specific model (we'll create this)
            helmet_model_path = os.path.join(models_dir, "helmet_yolov8n.pt")
            if os.path.exists(helmet_model_path):
                self.helmet_model = YOLO(helmet_model_path)
                self.helmet_model_loaded = True
                print("✓ Helmet-specific model loaded")
            else:
                print("⚠ Helmet-specific model not found, using general detection with helmet classification")
                self.helmet_model_loaded = False
                
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
        return True
    
    def detect_helmets_in_frame(self, frame, confidence_threshold=0.5):
        """
        Detect helmets in a frame using various methods
        Returns: processed_frame, detection_results
        """
        if self.helmet_model_loaded and self.helmet_model:
            # Use dedicated helmet model if available
            return self.detect_with_helmet_model(frame, confidence_threshold)
        else:
            # Use general model with helmet classification logic
            return self.detect_with_classification(frame, confidence_threshold)
    
    def detect_with_helmet_model(self, frame, confidence_threshold):
        """Use dedicated helmet detection model"""
        results = self.helmet_model(frame, conf=confidence_threshold)
        processed_frame, detections = self.process_helmet_results(frame, results)
        return processed_frame, detections
    
    def detect_with_classification(self, frame, confidence_threshold):
        """Use general YOLO + helmet classification logic"""
        results = self.general_model(frame, conf=confidence_threshold)
        processed_frame, detections = self.process_general_results_for_helmets(frame, results)
        return processed_frame, detections
    
    def process_helmet_results(self, frame, results):
        """Process results from helmet-specific model"""
        detections = {
            'helmets': 0,
            'no_helmets': 0,
            'persons': 0,
            'motorcycles': 0
        }
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Helmet-specific classes (you'd need to define these based on your model)
                    if cls == 0:  # helmet
                        detections['helmets'] += 1
                        color = (0, 255, 0)  # Green for helmet
                        label = f'Helmet {conf:.2f}'
                    elif cls == 1:  # no helmet
                        detections['no_helmets'] += 1
                        color = (0, 0, 255)  # Red for no helmet
                        label = f'No Helmet {conf:.2f}'
                    else:
                        color = (255, 0, 0)  # Blue for others
                        label = f'Object {conf:.2f}'
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, detections
    
    def process_general_results_for_helmets(self, frame, results):
        """Process general YOLO results and apply helmet detection logic"""
        detections = {
            'helmets': 0,
            'no_helmets': 0,
            'persons': 0,
            'motorcycles': 0
        }
        
        person_boxes = []
        motorcycle_boxes = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # COCO class names
                    class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle'}
                    
                    if cls == 0:  # person
                        detections['persons'] += 1
                        person_boxes.append((x1, y1, x2, y2, conf))
                        
                        # Apply helmet detection logic to person's head area
                        helmet_detected = self.classify_helmet_on_person(frame, x1, y1, x2, y2)
                        
                        if helmet_detected:
                            detections['helmets'] += 1
                            color = (0, 255, 0)  # Green for helmet
                            label = f'Person+Helmet {conf:.2f}'
                        else:
                            detections['no_helmets'] += 1
                            color = (0, 0, 255)  # Red for no helmet
                            label = f'Person-NoHelmet {conf:.2f}'
                            
                    elif cls == 3:  # motorcycle
                        detections['motorcycles'] += 1
                        motorcycle_boxes.append((x1, y1, x2, y2, conf))
                        color = (255, 255, 0)  # Cyan for motorcycle
                        label = f'Motorcycle {conf:.2f}'
                    else:
                        color = (255, 0, 0)  # Blue for other objects
                        label = f'{class_names.get(cls, "Object")} {conf:.2f}'
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, detections
    
    def classify_helmet_on_person(self, frame, x1, y1, x2, y2):
        """
        Fine-tuned helmet classification - balanced accuracy
        """
        # Extract head region (top 32% of person bounding box)
        head_height = int((y2 - y1) * 0.32)
        head_region = frame[int(y1):int(y1 + head_height), int(x1):int(x2)]
        
        if head_region.size == 0:
            return False
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Optimized helmet color ranges (include transparent visors and reflections)
        helmet_colors = [
            # White/Light helmets
            ([0, 0, 180], [180, 40, 255]),
            # Black helmets
            ([0, 0, 0], [180, 80, 80]),
            # Red helmets (like yours - expanded for visor reflections)
            ([0, 50, 50], [15, 255, 255]),  # Lower red - more inclusive
            ([165, 50, 50], [180, 255, 255]),  # Upper red - more inclusive
            # Blue helmets
            ([100, 50, 80], [130, 255, 255]),
            # Yellow/Orange helmets
            ([15, 60, 100], [35, 255, 255]),
            # Green helmets
            ([50, 60, 80], [70, 255, 255]),
            # Silver/metallic helmets and visor reflections
            ([0, 0, 120], [180, 30, 255]),
            # Dark colored helmets (matte finish)
            ([0, 20, 40], [180, 100, 120]),
            # Transparent/clear visor areas (low saturation, medium-high brightness)
            ([0, 0, 100], [180, 50, 200]),
        ]
        
        total_pixels = head_region.shape[0] * head_region.shape[1]
        helmet_pixels = 0
        max_color_match = 0
        
        # Check each color range and find the best match
        for lower, upper in helmet_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            color_pixels = cv2.countNonZero(mask)
            helmet_pixels += color_pixels
            color_ratio = color_pixels / total_pixels
            max_color_match = max(max_color_match, color_ratio)
        
        # Total color ratio and best single color match
        total_color_ratio = helmet_pixels / total_pixels
        
    # Shape and texture analysis
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        
        # Check for helmet-like smooth texture
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_smooth_texture = laplacian_var < 500  # Helmets are smoother than hair
        
        # Check brightness distribution (helmets are more uniform)
        brightness_std = np.std(gray)
        is_uniform_brightness = brightness_std < 35
        
        # Edge-based shape analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        
        # Look for helmet-like curved shapes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_helmet_shape = False
        largest_contour_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_contour_area:
                largest_contour_area = area
            
            if area > 150:  # Reasonable minimum area
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    aspect_ratio = cv2.boundingRect(contour)[2] / cv2.boundingRect(contour)[3]
                    
                    # Helmet characteristics: reasonably circular, proper aspect ratio
                    if (circularity > 0.35 and 0.7 < aspect_ratio < 1.4 and area > 300):
                        has_helmet_shape = True
                        break
        
        # Connected helmet-colored blob analysis (require contiguous region)
        # Create combined helmet color mask to find contiguous regions
        combined_mask = None
        for lower, upper in helmet_colors:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            m = cv2.inRange(hsv, lower, upper)
            if combined_mask is None:
                combined_mask = m
            else:
                combined_mask = cv2.bitwise_or(combined_mask, m)

        # Morphological clean-up
        if combined_mask is None:
            combined_mask = np.zeros((head_region.shape[0], head_region.shape[1]), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find largest helmet-colored contour
        helmet_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_helmet_area = 0
        helmet_centroid = None
        for hc in helmet_contours:
            a = cv2.contourArea(hc)
            if a > largest_helmet_area:
                largest_helmet_area = a
                M = cv2.moments(hc)
                if M['m00'] != 0:
                    helmet_centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

        # Normalize helmet area ratio
        helmet_area_ratio = largest_helmet_area / (total_pixels + 1e-6)

        # Face detection to validate helmet position (use Haar cascade)
        face_above_blob = False
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0 and helmet_centroid is not None:
                # check whether helmet centroid is above or overlapping face bounding box
                for (fx, fy, fw, fh) in faces:
                    # face coordinates are relative to head_region
                    cx, cy = helmet_centroid
                    if cy < fy + int(fh * 0.6):
                        face_above_blob = True
                        break
        except Exception:
            face_above_blob = False

        # Balanced detection criteria (catch real helmets, avoid false positives)
        criteria_met = 0
        
        # Criterion 1: Reasonable color match for helmets with visors
        if max_color_match > 0.25 or total_color_ratio > 0.30:
            criteria_met += 1
        
        # Criterion 2: Surface analysis (helmets can have visors = mixed texture)
        if is_smooth_texture or is_uniform_brightness:  # Either condition is enough
            criteria_met += 1
        
        # Criterion 3: Shape detection
        if has_helmet_shape:
            criteria_met += 1
        
        # Criterion 4: Moderate color evidence with shape
        if total_color_ratio > 0.20 and largest_contour_area > 300:
            criteria_met += 1
        
        # Decision: Need at least 2 out of 4 criteria (balanced approach)
        helmet_detected = criteria_met >= 2
        
        # Special case: Strong color evidence (helmets with good color match)
        if max_color_match > 0.35 and total_color_ratio > 0.25:
            helmet_detected = True
        
        # Helmet with visor detection: good color + reasonable size
        if (max_color_match > 0.20 or total_color_ratio > 0.25) and largest_contour_area > 250:
            helmet_detected = True
        
        # Still reject very weak evidence (hair/background)
        if max_color_match < 0.12 and total_color_ratio < 0.15:
            helmet_detected = False

        # Require contiguous helmet-colored region and face alignment for reliable detection
        # Minimum helmet blob area ratio and face overlap check
        if largest_helmet_area > 0:
            if helmet_area_ratio < 0.03:  # less than 3% of head region
                helmet_detected = False
            elif not face_above_blob and helmet_centroid is not None:
                # If the color blob is not positioned above/overlapping detected face, more likely background
                helmet_detected = False
        
        # Don't reject based on texture alone (visors can create mixed textures)
        # Remove the hair texture rejection for now
        
        # Debug output for fine-tuning
        if helmet_detected:
            print(f"✅ HELMET: Criteria: {criteria_met}/4, Color:{max_color_match:.2f} Texture:{laplacian_var:.0f} Smooth:{is_smooth_texture}")
        else:
            print(f"❌ NO HELMET: Criteria: {criteria_met}/4, Color:{max_color_match:.2f} Total:{total_color_ratio:.2f} Texture:{laplacian_var:.0f}")
        
        return helmet_detected