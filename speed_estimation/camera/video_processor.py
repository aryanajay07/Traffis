import cv2
import numpy as np
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from ultralytics import YOLO
import supervision as sv
import logging

from speed_estimation.tracking.tracker import get_tracker, BYTETracker
from speed_estimation.utils.transform import ViewTransformer, SOURCE, TARGET
from speed_estimation.utils.speed import calculate_speed
from speed_estimation.ocr.license_plate import detect_license_plate_and_number
from user_app.models import Record, Station
from .video_capture import VideoCapture

# Set up logging
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, video_path="speed_estimation/Test_video/test.mp4"):
        """Initialize video processor with capture and processing components"""
        try:
            # Initialize video capture
            self.video_capture = VideoCapture(video_path)
            
            # Load YOLO model
            model_path = "models/yolov8m.pt"
            logger.info(f"Checking YOLO model path: {os.path.abspath(model_path)}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
            
            logger.info("Loading YOLO model...")
            self.model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
            
            # Initialize processing components
            self._init_components()
            
        except Exception as e:
            logger.error(f"Error initializing VideoProcessor: {str(e)}")
            raise

    def _init_components(self):
        """Initialize processing components and parameters"""
        video_info = self.video_capture.get_video_info()
        self.fps = video_info['fps']
        self.frame_width = video_info['width']
        self.frame_height = video_info['height']
        
        # Processing components
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        self.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        self.byte_tracker = BYTETracker(get_tracker(), frame_rate=self.fps)
        
        # State tracking
        self.tracker_id_to_coordinates = defaultdict(lambda: deque(maxlen=30))
        self.saved_photos = set()
        self.license_detected = defaultdict(lambda: False)
        self.max_speeds = defaultdict(lambda: 0)
        self.detection_attempts = defaultdict(int)
        self.speed_violation_frames = {}
        self.vehicle_data = defaultdict(dict)
        self.vehicle_classes = defaultdict(int)  # Store class IDs for each tracker
        
        # Configuration
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.VEHICLE_CLASS_NAMES = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        self.SPEED_LIMIT = 5.0  # km/h
        self.MIN_FRAMES_FOR_SPEED = 5
        self.MAX_DETECTION_ATTEMPTS = 3
        self.HIGH_SPEED_THRESHOLD = 150.0  # km/h
        self.MAC_ADDRESS = '8c:aa:ce:51:67:e9'
        self.PHOTO_DIR = "Captured_Photos/Speeding_Vehicles"
        self.VIOLATIONS_DIR = os.path.join(self.PHOTO_DIR, "Initial_Violations")
        self.PLATES_DIR = os.path.join(self.PHOTO_DIR, "License_Plates")
        os.makedirs(self.VIOLATIONS_DIR, exist_ok=True)
        os.makedirs(self.PLATES_DIR, exist_ok=True)

    def get_frame(self):
        """Process a single frame from the video stream"""
        # Read frame from capture
        frame = self.video_capture.read_frame()
        if frame is None:
            return None

        # Process frame
        frame = self._process_frame(frame)
        if frame is None:
            return None

        # Encode frame
        frame = self._prepare_frame_for_display(frame)
        return frame

    def _process_frame(self, frame):
        """Process frame with object detection and tracking"""
        if frame is None or frame.size == 0:
            print(f"Empty frame received")
            return None

        # Run YOLO detection
        print("Processing frame with YOLO")
        results = self.model(frame, imgsz=1280)[0]
        detections = self._filter_vehicle_detections(results)
        
        if len(detections) == 0:
            return frame

        # Process detections with ByteTrack
        dets = np.hstack((detections.xyxy, detections.confidence.reshape(-1, 1)))
        online_targets = self.byte_tracker.update(
            dets, 
            [self.frame_height, self.frame_width],
            [self.frame_height, self.frame_width]
        )

        # Update vehicle classes from detections
        for det, cls in zip(detections.xyxy, detections.class_id):
            # Find the closest tracker to this detection
            for target in online_targets:
                t_box = target.tlbr
                if self._box_iou(det, t_box) > 0.5:  # If IOU > 0.5, consider it a match
                    self.vehicle_classes[int(target.track_id)] = int(cls)
                    break

        # Process each tracked vehicle
        for target in online_targets:
            frame = self._process_vehicle(frame, target)

        # Draw annotations
        frame = self._draw_annotations(frame, online_targets)
        return frame

    def _filter_vehicle_detections(self, results):
        """Filter YOLO detections for vehicles only"""
        boxes = results.boxes
        class_id = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        
        vehicle_mask = np.isin(class_id, self.VEHICLE_CLASSES)
        return sv.Detections(
            xyxy=xyxy[vehicle_mask],
            confidence=confidences[vehicle_mask],
            class_id=class_id[vehicle_mask]
        )

    def _box_iou(self, box1, box2):
        """Calculate IOU between two boxes"""
        # Convert to x1,y1,x2,y2 format if needed
        box1 = np.array(box1).reshape(-1, 4)
        box2 = np.array(box2).reshape(-1, 4)
        
        # Calculate intersection
        x1 = np.maximum(box1[:, 0], box2[:, 0])
        y1 = np.maximum(box1[:, 1], box2[:, 1])
        x2 = np.minimum(box1[:, 2], box2[:, 2])
        y2 = np.minimum(box1[:, 3], box2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)

    def _process_vehicle(self, frame, target):
        """Process a single tracked vehicle"""
        x1, y1, x2, y2 = target.tlbr
        tracker_id = int(target.track_id)
        
        # Calculate transformed point and speed
        center_bottom = np.array([[(x1 + x2) / 2, y2]])
        transformed_point = self.view_transformer.transform_points(center_bottom)[0]
        self.tracker_id_to_coordinates[tracker_id].append(transformed_point)
        
        # Log coordinate history
        coords = self.tracker_id_to_coordinates[tracker_id]
        logger.debug(f"Vehicle {tracker_id} coordinate history: {list(coords)}")
        logger.debug(f"Vehicle {tracker_id} has {len(coords)} tracked points")
        
        speed = calculate_speed(list(coords), self.fps)
        logger.info(f"Vehicle {tracker_id}: Speed = {speed:.1f} km/h (FPS: {self.fps})")
        
        # Update max speed and vehicle data
        if speed > self.max_speeds[tracker_id]:
            self.max_speeds[tracker_id] = speed
            logger.info(f"Vehicle {tracker_id} new max speed: {speed:.1f} km/h")
            
            # Get vehicle type from stored class ID
            vehicle_type = self.VEHICLE_CLASS_NAMES.get(
                self.vehicle_classes.get(tracker_id, -1), 
                'unknown'
            )
            
            # Update vehicle data
            self.vehicle_data[tracker_id].update({
                'max_speed': speed,
                'detection_time': datetime.now(),
                'vehicle_type': vehicle_type
            })
        
        # Process speeding vehicles
        if speed > self.SPEED_LIMIT:
            logger.warning(f"Speed Limit Breached! Vehicle {tracker_id} at {speed:.1f} km/h")
            
            # Save initial violation frame if not already saved
            if tracker_id not in self.saved_photos:
                # Get the vehicle crop with padding
                x1, y1, x2, y2 = map(int, target.tlbr)
                width = x2 - x1
                height = y2 - y1
                
                # Add padding
                pad_x = int(width * 0.2)
                pad_y = int(height * 0.2)
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(frame.shape[1], x2 + pad_x)
                crop_y2 = min(frame.shape[0], y2 + pad_y)
                
                # Save clean images without any overlays
                clean_frame = frame.copy()
                violation_img = clean_frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                # Save images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_path = os.path.join(self.VIOLATIONS_DIR, 
                                       f"full_violation_{timestamp}_{tracker_id}_{speed:.0f}kmh.jpg")
                crop_path = os.path.join(self.VIOLATIONS_DIR, 
                                       f"crop_violation_{timestamp}_{tracker_id}_{speed:.0f}kmh.jpg")
                
                cv2.imwrite(full_path, clean_frame)
                cv2.imwrite(crop_path, violation_img)
                logger.info(f"Saved clean violation photos to {full_path} and {crop_path}")
                
                # Update vehicle data with image paths
                self.vehicle_data[tracker_id].update({
                    'full_image_path': full_path,
                    'crop_image_path': crop_path,
                    'violation_speed': speed,
                    'violation_time': datetime.now(),
                    'location': 'Location Name',  # You should set this based on your station
                    'direction': 'North',  # You should set this based on your camera setup
                })
                
                # Store the cropped image for license plate detection
                self.speed_violation_frames[tracker_id] = violation_img
                self.saved_photos.add(tracker_id)
                
                # Save to database immediately
                self._save_violation_to_database(tracker_id)
            
            # Now proceed with license plate detection
            self._handle_speeding_vehicle(frame, tracker_id, speed, target.tlbr)
        
        # Draw vehicle box and speed (only for display frame)
        color = (0, 165, 255) if speed > self.SPEED_LIMIT else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Add speed text with background
        speed_text = f"{speed:.1f} km/h"
        text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - text_size[1] - 5),
                     (int(x1) + text_size[0], int(y1)),
                     color, -1)
        cv2.putText(frame, speed_text,
                   (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame

    def _handle_speeding_vehicle(self, frame, tracker_id, speed, bbox):
        """Handle license plate detection for a speeding vehicle"""
        if not self.license_detected[tracker_id] and self.detection_attempts[tracker_id] < self.MAX_DETECTION_ATTEMPTS:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Use the stored violation frame if available
            vehicle_img = self.speed_violation_frames.get(tracker_id)
            if vehicle_img is None:
                # Fallback to current frame if stored frame not available
                width = x2 - x1
                height = y2 - y1
                speed_factor = min(speed / 100.0, 2.0)
                # Increase padding for high-speed vehicles
                pad_factor = 0.3 if speed > self.HIGH_SPEED_THRESHOLD else 0.2
                pad_x = int(width * pad_factor * speed_factor)
                pad_y = int(height * pad_factor * speed_factor)
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(frame.shape[1], x2 + pad_x)
                crop_y2 = min(frame.shape[0], y2 + pad_y)
                vehicle_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            # Resize if needed - increase minimum dimension for high-speed vehicles
            min_dim = 300 if speed > self.HIGH_SPEED_THRESHOLD else 200
            if vehicle_img.shape[0] < min_dim or vehicle_img.shape[1] < min_dim:
                scale = min_dim / min(vehicle_img.shape[0], vehicle_img.shape[1])
                new_width = int(vehicle_img.shape[1] * scale)
                new_height = int(vehicle_img.shape[0] * scale)
                vehicle_img = cv2.resize(vehicle_img, (new_width, new_height), 
                                      interpolation=cv2.INTER_CUBIC)
                logger.info(f"Resized vehicle frame to {new_width}x{new_height}")
            
            self.detection_attempts[tracker_id] += 1
            logger.info(f"Attempting license plate detection for vehicle {tracker_id} (Attempt {self.detection_attempts[tracker_id]}/{self.MAX_DETECTION_ATTEMPTS})")
            
            # Enhanced preprocessing for high-speed vehicles
            if speed > self.HIGH_SPEED_THRESHOLD:
                logger.info(f"High-speed vehicle detected ({speed:.1f} km/h). Applying enhanced preprocessing.")
                
                # Apply motion deblurring
                kernel_size = int(min(vehicle_img.shape[:2]) * 0.15)
                kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
                
                # Gaussian deblurring
                deblurred = cv2.GaussianBlur(vehicle_img, (kernel_size, kernel_size), 0)
                vehicle_img = cv2.addWeighted(vehicle_img, 1.5, deblurred, -0.5, 0)
                
                # Enhance contrast
                lab = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                vehicle_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Sharpen
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                vehicle_img = cv2.filter2D(vehicle_img, -1, kernel)
            
            license_plate, plate_bbox = detect_license_plate_and_number(vehicle_img)
            
            if license_plate and plate_bbox is not None:
                logger.info(f"License plate detected: {license_plate}")
                
                # Save the plate detection image without annotations
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_path = os.path.join(self.PLATES_DIR, 
                                        f"plate_{timestamp}_{tracker_id}_{license_plate}.jpg")
                
                # Save clean cropped image of the plate region
                x1, y1, x2, y2 = plate_bbox
                plate_img = vehicle_img[y1:y2, x1:x2].copy()
                cv2.imwrite(plate_path, plate_img)
                logger.info(f"Saved clean plate image to {plate_path}")
                
                # Update database record with license plate info
                self._update_violation_record(tracker_id, license_plate, plate_path)
                self.license_detected[tracker_id] = True
            else:
                logger.warning(f"No license plate detected for vehicle {tracker_id}")
            
            # Update visualization on main frame (only for display)
            color = (0, 0, 128) if speed > self.HIGH_SPEED_THRESHOLD else (0, 0, 255)
            if self.license_detected[tracker_id]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{license_plate} ({speed:.1f} km/h)"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color, -1)
                cv2.putText(frame, text,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (255, 255, 255), 2)
            else:
                color = (0, 69, 255) if speed > self.HIGH_SPEED_THRESHOLD else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"No Plate ({self.detection_attempts[tracker_id]}/{self.MAX_DETECTION_ATTEMPTS})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color, -1)
                cv2.putText(frame, text,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (255, 255, 255), 2)

    def _update_violation_record(self, tracker_id, license_plate, plate_path):
        """Update violation record with license plate information"""
        try:
            if 'violation_id' not in self.vehicle_data[tracker_id]:
                logger.error(f"No violation record found for vehicle {tracker_id}")
                return
                
            violation_id = self.vehicle_data[tracker_id]['violation_id']
            logger.info(f"Updating violation record {violation_id} with license plate {license_plate}")
            
            # Update the record
            Record.objects.filter(id=violation_id).update(
                licenseplate_no=license_plate,
                license_plate_image=plate_path,
                plate_detection_time=timezone.now()
            )
            
            logger.info(f"Successfully updated violation record {violation_id} with license plate information")
            
        except Exception as e:
            logger.error(f"Error updating violation record: {str(e)}")
            logger.exception("Full traceback:")

    def _save_to_database(self, speed, license_plate, photo_path):
        """Save violation record to database"""
        try:
            logger.info(f"Attempting to save violation record to database...")
            logger.debug(f"Looking up station with MAC: {self.MAC_ADDRESS}")
            station = Station.objects.get(mac_address=self.MAC_ADDRESS)
            
            record = Record.objects.create(
                station=station,
                license_plate=license_plate,
                speed=speed,
                photo_path=photo_path
            )
            logger.info(f"Successfully saved record to database: {license_plate}, {speed} km/h (Record ID: {record.id})")
        except Station.DoesNotExist:
            logger.error(f"Station not found with MAC address: {self.MAC_ADDRESS}")
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            logger.exception("Full traceback:")

    def _draw_annotations(self, frame, online_targets):
        """Draw annotations on the frame"""
        for target in online_targets:
            tracker_id = int(target.track_id)
            if tracker_id in self.max_speeds:
                max_speed = self.max_speeds[tracker_id]
                cv2.putText(frame, f"Max: {max_speed:.1f} km/h", 
                           (int(target.tlbr[0]), int(target.tlbr[1])-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return frame

    def _prepare_frame_for_display(self, frame):
        """Prepare frame for display (encode to JPEG)"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        except Exception as e:
            print(f"Error encoding frame: {str(e)}")
            return None

    def _save_violation_to_database(self, tracker_id):
        """Save vehicle violation data to database"""
        try:
            logger.info(f"Saving violation data for vehicle {tracker_id} to database")
            vehicle_info = self.vehicle_data[tracker_id]
            
            # Get station
            station = Station.objects.get(mac_address=self.MAC_ADDRESS)
            
            # Create violation record
            violation = Record.objects.create(
                stationID=station,
                vehicle_type=vehicle_info['vehicle_type'],
                speed=vehicle_info['violation_speed'],
                detection_time=vehicle_info['violation_time'],
                vehicle_image=vehicle_info['full_image_path'],
                license_plate_image=vehicle_info['crop_image_path'],
                max_speed=vehicle_info['max_speed'],
                location=vehicle_info['location'],
                direction=vehicle_info['direction']
            )
            
            # Store violation ID for later updates
            self.vehicle_data[tracker_id]['violation_id'] = violation.id
            logger.info(f"Successfully saved violation record for vehicle {tracker_id} (Record ID: {violation.id})")
            
        except Station.DoesNotExist:
            logger.error(f"Station not found with MAC address: {self.MAC_ADDRESS}")
        except Exception as e:
            logger.error(f"Error saving violation to database: {str(e)}")
            logger.exception("Full traceback:") 