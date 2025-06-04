# ocr/license_plate.py

import cv2
import numpy as np
import logging
from speed_estimation.Number_plate_detection import detect_license_plate_and_number as detect_plate
import os

logger = logging.getLogger(__name__)

def save_vehicle_photo(frame, bbox, tracker_id, photo_folder, frame_idx):
    """
    Save a cropped photo of a vehicle.
    
    Args:
        frame: The video frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        tracker_id: ID of the tracker
        photo_folder: Directory to save photos
        frame_idx: Current frame index
        
    Returns:
        str: Path to saved photo
    """
    x1, y1, x2, y2 = bbox
    x1_int = np.around(x1).astype(int)
    y1_int = np.around(y1).astype(int)
    x2_int = np.around(x2).astype(int)
    y2_int = np.around(y2).astype(int)

    vehicle_crop = frame[y1_int:y2_int, x1_int:x2_int]
    photo_path = os.path.join(photo_folder, f"tracker_{tracker_id}_frame_{frame_idx}.jpg")
    cv2.imwrite(photo_path, vehicle_crop)
    return photo_path

def detect_license_plate_and_number(frame):
    """
    Process vehicle image to detect and read license plate
    
    Args:
        frame: Vehicle image frame
        
    Returns:
        tuple: (license_plate_text, bbox) where bbox is (x1, y1, x2, y2) or None if no plate detected
    """
    try:
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to license plate detection")
            return None, None
            
        # Log frame dimensions and check minimum size
        height, width = frame.shape[:2]
        logger.debug(f"Processing frame of size {width}x{height}")
        
        if width < 100 or height < 100:
            logger.warning(f"Frame too small for reliable detection: {width}x{height}")
            return None, None
            
        # Create different versions of the image for detection
        processed_frames = []
        
        # Original frame
        processed_frames.append(frame)
        
        # Enhanced contrast version
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        processed_frames.append(enhanced_color)
        
        # Sharpened version
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        processed_frames.append(sharpened)
        
        # Motion deblur version
        kernel_motion_blur = np.zeros((7, 7))
        kernel_motion_blur[3, :] = np.ones(7)
        kernel_motion_blur = kernel_motion_blur / 7
        motion_deblurred = cv2.filter2D(frame, -1, kernel_motion_blur)
        processed_frames.append(motion_deblurred)
        
        # Denoised version
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        processed_frames.append(denoised)
        
        # Edge enhanced version
        edges = cv2.Canny(gray, 100, 200)
        edge_enhanced = cv2.addWeighted(frame, 0.8, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.2, 0)
        processed_frames.append(edge_enhanced)
        
        # Try detection on all processed frames
        best_result = None
        best_confidence = 0
        
        for idx, proc_frame in enumerate(processed_frames):
            try:
                logger.debug(f"Attempting detection on processed frame {idx}")
                result = detect_plate(proc_frame)
                
                if result and len(result) == 2:
                    license_plate, plate_bbox = result
                    
                    if license_plate and isinstance(license_plate, str):
                        # Clean and validate plate text
                        plate_text = ''.join(c for c in license_plate.strip().upper() if c.isalnum())
                        
                        if len(plate_text) >= 3:  # Minimum 3 alphanumeric characters
                            # Calculate confidence based on text length and character types
                            confidence = len(plate_text) / 8.0  # Assume max plate length is 8
                            alpha_count = sum(c.isalpha() for c in plate_text)
                            num_count = sum(c.isdigit() for c in plate_text)
                            
                            if alpha_count > 0 and num_count > 0:  # Most plates have both letters and numbers
                                confidence += 0.3
                                
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = (plate_text, plate_bbox)
                                logger.debug(f"New best result: {plate_text} (confidence: {confidence:.2f})")
            
            except Exception as e:
                logger.debug(f"Error processing frame {idx}: {str(e)}")
                continue
        
        if best_result:
            logger.info(f"Best detected license plate: {best_result[0]} (confidence: {best_confidence:.2f})")
            return best_result
        else:
            logger.debug("No valid license plate detected in any processed frame")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in license plate detection: {str(e)}")
        logger.exception("Full traceback:")
        return None, None
