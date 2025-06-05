import cv2
import os
import easyocr
import logging
from ultralytics import YOLO
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the files and folders
model_path = 'models/license_detection_model.pt'
output_folder = './Plate_photo'

# Load the YOLOv8 model with error handling
try:
    model = YOLO(model_path)
    logging.info("YOLO model loaded successfully.")
    # print("Class names:", model.names)
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    raise

# Initialize EasyOCR reader for both Nepali (Devnagari script) and English
try:
    reader = easyocr.Reader(['en','ne','hi'], gpu=True)
except:
    logging.warning("GPU not available for EasyOCR. Using CPU.")
    reader = easyocr.Reader(['en','ne','hi',], gpu=False)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def preprocess_image_for_ocr(image):
    """
    Preprocess image for OCR using OpenCV instead of PIL
    """
    if image is None:
        return None
        
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)

        # Convert back to RGB for OCR
        rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        
        return rgb
    except Exception as e:
        logging.error(f"Error in image preprocessing: {e}")
        return None

def detect_license_plate_and_number(image_input):
    print("Inside plate detection")

    # Handle both file paths and image arrays
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input

    plates = []
    plate_numbers = []

    if image is None:
        logging.error("Error: Invalid image input")
        return None, None

    results = model(image, show=True)
    if len(results) == 0 or not hasattr(results[0], 'boxes'):
        logging.warning("YOLO returned no detections.")
        return None, None
    boxes = results[0].boxes

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = box.conf.item()
        class_id = int(box.cls.item())
      
        if class_id == 0 and confidence > 0.3:
            # Crop plate region
            plate = image[y1:y2, x1:x2]
            plates.append(plate)
            print("Before preprocessing:Image ")
            
            # Save cropped plate image
            cv2.imwrite(os.path.join(output_folder, f"plate_{len(plates)}.jpg"), plate)
            
            try:
                # Initial resize
                plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # Preprocess using our custom function
                processed_plate = preprocess_image_for_ocr(plate)
                
                if processed_plate is not None:
                    # OCR
                    ocr_result = reader.readtext(processed_plate)
                    print(f"OCR raw result for plate {i}: {ocr_result}")

                    if ocr_result:
                        text = max(ocr_result, key=lambda x: x[2])[1]
                        cleaned = text.replace(' ', '').replace('\n', '').strip()
                        plate_numbers.append(cleaned)
                    else:
                        plate_numbers.append(None)
                else:
                    plate_numbers.append(None)
                    
            except Exception as e:
                logging.warning(f"OCR Error: {e}")
                plate_numbers.append(None)

    if not plates:
        logging.info("No license plates detected.")
        return None, None

    return plates, plate_numbers

# Example usage:
# plates, numbers = detect_license_plate_and_number("license-plates-5/test/images/b1a50a3824887ee2_jpg.rf.68a4fd34fce20184287592f2680f895b.jpg")
# if  plates: 
#     for i, num in enumerate(numbers):
#         print(f"Plate {i+1}: {num}")
#         cv2.imshow(f"Plate {i+1}", plates[i])
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
