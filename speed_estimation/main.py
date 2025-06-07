import cv2
import numpy as np
import supervision as sv
# from speed_estimation.camera.video_processor import VideoProcessor
from speed_estimation.camera.get_frame import get_frames
from speed_estimation.utils.detector import VehicleDetector
from speed_estimation.utils.speed import calculate_speed
from speed_estimation.tracking.tracker import VehicleTracker
from speed_estimation.ocr.license_plate import detect_license_plate_and_number
from speed_estimation.utils.video_utils import draw_bounding_box, get_center
from speed_estimation.utils.speed_estimator import SpeedEstimator
id_mapping = {}
next_id = 0

detected=[]
src_points = np.float32([(333, 676), (1224, 661), (980, 294), (765, 294)])
dst_points = np.float32([(0, 0), (600, 0), (600, 800), (0, 800)])
detector=VehicleDetector("models/yolov8m.pt")
# license_plate_detector=LicensePlateDetector("models/license_plate_detector.pt")
TARGET_CLASSES = [2, 3, 5, 7]
def process_video(video_path="speed_estimation/Test_video/car1.mp4"):
    global id_mapping,next_id
    print("Something")
    try:
        frame_gen=get_frames(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        tracker=VehicleTracker(fps)
        speed_estimator = SpeedEstimator(src_points, dst_points, ppm=8.2, fps=30)
        for frame in frame_gen:
            results=detector.detect(frame)
            
            # Manually extract bounding boxes, confidences, class ids
            xyxy = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            detections = sv.Detections(
    xyxy=xyxy,
    confidence=confidences,
    class_id=class_ids
)
            mask = np.array([cls in TARGET_CLASSES for cls in detections.class_id])
            # Apply the mask to filter detections
            detections = detections[mask]

            frame_resolution = (frame.shape[1], frame.shape[0]) 
            tracked = tracker.update(detections,frame_resolution=frame_resolution)
            
            for det in tracked:
                    id = det.track_id
                    
                    if id not in id_mapping:
                        id_mapping[id] = next_id
                        next_id += 1

                    display_id = id_mapping[id]  # use this ID for drawing
                    bbox = det.tlwh
                    center = get_center(bbox)
                    speed = speed_estimator.estimate_speed(id, center)
                    color = (0, 0, 255) if speed > 30 else (0, 255, 0)
                    draw_bounding_box(frame, bbox, color, f"#{display_id}:{speed:.2f} km/h")
                    # print(det)
                    print(f"Vehicle{display_id} Speed:{speed}")
                    if speed>30 and display_id not in detected:
                        print("Crossed the limit 🥹")
                        detected.append(display_id)
                        x, y, w, h = bbox
                        vehicle_crop = frame[int(y):int(y + h), int(x):int(x + w)]
                        if vehicle_crop is not None and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                            _,plates=detect_license_plate_and_number(vehicle_crop)
                            cv2.imshow("Plates", plates)
                    

            cv2.imshow('Speed Detection', frame)
            
                # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error in video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

def process_video_stream():
    """Generator function for web streaming"""
    
    camera=cv2.VideoCapture(0)
    while True:
        frame_data = camera.get_frame()
        if frame_data is None:
            break
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


# process_video("speed_estimation/Test_video/car1.mp4")
if __name__ == "__main__":
    process_video("speed_estimation/Test_video/car1.mp4") 