import cv2
import numpy as np
from speed_estimation.camera.video_camera import VideoCamera

def process_video(video_path="speed_estimation/Test_video/v3.mp4"):
    """
    Main function to process video stream
    
    Args:
        video_path: Path to video file
    """
    try:
        # Initialize video camera with processing pipeline
        print("Initializing video camera...")
        camera = VideoCamera(video_path)
        
        print("Starting video processing...")
        while True:
            # Get processed frame
            frame_data = camera.get_frame()
            
            if frame_data is None:
                print("Video processing completed")
                break
                
            # Display frame (optional, for debugging)
            frame = cv2.imdecode(
                np.frombuffer(frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            cv2.imshow('Speed Detection', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
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
    camera = VideoCamera()
    
    while True:
        frame_data = camera.get_frame()
        if frame_data is None:
            break
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

if __name__ == "__main__":
    process_video() 