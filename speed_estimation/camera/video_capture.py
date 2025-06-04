import cv2
import os

class VideoCapture:
    def __init__(self, video_path="speed_estimation/Test_video/test.mp4"):
        """Initialize video capture"""
        try:
            print(f"Checking video path: {os.path.abspath(video_path)}")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            print("Opening video file...")
            self.video = cv2.VideoCapture(video_path)
            if not self.video.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Initialize video properties
            self._init_properties()
            
            print(f"Video capture initialized successfully:")
            print(f"- Video path: {os.path.abspath(video_path)}")
            print(f"- Video properties: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            print(f"- Total frames: {self.frame_count}")
            
        except Exception as e:
            print(f"Error initializing VideoCapture: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _init_properties(self):
        """Initialize video properties"""
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'video'):
            self.video.release()

    def read_frame(self):
        """Read a single frame from the video stream"""
        try:
            success, frame = self.video.read()
            print(f"Frame {self.frame_idx}: Read success = {success}")
            
            if not success:
                print(f"Failed to read frame {self.frame_idx}, resetting video")
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                success, frame = self.video.read()
                if not success:
                    print("Failed to read frame after reset")
                    return None

            self.frame_idx += 1
            return frame
            
        except Exception as e:
            print(f"Error in read_frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_video_info(self):
        """Get video properties"""
        return {
            'fps': self.fps,
            'width': self.frame_width,
            'height': self.frame_height,
            'frame_count': self.frame_count,
            'current_frame': self.frame_idx
        } 