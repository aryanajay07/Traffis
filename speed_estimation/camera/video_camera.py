from .video_processor import VideoProcessor

class VideoCamera:
    def __init__(self, video_path="speed_estimation/Test_video/v2.mp4"):
        """Initialize video processor"""
        self.processor = VideoProcessor(video_path)

    def get_frame(self):
        """Get processed frame from video processor"""
        return self.processor.get_frame()

    def get_video_info(self):
        """Get video properties"""
        return self.processor.video_capture.get_video_info()

    def get_stats(self):
        """Get current tracking statistics"""
        return {
            'tracker_id_to_coordinates': self.processor.tracker_id_to_coordinates,
            'max_speeds': self.processor.max_speeds,
            'saved_photos': self.processor.saved_photos,
            'license_detected': self.processor.license_detected
        }

    def __del__(self):
        """Clean up resources"""
        del self.processor
