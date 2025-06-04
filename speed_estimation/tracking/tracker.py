# tracking/tracker.py

import sys
import os
import argparse
import numpy as np
from collections import defaultdict, deque
import supervision as sv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ByteTrack')))
from yolox.tracker.byte_tracker import BYTETracker

def get_tracker():
    """Get ByteTrack configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--gate_thresh", type=float, default=0.15)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6)
    parser.add_argument("--min_box_area", type=float, default=10)
    parser.add_argument("--mot20", action='store_true')
    parser.add_argument("--track_type", type=str, default='bytetrack')
    return parser.parse_args([])

class VehicleTracker:
    def __init__(self, fps):
        self.args = get_tracker()
        self.tracker = BYTETracker(self.args, frame_rate=fps)
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.max_speeds = defaultdict(lambda: 0)
        
    def update(self, detections, frame_resolution):
        """
        Update tracker with new detections
        
        Args:
            detections: Detections from YOLO model
            frame_resolution: Tuple of (height, width)
            
        Returns:
            list: List of tracked objects
        """
        height, width = frame_resolution
        
        # Prepare detections for ByteTrack
        if len(detections) == 0:
            return []
            
        dets = np.hstack((detections.xyxy, detections.confidence.reshape(-1, 1)))
        online_targets = self.tracker.update(dets, [height, width], [height, width])
        
        return online_targets
        
    def annotate_frame(self, frame, online_targets):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Video frame
            online_targets: List of tracked objects
            
        Returns:
            ndarray: Annotated frame
        """
        # Prepare detections for annotation
        det_xyxy = np.array([target.tlbr for target in online_targets])
        if det_xyxy.size == 0:
            det_xyxy = det_xyxy.reshape(0, 4)
            
        det_conf = np.ones(len(online_targets)) if len(online_targets) > 0 else np.array([])
        det_class = np.zeros(len(online_targets), dtype=int) if len(online_targets) > 0 else np.array([])
        
        labels = [
            f"#{target.track_id} {self.max_speeds[target.track_id]:.1f} km/h"
            for target in online_targets
        ]
        
        detection_objs = sv.Detections(
            xyxy=det_xyxy,
            confidence=det_conf,
            class_id=det_class
        )
        
        return self.box_annotator.annotate(
            scene=frame,
            detections=detection_objs,
            labels=labels
        )

# Re-export BYTETracker for convenience
__all__ = ['BYTETracker', 'get_tracker']
