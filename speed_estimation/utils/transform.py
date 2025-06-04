# utils/transform.py

import numpy as np
import cv2

# Define source points in image space (pixels)
SOURCE = np.array([[1272, 1178], [1839, 1119], [1839, 1834], [7, 1710]])

# Define target dimensions in real-world meters
# Assuming the road section is approximately 15 meters wide and 45 meters long
TARGET_WIDTH = 15.0  # meters
TARGET_HEIGHT = 45.0  # meters

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH, 0],
        [TARGET_WIDTH, TARGET_HEIGHT],
        [0, TARGET_HEIGHT],
    ],
    dtype=np.float32
)

class ViewTransformer:
    """Handles perspective transformation of points"""
    
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Initialize perspective transform
        
        Args:
            source: Source points in image space
            target: Target points in transformed space
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from image space to transformed space
        
        Args:
            points: Points to transform (N x 2)
            
        Returns:
            np.ndarray: Transformed points
        """
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)