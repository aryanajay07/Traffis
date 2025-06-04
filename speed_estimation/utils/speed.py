# utils/speed.py

from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_speed(coordinates: list, fps: int) -> float:
    """
    Calculate speed from tracked coordinates using moving average
    
    Args:
        coordinates: List of coordinate points in meters (from transformed space)
        fps: Video frames per second
        
    Returns:
        float: Speed in km/h
    """
    # Need at least half a second of data for reliable speed calculation
    min_points = max(int(fps / 2), 2)  # At least 2 points or half a second of data
    if len(coordinates) < min_points:
        logger.debug(f"Not enough points for speed calculation. Have {len(coordinates)}, need {min_points}")
        return 0.0

    try:
        # Convert coordinates to numpy array
        coords_array = np.array(coordinates)
        
        # Calculate distances between consecutive points using Euclidean distance
        diffs = np.diff(coords_array, axis=0)  # Get differences between consecutive points
        distances = np.sqrt(np.sum(diffs**2, axis=1))  # Euclidean distances
        
        # Time between frames
        time = 1.0 / fps  # seconds
        
        # Calculate speeds
        if time > 0:
            speeds = (distances / time) * 3.6  # Convert m/s to km/h
            # Filter out unrealistic speeds (adjusted to be more lenient)
            speeds = speeds[(speeds >= 0) & (speeds <= 300)]  # Allow up to 300 km/h
        
        if not speeds.size:
            logger.debug("No valid speeds calculated")
            return 0.0
            
        # Use median for robustness against outliers
        median_speed = float(np.median(speeds))
        logger.debug(f"Calculated speed: {median_speed:.1f} km/h from {len(speeds)} measurements")
        return median_speed
        
    except Exception as e:
        logger.error(f"Error calculating speed: {str(e)}")
        logger.debug(f"Coordinates: {coordinates}")
        return 0.0
