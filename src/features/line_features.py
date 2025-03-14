import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math


def extract_hough_lines(edge_image: np.ndarray, 
                       threshold: int = 100, 
                       min_line_length: int = 50, 
                       max_line_gap: int = 5) -> np.ndarray:
    """
    Extract lines from an edge image using the Hough transform
    
    Args:
        edge_image: Binary edge image
        threshold: Accumulator threshold parameter
        min_line_length: Minimum line length
        max_line_gap: Maximum gap between line segments
        
    Returns:
        Array of line segments [x1, y1, x2, y2]
    """
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(
        edge_image, 
        rho=1, 
        theta=np.pi/180, 
        threshold=threshold, 
        minLineLength=min_line_length, 
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return np.array([])
    
    return lines.reshape(-1, 4)  # Reshape to [x1, y1, x2, y2] format


def calculate_line_angles(lines: np.ndarray) -> np.ndarray:
    """
    Calculate angles of lines in radians
    
    Args:
        lines: Array of line segments [x1, y1, x2, y2]
        
    Returns:
        Array of angles in radians
    """
    if len(lines) == 0:
        return np.array([])
    
    angles = []
    for x1, y1, x2, y2 in lines:
        angle = np.arctan2(y2 - y1, x2 - x1)
        # Normalize to [0, pi]
        if angle < 0:
            angle += np.pi
        angles.append(angle)
    
    return np.array(angles)


def calculate_line_lengths(lines: np.ndarray) -> np.ndarray:
    """
    Calculate lengths of line segments
    
    Args:
        lines: Array of line segments [x1, y1, x2, y2]
        
    Returns:
        Array of line lengths
    """
    if len(lines) == 0:
        return np.array([])
    
    lengths = []
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lengths.append(length)
    
    return np.array(lengths)


def calculate_line_curvature(contours: List[np.ndarray]) -> Dict[str, float]:
    """
    Calculate curvature statistics of contours
    
    Args:
        contours: List of contours
        
    Returns:
        Dictionary of curvature statistics
    """
    curvature_stats = {
        'mean_curvature': 0.0,
        'std_curvature': 0.0,
        'max_curvature': 0.0,
        'min_curvature': 0.0
    }
    
    if not contours:
        return curvature_stats
    
    curvatures = []
    
    for contour in contours:
        if len(contour) < 5:  # Need at least 5 points for ellipse fitting
            continue
            
        try:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Calculate eccentricity
            a = max(axes) / 2
            b = min(axes) / 2
            
            if a > 0 and b > 0:
                # Calculate curvature as 1/eccentricity
                eccentricity = np.sqrt(1 - (b/a)**2)
                curvature = 1 / (1 + eccentricity) if eccentricity < 1 else 0
                curvatures.append(curvature)
        except:
            # Skip contours that can't be fit with an ellipse
            pass
    
    if curvatures:
        curvature_stats['mean_curvature'] = np.mean(curvatures)
        curvature_stats['std_curvature'] = np.std(curvatures)
        curvature_stats['max_curvature'] = np.max(curvatures)
        curvature_stats['min_curvature'] = np.min(curvatures)
    
    return curvature_stats


def calculate_line_orientation_histogram(angles: np.ndarray, 
                                       bins: int = 18) -> np.ndarray:
    """
    Calculate histogram of line orientations
    
    Args:
        angles: Array of angles in radians
        bins: Number of histogram bins
        
    Returns:
        Histogram of line orientations
    """
    if len(angles) == 0:
        return np.zeros(bins)
    
    # Calculate histogram
    hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi))
    
    # Normalize histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    return hist


def calculate_line_statistics(lines: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about the lines
    
    Args:
        lines: Array of line segments [x1, y1, x2, y2]
        
    Returns:
        Dictionary of line statistics
    """
    stats = {
        'line_count': 0,
        'mean_length': 0.0,
        'std_length': 0.0,
        'max_length': 0.0,
        'min_length': 0.0,
        'horizontal_ratio': 0.0,
        'vertical_ratio': 0.0,
        'diagonal_ratio': 0.0
    }
    
    if len(lines) == 0:
        return stats
    
    # Calculate line angles and lengths
    angles = calculate_line_angles(lines)
    lengths = calculate_line_lengths(lines)
    
    # Count lines
    stats['line_count'] = len(lines)
    
    # Length statistics
    stats['mean_length'] = np.mean(lengths)
    stats['std_length'] = np.std(lengths)
    stats['max_length'] = np.max(lengths)
    stats['min_length'] = np.min(lengths)
    
    # Count orientation categories
    horizontal_count = 0
    vertical_count = 0
    diagonal_count = 0
    
    for angle in angles:
        # Horizontal: within 15 degrees of 0 or 180 degrees
        if angle < np.radians(15) or angle > np.radians(165):
            horizontal_count += 1
        # Vertical: within 15 degrees of 90 degrees
        elif np.radians(75) < angle < np.radians(105):
            vertical_count += 1
        # Diagonal: everything else
        else:
            diagonal_count += 1
    
    # Calculate ratios
    total_lines = len(lines)
    stats['horizontal_ratio'] = horizontal_count / total_lines if total_lines > 0 else 0
    stats['vertical_ratio'] = vertical_count / total_lines if total_lines > 0 else 0
    stats['diagonal_ratio'] = diagonal_count / total_lines if total_lines > 0 else 0
    
    return stats


def calculate_line_intersections(lines: np.ndarray) -> int:
    """
    Calculate the number of line intersections
    
    Args:
        lines: Array of line segments [x1, y1, x2, y2]
        
    Returns:
        Number of intersections
    """
    if len(lines) < 2:
        return 0
    
    intersections = 0
    
    # Check all pairs of lines
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Check if the lines intersect
            denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            
            if denominator == 0:  # Lines are parallel
                continue
                
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator
            
            # Check if the intersection point is within both line segments
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                intersections += 1
    
    return intersections


def extract_line_features(edge_image: np.ndarray, 
                         contours: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """
    Extract comprehensive line features from an edge image
    
    Args:
        edge_image: Binary edge image
        contours: Optional list of contours (if already calculated)
        
    Returns:
        Dictionary of line features
    """
    # Extract Hough lines
    lines = extract_hough_lines(edge_image)
    
    # Get contours if not provided
    if contours is None:
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate line statistics
    line_stats = calculate_line_statistics(lines)
    
    # Calculate line curvature
    curvature_stats = calculate_line_curvature(contours)
    
    # Calculate line orientations
    angles = calculate_line_angles(lines)
    orientation_hist = calculate_line_orientation_histogram(angles)
    
    # Calculate line intersections
    intersections = calculate_line_intersections(lines)
    
    # Calculate contour complexity
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    contour_perimeters = [cv2.arcLength(contour, True) for contour in contours]
    
    complexity = 0
    for area, perimeter in zip(contour_areas, contour_perimeters):
        if perimeter > 0:
            # Complexity as a ratio of perimeter squared to area (inverse of circularity)
            complexity += (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
    
    avg_complexity = complexity / len(contours) if contours else 0
    
    # Combine all features into a single dictionary
    features = {
        **line_stats,
        **curvature_stats,
        'orientation_histogram': orientation_hist,
        'intersection_count': intersections,
        'contour_count': len(contours),
        'avg_complexity': avg_complexity
    }
    
    return features


def visualize_lines(image: np.ndarray, 
                   lines: np.ndarray, 
                   color: Tuple[int, int, int] = (0, 0, 255), 
                   thickness: int = 2) -> np.ndarray:
    """
    Visualize detected lines on an image
    
    Args:
        image: Input image
        lines: Array of line segments [x1, y1, x2, y2]
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn lines
    """
    # Create a copy of the image
    result = image.copy()
    
    # Draw lines
    for x1, y1, x2, y2 in lines:
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return result 