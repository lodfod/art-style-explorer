import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import ndimage


def calculate_rule_of_thirds(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics related to the rule of thirds
    
    Args:
        image: Input image (grayscale or edge image)
        
    Returns:
        Dictionary of rule of thirds metrics
    """
    height, width = image.shape[:2]
    
    # Define rule of thirds lines
    h_lines = [height // 3, 2 * height // 3]
    v_lines = [width // 3, 2 * width // 3]
    
    # Define rule of thirds intersection points
    intersections = [
        (v_lines[0], h_lines[0]),  # Top-left
        (v_lines[1], h_lines[0]),  # Top-right
        (v_lines[0], h_lines[1]),  # Bottom-left
        (v_lines[1], h_lines[1])   # Bottom-right
    ]
    
    # Calculate energy along lines and at intersections
    h_line_energy = [np.mean(image[line, :]) for line in h_lines]
    v_line_energy = [np.mean(image[:, line]) for line in v_lines]
    
    # Calculate energy at intersection points (with small neighborhoods)
    intersection_energy = []
    neighborhood_size = 10  # Size of neighborhood around intersection points
    
    for x, y in intersections:
        x_min = max(0, x - neighborhood_size)
        x_max = min(width, x + neighborhood_size)
        y_min = max(0, y - neighborhood_size)
        y_max = min(height, y + neighborhood_size)
        
        neighborhood = image[y_min:y_max, x_min:x_max]
        energy = np.mean(neighborhood)
        intersection_energy.append(energy)
    
    # Calculate rule of thirds metrics
    metrics = {
        'horizontal_line_energy': np.mean(h_line_energy),
        'vertical_line_energy': np.mean(v_line_energy),
        'intersection_energy': np.mean(intersection_energy),
        'thirds_adherence': np.mean(h_line_energy + v_line_energy + intersection_energy)
    }
    
    return metrics


def calculate_symmetry(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate horizontal and vertical symmetry metrics
    
    Args:
        image: Input image (grayscale or edge image)
        
    Returns:
        Dictionary of symmetry metrics
    """
    height, width = image.shape[:2]
    
    # Calculate horizontal symmetry
    left_half = image[:, :width//2]
    right_half = np.fliplr(image[:, width//2:])
    
    # Ensure same dimensions for comparison
    min_width = min(left_half.shape[1], right_half.shape[1])
    h_diff = np.abs(left_half[:, :min_width] - right_half[:, :min_width])
    h_symmetry = 1 - np.mean(h_diff) / 255  # Higher value means more symmetry
    
    # Calculate vertical symmetry
    top_half = image[:height//2, :]
    bottom_half = np.flipud(image[height//2:, :])
    
    # Ensure same dimensions for comparison
    min_height = min(top_half.shape[0], bottom_half.shape[0])
    v_diff = np.abs(top_half[:min_height, :] - bottom_half[:min_height, :])
    v_symmetry = 1 - np.mean(v_diff) / 255  # Higher value means more symmetry
    
    # Calculate combined symmetry
    diag1 = np.fliplr(np.flipud(image))
    diag_diff = np.abs(image - diag1)
    diag_symmetry = 1 - np.mean(diag_diff) / 255
    
    metrics = {
        'horizontal_symmetry': float(h_symmetry),
        'vertical_symmetry': float(v_symmetry),
        'diagonal_symmetry': float(diag_symmetry),
        'overall_symmetry': float((h_symmetry + v_symmetry) / 2)
    }
    
    return metrics


def calculate_golden_ratio(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics related to the golden ratio
    
    Args:
        image: Input image (grayscale or edge image)
        
    Returns:
        Dictionary of golden ratio metrics
    """
    height, width = image.shape[:2]
    golden_ratio = (1 + np.sqrt(5)) / 2  # Approximately 1.618
    
    # Calculate golden ratio horizontal lines
    golden_h = [int(height / golden_ratio), int(height - (height / golden_ratio))]
    
    # Calculate golden ratio vertical lines
    golden_v = [int(width / golden_ratio), int(width - (width / golden_ratio))]
    
    # Define golden ratio spiral points (approximation)
    spiral_points = []
    
    # Start with full image
    x, y = 0, 0
    w, h = width, height
    
    # Generate spiral points for a few iterations
    for _ in range(5):
        # Calculate golden section
        golden_w = int(w / golden_ratio)
        golden_h = int(h / golden_ratio)
        
        # Add corner point
        spiral_points.append((x + w - golden_w, y + h - golden_h))
        
        # Reduce to golden section rectangle
        w = golden_w
        y += h - golden_h
        h = golden_h
        
        # Another section
        golden_w = int(w / golden_ratio)
        spiral_points.append((x + w - golden_w, y))
        
        # Reduce again
        h = golden_h - int(golden_h / golden_ratio)
        x += w - golden_w
        w = golden_w
    
    # Calculate energy along golden ratio lines
    h_line_energy = [np.mean(image[line, :]) for line in golden_h]
    v_line_energy = [np.mean(image[:, line]) for line in golden_v]
    
    # Calculate energy at spiral points
    spiral_energy = []
    neighborhood_size = 10
    
    for sx, sy in spiral_points:
        if 0 <= sx < width and 0 <= sy < height:
            x_min = max(0, sx - neighborhood_size)
            x_max = min(width, sx + neighborhood_size)
            y_min = max(0, sy - neighborhood_size)
            y_max = min(height, sy + neighborhood_size)
            
            neighborhood = image[y_min:y_max, x_min:x_max]
            energy = np.mean(neighborhood)
            spiral_energy.append(energy)
    
    # Calculate golden ratio metrics
    metrics = {
        'golden_horizontal_energy': np.mean(h_line_energy),
        'golden_vertical_energy': np.mean(v_line_energy),
        'golden_spiral_energy': np.mean(spiral_energy) if spiral_energy else 0,
        'golden_ratio_adherence': np.mean(h_line_energy + v_line_energy + (spiral_energy if spiral_energy else [0]))
    }
    
    return metrics


def calculate_balance(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate visual balance metrics
    
    Args:
        image: Input image (grayscale or edge image)
        
    Returns:
        Dictionary of balance metrics
    """
    height, width = image.shape[:2]
    
    # Calculate center of mass
    total_mass = np.sum(image)
    
    if total_mass == 0:
        return {
            'horizontal_balance': 0.5,
            'vertical_balance': 0.5,
            'radial_balance': 0.5,
            'center_offset_ratio': 0.0
        }
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[:height, :width]
    
    # Calculate center of mass
    cx = np.sum(x_coords * image) / total_mass
    cy = np.sum(y_coords * image) / total_mass
    
    # Normalize to [0, 1] range
    cx_norm = cx / width
    cy_norm = cy / height
    
    # Calculate distance from center
    center_x, center_y = width / 2, height / 2
    offset_distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    
    # Normalize offset distance
    max_distance = np.sqrt(center_x**2 + center_y**2)
    offset_ratio = offset_distance / max_distance
    
    # Calculate horizontal and vertical balance
    h_balance = 1 - abs(cx_norm - 0.5) * 2  # 1 is perfect balance, 0 is total imbalance
    v_balance = 1 - abs(cy_norm - 0.5) * 2
    
    # Calculate quadrant weights
    q1 = np.sum(image[:height//2, :width//2])  # Top-left
    q2 = np.sum(image[:height//2, width//2:])  # Top-right
    q3 = np.sum(image[height//2:, :width//2])  # Bottom-left
    q4 = np.sum(image[height//2:, width//2:])  # Bottom-right
    
    quadrant_sum = q1 + q2 + q3 + q4
    
    if quadrant_sum == 0:
        radial_balance = 0.5
    else:
        # Calculate how evenly distributed the weight is across quadrants
        q1_ratio = q1 / quadrant_sum
        q2_ratio = q2 / quadrant_sum
        q3_ratio = q3 / quadrant_sum
        q4_ratio = q4 / quadrant_sum
        
        # Perfect balance would be 0.25 for each quadrant
        radial_balance = 1 - (abs(q1_ratio - 0.25) + abs(q2_ratio - 0.25) + 
                             abs(q3_ratio - 0.25) + abs(q4_ratio - 0.25))
    
    metrics = {
        'horizontal_balance': float(h_balance),
        'vertical_balance': float(v_balance),
        'radial_balance': float(radial_balance),
        'center_offset_ratio': float(offset_ratio)
    }
    
    return metrics


def calculate_focal_points(image: np.ndarray, 
                          num_points: int = 3,
                          min_distance: int = 50) -> Dict[str, Any]:
    """
    Identify potential focal points in the image
    
    Args:
        image: Input image (grayscale or edge image)
        num_points: Number of focal points to identify
        min_distance: Minimum distance between focal points
        
    Returns:
        Dictionary containing focal point information
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Find local maxima
    data_max = ndimage.maximum_filter(blurred, size=min_distance)
    maxima = (blurred == data_max)
    
    # Remove background pixels
    threshold = np.mean(blurred) + np.std(blurred)
    maxima[blurred < threshold] = 0
    
    # Find coordinates of maxima
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    # Extract coordinates and values
    points = []
    for dy, dx in slices:
        x_center = int((dx.start + dx.stop - 1) / 2)
        y_center = int((dy.start + dy.stop - 1) / 2)
        val = blurred[y_center, x_center]
        points.append((x_center, y_center, val))
    
    # Sort by value (descending)
    points.sort(key=lambda p: p[2], reverse=True)
    
    # Take top N points
    top_points = points[:num_points] if len(points) > num_points else points
    
    # Convert to normalized coordinates
    height, width = image.shape[:2]
    normalized_points = [(x/width, y/height, val) for x, y, val in top_points]
    
    # Calculate distance from rule-of-thirds intersections
    roi_points = [
        (1/3, 1/3), (2/3, 1/3),
        (1/3, 2/3), (2/3, 2/3)
    ]
    
    roi_distances = []
    for x, y, _ in normalized_points:
        min_dist = min(np.sqrt((x - rx)**2 + (y - ry)**2) for rx, ry in roi_points)
        roi_distances.append(min_dist)
    
    # Calculate distance from golden ratio points
    golden_ratio = (1 + np.sqrt(5)) / 2
    gr = 1 / golden_ratio
    
    gr_points = [
        (gr, gr), (1-gr, gr),
        (gr, 1-gr), (1-gr, 1-gr)
    ]
    
    gr_distances = []
    for x, y, _ in normalized_points:
        min_dist = min(np.sqrt((x - gx)**2 + (y - gy)**2) for gx, gy in gr_points)
        gr_distances.append(min_dist)
    
    # Calculate spread of focal points (average distance between points)
    spread = 0
    if len(normalized_points) > 1:
        distances = []
        for i in range(len(normalized_points)):
            for j in range(i+1, len(normalized_points)):
                x1, y1, _ = normalized_points[i]
                x2, y2, _ = normalized_points[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distances.append(dist)
        spread = np.mean(distances)
    
    metrics = {
        'focal_points': normalized_points,
        'roi_adherence': 1 - np.mean(roi_distances) if roi_distances else 0,
        'golden_ratio_adherence': 1 - np.mean(gr_distances) if gr_distances else 0,
        'focal_spread': spread
    }
    
    return metrics


def extract_composition_features(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract comprehensive composition features from an image
    
    Args:
        image: Input image (grayscale or edge image)
        
    Returns:
        Dictionary of composition features
    """
    # Ensure image is grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize if needed
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    
    # Extract various composition metrics
    rule_of_thirds = calculate_rule_of_thirds(gray)
    symmetry = calculate_symmetry(gray)
    golden_ratio = calculate_golden_ratio(gray)
    balance = calculate_balance(gray)
    focal_points = calculate_focal_points(gray)
    
    # Combine all features into a single dictionary
    features = {
        **rule_of_thirds,
        **symmetry,
        **golden_ratio,
        **balance,
        'focal_points_info': focal_points
    }
    
    return features 