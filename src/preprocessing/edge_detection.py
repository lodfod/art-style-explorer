import cv2
import numpy as np
from typing import Tuple, Union, Optional
import cv2.cuda as cv2cuda


def read_image(image_path: str) -> np.ndarray:
    """
    Read an image from a file path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        The image as a numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Preprocess image by resizing and converting to grayscale
    
    Args:
        image: Input image as numpy array
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image
    """
    # Resize the image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale if it's a color image
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
        
    return gray


def extract_edges_gpu(image: np.ndarray, method: str = 'canny'):
    # Move image to GPU
    gpu_img = cv2cuda.GpuMat()
    gpu_img.upload(image)
    
    if method.lower() == 'canny':
        # GPU Canny edge detection
        gpu_blur = cv2cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
        gpu_blurred = cv2cuda.GpuMat()
        gpu_blur.apply(gpu_img, gpu_blurred)
        
        gpu_canny = cv2cuda.createCannyEdgeDetector(50, 150)
        gpu_edges = cv2cuda.GpuMat()
        gpu_canny.detect(gpu_blurred, gpu_edges)
        
        # Download result back to CPU
        edges = gpu_edges.download()
        return edges
    
    # Similar implementations for Sobel and Laplacian
    # ...


def extract_edges(image: np.ndarray, 
                  method: str = 'canny',
                  low_threshold: int = 50, 
                  high_threshold: int = 150,
                  aperture_size: int = 3) -> np.ndarray:
    """
    Extract edges from an image using various methods
    
    Args:
        image: Input grayscale image
        method: Edge detection method ('canny', 'sobel', 'laplacian')
        low_threshold: Lower threshold for the hysteresis procedure (Canny only)
        high_threshold: Higher threshold for the hysteresis procedure (Canny only)
        aperture_size: Size of the Sobel kernel (3, 5, or 7)
        
    Returns:
        Binary edge image
    """
    if method.lower() == 'canny':
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)
    
    elif method.lower() == 'sobel':
        # Calculate the x and y gradients
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=aperture_size)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=aperture_size)
        
        # Calculate the gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize and convert to uint8
        edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply thresholding to get binary edge image
        _, edges = cv2.threshold(edges, low_threshold, 255, cv2.THRESH_BINARY)
    
    elif method.lower() == 'laplacian':
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=aperture_size)
        
        # Convert to absolute values and normalize
        abs_laplacian = np.abs(laplacian)
        edges = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply thresholding to get binary edge image
        _, edges = cv2.threshold(edges, low_threshold, 255, cv2.THRESH_BINARY)
    
    else:
        raise ValueError(f"Unsupported edge detection method: {method}")
    
    return edges


def thin_edges(edges: np.ndarray, method: str = 'zhang_suen') -> np.ndarray:
    """
    Thin edges to single-pixel width using skeletonization
    
    Args:
        edges: Binary edge image
        method: Thinning method ('zhang_suen' or 'guo_hall')
        
    Returns:
        Thinned edge image
    """
    # Create a copy to avoid modifying the original
    result = edges.copy()
    
    if method == 'zhang_suen':
        # Zhang-Suen thinning algorithm (simplified implementation)
        # This is a basic implementation and can be replaced with more sophisticated methods
        prev = np.zeros_like(result)
        
        while not np.array_equal(result, prev):
            prev = result.copy()
            
            # Identification and deletion of contour points
            # This is simplified and would normally be a more complex operation
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(result, kernel)
            result = cv2.dilate(eroded, kernel)
    
    elif method == 'guo_hall':
        # Here we would implement the Guo-Hall thinning algorithm
        # For simplicity, we'll use morphological operations as a placeholder
        kernel = np.ones((3, 3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    else:
        raise ValueError(f"Unsupported thinning method: {method}")
    
    return result


def process_artwork(image_path: str, 
                   target_size: Tuple[int, int] = (512, 512),
                   edge_method: str = 'canny',
                   low_threshold: int = 50,
                   high_threshold: int = 150,
                   thin_method: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process artwork to extract line work
    
    Args:
        image_path: Path to the artwork image
        target_size: Target size for resizing
        edge_method: Edge detection method
        low_threshold: Lower threshold for edge detection
        high_threshold: Higher threshold for edge detection
        thin_method: Method for edge thinning (None to skip)
        
    Returns:
        Tuple of (preprocessed image, line work)
    """
    # Read and preprocess the image
    original = read_image(image_path)
    preprocessed = preprocess_image(original, target_size)
    
    # Extract edges
    edges = extract_edges(
        preprocessed, 
        method=edge_method,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    # Thin edges if requested
    if thin_method:
        edges = thin_edges(edges, method=thin_method)
    
    return preprocessed, edges


def detect_contours(edge_image: np.ndarray) -> list:
    """
    Detect contours in an edge image
    
    Args:
        edge_image: Binary edge image
        
    Returns:
        List of contours
    """
    contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def analyze_contours(contours: list) -> dict:
    """
    Analyze contours to extract features related to line work
    
    Args:
        contours: List of contours
        
    Returns:
        Dictionary of contour features
    """
    features = {
        'count': len(contours),
        'avg_length': 0,
        'avg_curvature': 0,
        'length_variance': 0,
        'contour_areas': [],
        'contour_perimeters': [],
        'complexity': 0
    }
    
    if not contours:
        return features
    
    # Calculate perimeters and areas
    perimeters = [cv2.arcLength(contour, True) for contour in contours]
    areas = [cv2.contourArea(contour) for contour in contours]
    
    features['contour_areas'] = areas
    features['contour_perimeters'] = perimeters
    
    # Average length (perimeter)
    features['avg_length'] = np.mean(perimeters) if perimeters else 0
    
    # Length variance
    features['length_variance'] = np.var(perimeters) if perimeters else 0
    
    # Average curvature (area/perimeter ratio)
    curvatures = []
    for area, perimeter in zip(areas, perimeters):
        if perimeter > 0:
            # Circularity: 4π*area/perimeter²
            curvature = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            curvatures.append(curvature)
    
    features['avg_curvature'] = np.mean(curvatures) if curvatures else 0
    
    # Complexity: ratio of total perimeter to number of contours
    features['complexity'] = sum(perimeters) / len(contours) if contours else 0
    
    return features 