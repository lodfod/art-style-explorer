import cv2
import numpy as np
from typing import Tuple, Optional


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to the range [0, 1]
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    else:
        # Ensure the image is in the range [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        
        if min_val == max_val:
            return np.zeros_like(image, dtype=np.float32)
        
        return (image - min_val) / (max_val - min_val)


def adjust_brightness_contrast(image: np.ndarray, 
                               alpha: float = 1.0, 
                               beta: float = 0) -> np.ndarray:
    """
    Adjust brightness and contrast of an image
    
    Args:
        image: Input image
        alpha: Contrast control (1.0 means no change)
        beta: Brightness control (0 means no change)
        
    Returns:
        Adjusted image
    """
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Equalize the histogram of an image to improve contrast
    
    Args:
        image: Input grayscale image
        
    Returns:
        Equalized image
    """
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Check if image is grayscale or color
    if len(image.shape) == 2 or image.shape[2] == 1:
        return cv2.equalizeHist(image)
    else:
        # Convert to YCrCb and equalize Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def remove_background(image: np.ndarray, 
                      threshold: int = 240,
                      kernel_size: int = 5) -> np.ndarray:
    """
    Remove the background from an image using thresholding and morphological operations
    
    Args:
        image: Input image
        threshold: Threshold value for background separation
        kernel_size: Size of the morphological kernel
        
    Returns:
        Image with removed background
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Ensure uint8 type
    if gray.dtype != np.uint8:
        if np.max(gray) <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    # Create a binary mask using threshold
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Perform morphological operations to clean up the mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the mask to the original image
    if len(image.shape) == 3:
        # Color image
        result = image.copy()
        result[mask == 0] = [255, 255, 255]  # Set background to white
    else:
        # Grayscale image
        result = image.copy()
        result[mask == 0] = 255  # Set background to white
        
    return result


def adaptive_thresholding(image: np.ndarray, 
                         block_size: int = 11, 
                         C: int = 2) -> np.ndarray:
    """
    Apply adaptive thresholding to an image
    
    Args:
        image: Input grayscale image
        block_size: Size of the pixel neighborhood used for threshold calculation
        C: Constant subtracted from the mean
        
    Returns:
        Binary image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Ensure uint8 type
    if gray.dtype != np.uint8:
        if np.max(gray) <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    
    # Apply adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        block_size, 
        C
    )
    
    return binary


def denoise_image(image: np.ndarray, 
                 method: str = 'gaussian',
                 kernel_size: int = 5,
                 strength: float = 1.0) -> np.ndarray:
    """
    Denoise an image using various methods
    
    Args:
        image: Input image
        method: Denoising method ('gaussian', 'median', 'bilateral', 'nlm')
        kernel_size: Size of the kernel
        strength: Strength of the denoising effect
        
    Returns:
        Denoised image
    """
    # Ensure uint8 type
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), strength)
    
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    
    elif method == 'bilateral':
        d = kernel_size
        sigma_color = strength * 75
        sigma_space = strength * 75
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    elif method == 'nlm':
        # Non-local means denoising
        h = strength * 10  # Filter strength
        h_color = strength * 10  # Color component filter strength
        template_window_size = kernel_size
        search_window_size = kernel_size * 2 + 1
        
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, 
                None, 
                h=h, 
                hColor=h_color, 
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, 
                None, 
                h=h, 
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
    
    else:
        raise ValueError(f"Unsupported denoising method: {method}")


def standardize_artwork(image: np.ndarray, 
                       target_size: Tuple[int, int] = (512, 512),
                       remove_bg: bool = True,
                       denoise: bool = True,
                       equalize: bool = True,
                       threshold: Optional[int] = None) -> np.ndarray:
    """
    Standardize artwork image for consistent processing
    
    Args:
        image: Input image
        target_size: Target size for resizing
        remove_bg: Whether to remove the background
        denoise: Whether to denoise the image
        equalize: Whether to equalize the histogram
        threshold: Threshold value for optional adaptive thresholding
        
    Returns:
        Standardized image
    """
    # Resize the image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Denoise if requested
    if denoise:
        # Use bilateral filter to preserve edges
        resized = denoise_image(resized, method='bilateral')
    
    # Remove background if requested
    if remove_bg:
        resized = remove_background(resized)
    
    # Equalize histogram if requested
    if equalize:
        resized = equalize_histogram(resized)
    
    # Apply thresholding if requested
    if threshold is not None:
        resized = adaptive_thresholding(resized, block_size=11, C=threshold)
    
    return resized 