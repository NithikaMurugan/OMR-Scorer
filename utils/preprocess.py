import cv2
import numpy as np
from PIL import Image

def correct_orientation_and_perspective(pil_image):
    """
    Simple and safe image preprocessing that only enhances without distortion.
    Returns PIL Image in the same format as input.
    """
    print("Starting image preprocessing...")
    
    # Convert PIL to numpy array
    img = np.array(pil_image)
    print(f"Input image shape: {img.shape}")
    
    # For now, let's just apply minimal processing to avoid distortion
    # Only apply basic contrast enhancement and noise reduction
    
    if len(img.shape) == 3:
        # Color image - apply CLAHE to each channel
        print("Processing color image...")
        enhanced = img.copy()
        
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
    else:
        # Grayscale image
        print("Processing grayscale image...")
        # Apply gentle noise reduction
        denoised = cv2.fastNlMeansDenoising(img)
        
        # Apply gentle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
    
    print(f"Output image shape: {enhanced.shape}")
    print("Preprocessing completed - returning enhanced image")
    
    # Convert back to PIL Image
    return Image.fromarray(enhanced)

def detect_registration_markers(gray):
    """
    Detect circular or square registration markers typically found on OMR sheets.
    """
    registration_points = []
    
    # Apply preprocessing
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Method 1: Detect circular markers using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                              param1=50, param2=30, minRadius=10, maxRadius=50)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Verify it's a registration marker by checking if it's filled/dark
            roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                      max(0, x-r):min(gray.shape[1], x+r)]
            if roi.size > 0:
                mean_intensity = np.mean(roi)
                if mean_intensity < 100:  # Dark marker
                    registration_points.append((x, y))
    
    # Method 2: Detect square markers
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 2000:  # Size filter for markers
            # Check if contour is roughly square
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # Square-like shape
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.8 <= aspect_ratio <= 1.2:  # Nearly square
                    center_x, center_y = x + w//2, y + h//2
                    registration_points.append((center_x, center_y))
    
    # Remove duplicate points that are too close
    filtered_points = []
    for point in registration_points:
        is_duplicate = False
        for existing in filtered_points:
            if np.linalg.norm(np.array(point) - np.array(existing)) < 30:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_points.append(point)
    
    return filtered_points

def align_using_markers(gray, registration_points):
    """
    Align image using detected registration markers.
    """
    if len(registration_points) < 4:
        return None
    
    # Sort points: top-left, top-right, bottom-right, bottom-left
    points = np.array(registration_points)
    
    # Find corners based on position
    center = np.mean(points, axis=0)
    
    top_points = points[points[:, 1] < center[1]]
    bottom_points = points[points[:, 1] >= center[1]]
    
    if len(top_points) >= 2 and len(bottom_points) >= 2:
        # Sort top points by x-coordinate
        top_points = top_points[top_points[:, 0].argsort()]
        bottom_points = bottom_points[bottom_points[:, 0].argsort()]
        
        # Take leftmost and rightmost points
        top_left = top_points[0]
        top_right = top_points[-1]
        bottom_left = bottom_points[0]
        bottom_right = bottom_points[-1]
        
        src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        
        # Define destination points (standard A4 ratio)
        width = max(np.linalg.norm(top_right - top_left), 
                   np.linalg.norm(bottom_right - bottom_left))
        height = max(np.linalg.norm(top_left - bottom_left),
                    np.linalg.norm(top_right - bottom_right))
        
        dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], 
                             dtype="float32")
        
        # Apply perspective transformation
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(gray, M, (int(width), int(height)))
        
        return warped
    
    return None

def align_using_contours(gray):
    """
    Improved contour-based alignment with better parameter tuning.
    """
    # Apply preprocessing with more conservative parameters
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple thresholding approaches
    thresh_methods = [
        # Method 1: OTSU thresholding
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        
        # Method 2: Adaptive thresholding
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2),
        
        # Method 3: Fixed threshold
        lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1]
    ]
    
    for thresh_method in thresh_methods:
        try:
            thresh = thresh_method(blurred)
            
            # Clean up the threshold image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Filter contours by area (should be significant portion of image)
            min_area = (gray.shape[0] * gray.shape[1]) * 0.1  # At least 10% of image
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if not valid_contours:
                continue
            
            # Sort by area and take the largest
            contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours_sorted[:3]:  # Try top 3 largest contours
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # If we have 4 points, try perspective transform
                if len(approx) >= 4:
                    # Take first 4 points if more than 4
                    pts = approx[:4].reshape(4, 2)
                    
                    # Check if points form a reasonable quadrilateral
                    if is_valid_quadrilateral(pts, gray.shape):
                        rect = order_points(pts)
                        
                        # Calculate dimensions with aspect ratio validation
                        width = max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3]))
                        height = max(np.linalg.norm(rect[0]-rect[3]), np.linalg.norm(rect[1]-rect[2]))
                        
                        # Check aspect ratio (should be reasonable for OMR sheet)
                        aspect_ratio = width / height if height > 0 else 1
                        if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio
                            
                            # Apply perspective transformation
                            dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], 
                                          dtype="float32")
                            M = cv2.getPerspectiveTransform(rect, dst)
                            warped = cv2.warpPerspective(gray, M, (int(width), int(height)))
                            
                            # Validate the result
                            if warped.shape[0] > 100 and warped.shape[1] > 100:
                                return warped
        
        except Exception as e:
            print(f"Error in contour method: {e}")
            continue
    
    # If all methods fail, return original
    return gray

def order_points(pts):
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # Top-left point has smallest sum
    rect[0] = pts[np.argmin(s)]
    
    # Bottom-right point has largest sum
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point has smallest difference
    rect[1] = pts[np.argmin(diff)]
    
    # Bottom-left point has largest difference
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def detect_sheet_orientation(gray):
    """
    Detect if the sheet is rotated and suggest correction angle.
    """
    # Detect lines using HoughLines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for rho, theta in lines[:10]:  # Use first 10 lines
            angle = theta * 180 / np.pi
            # Convert to rotation angle
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
        
        # Find the most common angle
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 2:  # If rotation is significant
                return median_angle
    
    return 0  # No rotation needed

def is_valid_quadrilateral(pts, image_shape):
    """
    Check if the 4 points form a valid quadrilateral for OMR sheet.
    """
    if len(pts) != 4:
        return False
    
    # Check if points are within image bounds
    height, width = image_shape
    for point in pts:
        x, y = point
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
    
    # Check if points form a convex quadrilateral
    # Calculate the area using shoelace formula
    area = 0.5 * abs(sum(pts[i][0] * (pts[(i+1)%4][1] - pts[(i-1)%4][1]) for i in range(4)))
    
    # Area should be reasonable (not too small)
    min_area = (width * height) * 0.1  # At least 10% of image
    if area < min_area:
        return False
    
    return True

def is_valid_processed_image(processed, original):
    """
    Validate if the processed image is reasonable.
    """
    if processed is None:
        return False
    
    # Check dimensions
    if processed.shape[0] < 100 or processed.shape[1] < 100:
        return False
    
    # Check if image is not too distorted (compare variance)
    orig_var = np.var(original)
    proc_var = np.var(processed)
    
    # Processed image should have reasonable variance
    if proc_var < orig_var * 0.1 or proc_var > orig_var * 10:
        return False
    
    # Check if image is not mostly black or white
    mean_intensity = np.mean(processed)
    if mean_intensity < 10 or mean_intensity > 245:
        return False
    
    return True

def apply_same_transform_to_color(original_color, original_gray, transformed_gray):
    """
    Apply the same transformation that was applied to grayscale to the color image.
    This is a simplified approach - for more complex cases, you'd need to store the transformation matrix.
    """
    # For now, return the original color image as this is complex to implement properly
    # In a production system, you'd want to store the transformation matrix and apply it
    return original_color
