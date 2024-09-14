import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from pathlib import Path

def augment_image(image_path, save_dir):
    """
    Perform image augmentation on a single image and save augmented images.
    """
    transformation_dict = {
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotation_range": 40,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": [0.5, 1.0],
        "brightness_range": [1.1, 1.5],
    }
    
    img = load_img(image_path)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    
    augmented_images = []
    
    for transformation, params in transformation_dict.items():
        datagen = ImageDataGenerator(**{transformation: params})
        it = datagen.flow(samples, batch_size=1)
        batch = next(it)
        augmented_image = batch[0].astype('uint8')
        augmented_image_path = os.path.join(save_dir, f"{Path(image_path).stem}_{transformation}.jpeg")
        cv2.imwrite(augmented_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        augmented_images.append(augmented_image_path)
    
    return augmented_images

def enhance_image(image_path, save_dir):
    """
    Enhance a single image by applying histogram equalization and sharpening.
    """
    image = cv2.imread(image_path)
    enhanced_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(enhanced_image)
    
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, sharpening_kernel)
    
    enhanced_image_path = os.path.join(save_dir, f"enhanced_{Path(image_path).name}")
    cv2.imwrite(enhanced_image_path, enhanced_image)
    
    return enhanced_image_path

def apply_median_blur(image_path, save_dir):
    """
    Apply median blur to a single image.
    """
    image = cv2.imread(image_path)
    blurred_image = cv2.medianBlur(image, 5)
    
    blurred_image_path = os.path.join(save_dir, f"blurred_{Path(image_path).name}")
    cv2.imwrite(blurred_image_path, blurred_image)
    
    return blurred_image_path

def detect_edges(image_path, save_dir):
    """
    Apply Canny edge detection to a single image and save the result.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    
    edge_image_path = os.path.join(save_dir, f"edges_{Path(image_path).name}")
    cv2.imwrite(edge_image_path, edges)
    
    return edge_image_path

def detect_edges_and_find_best_landing_spot(image_path, save_dir, min_area_threshold=500):
    """
    Apply Canny edge detection and find the best spot for landing (flat area), considering both 
    closed and open areas. Areas with low edge density and above a minimum area threshold 
    are considered good landing spots. Returns the best possible spot if no ideal spot is found.
    
    :param image_path: Path to the input image.
    :param save_dir: Directory to save the image with the marked landing spot.
    :param min_area_threshold: Minimum area size for the grid cell to be considered. 
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    
    # Parameters for grid-based edge density evaluation
    grid_size = 50  # Size of grid cells to evaluate edge density
    best_spot = None
    min_edge_density = float('inf')  # To store the minimum edge density found
    best_spot_found = False
    
    image_height, image_width = edges.shape
    
    # Iterate through the image in grid sections
    for y in range(0, image_height, grid_size):
        for x in range(0, image_width, grid_size):
            # Define the region of interest (ROI) for the current grid cell
            roi = edges[y:y+grid_size, x:x+grid_size]
            
            # Calculate edge density (number of white pixels in the ROI)
            edge_density = cv2.countNonZero(roi) / (grid_size * grid_size)
            
            # Calculate the area of the current grid cell
            area = cv2.countNonZero(roi)
            
            # Keep track of the region with the lowest edge density
            if edge_density < min_edge_density and area >= min_area_threshold:
                min_edge_density = edge_density
                best_spot = (x, y, grid_size, grid_size)  # Store the top-left corner and size of the region
                best_spot_found = True
    
    if not best_spot_found:
        # If no spot met the criteria, select the grid cell with the lowest edge density
        for y in range(0, image_height, grid_size):
            for x in range(0, image_width, grid_size):
                roi = edges[y:y+grid_size, x:x+grid_size]
                edge_density = cv2.countNonZero(roi) / (grid_size * grid_size)
                
                if edge_density < min_edge_density:
                    min_edge_density = edge_density
                    best_spot = (x, y, grid_size, grid_size)
    
    if best_spot is not None:
        # Draw a green rectangle on the original image to mark the best landing spot
        x, y, w, h = best_spot
        landing_image = image.copy()
        cv2.rectangle(landing_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the image with the landing spot marked
        landing_image_path = os.path.join(save_dir, f"landing_spot_{Path(image_path).name}")
        cv2.imwrite(landing_image_path, landing_image)
        
        return landing_image_path
    else:
        return None