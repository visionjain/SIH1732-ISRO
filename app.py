import streamlit as st
import os
import tempfile
import zipfile
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
from image_enhancer import (
    augment_image,
    enhance_image,
    apply_median_blur,
    detect_edges_and_find_best_landing_spot,
    detect_edges
)

# Set up directories for saving images
BASE_DIR = tempfile.gettempdir()
AUGMENTED_DIR = os.path.join(BASE_DIR, 'augmented_images')
ENHANCED_DIR = os.path.join(BASE_DIR, 'enhanced_images')
BLURRED_DIR = os.path.join(BASE_DIR, 'blurred_images')
EDGES_DIR = os.path.join(BASE_DIR, 'edges_images')

for directory in [AUGMENTED_DIR, ENHANCED_DIR, BLURRED_DIR, EDGES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Function to save image and ensure it is in RGB mode
def save_image(image, path):
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGB")
    image.save(path)

# Function to run inference using YOLO model
def run_inference(model, image):
    img = np.array(image)
    results = model(img)
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return image

# Function to convert annotated result to PIL image with labels
def result_to_pil_image(result):
    annotated_img = result.plot()
    pil_img = Image.fromarray(annotated_img)
    return pil_img

# Streamlit UI
st.title("Lunar Image Processor")
st.write("Upload multiple lunar images for enhancement and crater detection.")

# File uploader
uploaded_files = st.file_uploader("Choose JPG or JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    # Load the YOLO model
    model = YOLO('best.pt')  # Ensure best.pt is in the same directory as this script
    
    for index, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            save_image(image, tmp_file.name)
            tmp_image_path = tmp_file.name

        st.image(image, caption=f'Uploaded Image {index + 1}', use_column_width=True)
        
        # Perform image enhancements
        augmented_images = augment_image(tmp_image_path, AUGMENTED_DIR)
        enhanced_image_path = enhance_image(tmp_image_path, ENHANCED_DIR)
        blurred_image_path = apply_median_blur(tmp_image_path, BLURRED_DIR)
        edges_image_path = detect_edges(tmp_image_path, EDGES_DIR)
        landing_image_path = detect_edges_and_find_best_landing_spot(tmp_image_path, EDGES_DIR)

        # Display augmented images
        st.subheader(f"Augmented Images for Image {index + 1}")
        for aug_img_path in augmented_images:
            aug_image = Image.open(aug_img_path)
            st.image(aug_image, use_column_width=True)
        
        # Display enhanced image
        st.subheader(f"Enhanced Image {index + 1}")
        enhanced_image = Image.open(enhanced_image_path)
        st.image(enhanced_image, use_column_width=True, caption=f"Enhanced Image {index + 1}")
        
        # Display blurred image
        st.subheader(f"Median Blurred Image {index + 1}")
        blurred_image = Image.open(blurred_image_path)
        st.image(blurred_image, use_column_width=True, caption=f"Median Blurred Image {index + 1}")
        
        # Display edge-detected image
        st.subheader(f"Edge Detected Image {index + 1}")
        edges_image = Image.open(edges_image_path)
        st.image(edges_image, use_column_width=True, caption=f"Edge Detected Image {index + 1}")
        
        # Display edge-detected image with best landing spot
        st.subheader(f"Edge Detected Image {index + 1} with Best Landing Spot")
        if landing_image_path:
            landing_image = Image.open(landing_image_path)
            st.image(landing_image, use_column_width=True, caption=f"Edge Detected Image {index + 1} with Best Landing Spot")
        else:
            st.write("No suitable landing spot found.")

        # Run YOLO model for crater detection
        st.write('Processing crater detection...')
        results = run_inference(model, image)
        
        # Extract bounding boxes
        boxes = []
        for result in results:
            if result.boxes:
                boxes.extend(result.boxes.xyxy.cpu().numpy())  # Get bounding boxes
        
        # Draw bounding boxes on the image
        unlabeled_image = image
        labeled_image = draw_boxes(np.array(image), boxes)
        
        # Convert results to image with labels
        result_image_with_labels = result_to_pil_image(results[0])  # Assuming the first result
        
        # Display annotated image without labels
        st.subheader(f"Annotated Image {index + 1} (No Labels)")
        st.image(labeled_image, caption='Annotated Image with Craters (No Labels)', use_column_width=True)
        
        # Display annotated image with labels
        st.subheader(f"Annotated Image {index + 1} with Labels")
        st.image(result_image_with_labels, caption='Annotated Image with Craters and Labels', use_column_width=True)

    # Option to download all processed images as a ZIP file
    def create_zip():
        zip_path = os.path.join(BASE_DIR, 'processed_images.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for folder in [AUGMENTED_DIR, ENHANCED_DIR, BLURRED_DIR, EDGES_DIR]:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    zipf.write(file_path, arcname=os.path.join(os.path.basename(folder), filename))
        return zip_path

    if st.button("Download All Processed Images"):
        zip_file_path = create_zip()
        with open(zip_file_path, 'rb') as f:
            bytes_data = f.read()
            st.download_button(label="Download ZIP", data=bytes_data, file_name="processed_images.zip")
