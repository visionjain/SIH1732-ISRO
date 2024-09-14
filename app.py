import streamlit as st
import os
import tempfile
from PIL import Image
import zipfile
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

st.title("Lunar Polar Dark Image Enhancer")
st.write("Upload multiple lunar images, and get various enhanced versions!")

# Function to save image and ensure it is in RGB mode
def save_image(image, path):
    # Convert image to RGB if it has an alpha channel
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGB")
    
    # Save the image to the given path
    image.save(path)

# File uploader
uploaded_files = st.file_uploader("Choose JPG or JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for index, uploaded_file in enumerate(uploaded_files):
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption=f'Uploaded Image {index + 1}', use_column_width=True)
        
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            save_image(image, tmp_file.name)
            tmp_image_path = tmp_file.name
        
        st.write(f"Processing Image {index + 1}...")

        # Perform augmentations
        augmented_images = augment_image(tmp_image_path, AUGMENTED_DIR)
        
        # Enhance the original image
        enhanced_image_path = enhance_image(tmp_image_path, ENHANCED_DIR)
        
        # Apply median blur
        blurred_image_path = apply_median_blur(tmp_image_path, BLURRED_DIR)
        
        # Detect edges
        edges_image_path = detect_edges(tmp_image_path, EDGES_DIR)
        
        # Detect edges and find the best landing spot
        landing_image_path = detect_edges_and_find_best_landing_spot(tmp_image_path, EDGES_DIR)

        # Display augmented images
        st.subheader(f"Augmented Images for Image {index + 1}")
        for aug_img_path in augmented_images:
            aug_image = Image.open(aug_img_path)
            st.image(aug_image, use_column_width=True)
        
        # Display enhanced image
        st.subheader(f"Enhanced Image {index + 1} (Histogram Equalization & Sharpening)")
        enhanced_image = Image.open(enhanced_image_path)
        st.image(enhanced_image, use_column_width=True, caption=f"Enhanced Image {index + 1}")
        
        # Display blurred image
        st.subheader(f"Median Blurred Image {index + 1}")
        blurred_image = Image.open(blurred_image_path)
        st.image(blurred_image, use_column_width=True, caption=f"Median Blurred Image {index + 1}")
        
        # Display edge-detected image
        st.subheader(f"Edge Detected Image {index + 1} (Canny)")
        edges_image = Image.open(edges_image_path)
        st.image(edges_image, use_column_width=True, caption=f"Edge Detected Image {index + 1}")
        
        # Display edge-detected image with best landing spot
        st.subheader(f"Edge Detected Image {index + 1} with Best Landing Spot")
        if landing_image_path:
            landing_image = Image.open(landing_image_path)
            st.image(landing_image, use_column_width=True, caption=f"Edge Detected Image {index + 1} with Best Landing Spot")
        else:
            st.write("No suitable landing spot found.")
    
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
