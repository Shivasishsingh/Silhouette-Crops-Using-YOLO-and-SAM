# Install the required dependencies
# !pip install ultralytics
# !pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Import the necessary libraries
import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Function to process images and generate masks
def process_images(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    model = YOLO('yolov8n.pt')

    # Load SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # Process each image in the input folder
    for image_path in glob(os.path.join(input_folder, '*.png')):
        # Read and process the image
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        # Detect objects using YOLO
        results = model.predict(source=image_path, conf=0.25)

        # Initialize a white background and an empty mask
        white_background = np.ones_like(image) * 255
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Process each detected object
        for i, result in enumerate(results):
            for box in result.boxes.xyxy.tolist():
                input_box = np.array(box)

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                segmentation_mask = masks[0]
                binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
                combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)

        # Apply the combined mask to the original image
        final_image = white_background * (1 - combined_mask[..., np.newaxis]) + image * combined_mask[..., np.newaxis]

        # Convert the final image to grayscale
        gray = cv2.cvtColor(final_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Apply a threshold to create a binary image
        _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Crop the silhouette output data
        cropped_silhouette_path = crop_silhouette(binary_mask, image_path, output_folder)

        if cropped_silhouette_path:
            print(f"Cropped silhouette saved at: {cropped_silhouette_path}")
        else:
            print("No main parts found in the silhouette image.")

def crop_silhouette(binary_mask, image_path, output_folder):
    # Load the binary silhouette image using PIL
    processed_image = Image.fromarray(binary_mask)

    # Find contours on the cropped binary image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process the contours and crop the image if main parts are found
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ensure the adjusted coordinates do not exceed the image dimensions
        width, height = processed_image.size
        adjusted_x1 = max(0, min(x, width))
        adjusted_x2 = max(0, min(x + w, width))
        adjusted_y1 = max(0, min(y, height))
        adjusted_y2 = max(0, min(y + h, height))

        # Crop the image
        cropped_segmented_image = processed_image.crop((adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2))
        
        # Save the cropped silhouette
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        cropped_silhouette_path = os.path.join(output_folder, f'cropped_{base_filename}.png')
        cropped_segmented_image.save(cropped_silhouette_path)
        return cropped_silhouette_path
    else:
        return None

# Example usage
input_folder = 'um2-out'
output_folder = 'mainoutput'
process_images(input_folder, output_folder)