# Silhouette-Crops-Using-YOLO-and-SAM
Image Segmentation with YOLO and SAM This repository implements automated image segmentation and cropping using the YOLO object detection model and Segment Anything Model (SAM). 
Image Segmentation and Cropping with YOLO and SAM
This repository contains a Python implementation for automated image segmentation and cropping using the YOLO (You Only Look Once) object detection model and the Segment Anything Model (SAM). The primary goal of this project is to process images, detect objects, generate silhouette masks, and crop these silhouettes based on detected contours.
Features
Object Detection: Utilizes the YOLO model to detect objects in input images.
Segmentation: Employs the Segment Anything Model (SAM) to generate detailed segmentation masks for detected objects.
Silhouette Generation: Converts segmented images into binary silhouette masks.
Cropped Silhouettes: Automatically crops the silhouettes based on contours and saves them to the specified output directory.
Flexible Input/Output: Processes images from a specified input directory and saves cropped silhouettes to an output directory.
Installation
To run this code, you need to install the required dependencies. You can do this using pip:
bash
pip install ultralytics
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

Additionally, download the SAM model weights:
bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Usage
Place your input images in a folder (e.g., um2-out).
Specify the output folder where cropped silhouettes will be saved (e.g., mainoutput).
Run the script to process the images:
python
input_folder = 'um2-out'
output_folder = 'mainoutput'
process_images(input_folder, output_folder)

Example
After running the script, the cropped silhouettes will be saved in the specified output folder, ready for further analysis or use in other applications.
