# 1.Image Segmentation with YOLO and Segment Anything Model (SAM)

This repository contains Python code for performing image segmentation using the YOLO (You Only Look Once) model for object detection and the Segment Anything Model (SAM) for segmentation tasks. The primary goal of this project is to process images, generate silhouette masks, and crop these silhouettes based on detected contours.

## Features

- **Object Detection**: Utilizes YOLOv8 for detecting objects in images.
- **Segmentation**: Employs the Segment Anything Model (SAM) to create detailed segmentation masks.
- **Silhouette Cropping**: Automatically crops silhouettes from the segmented images and saves them in the specified output folder.
- **Flexible Input/Output**: Processes images from a specified input folder and saves cropped silhouettes to an output folder.

## Requirements

To run this code, you need to install the following dependencies:

```bash
pip install ultralytics
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# Additionally, download the SAM model weights:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Clone this repository:
git clone https://github.com/yourusername/ImageSegmentationWithYOLOandSAM.git
cd ImageSegmentationWithYOLOandSAM
# Run the script:
python mainseg.py



