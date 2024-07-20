
# 1.Image Segmentation with YOLO and Segment Anything Model (SAM)
## Use this code only for segmentation; the data contains person with umbrella.
This repository contains Python code for performing image segmentation using the YOLO (You Only Look Once) model for object detection and the Segment Anything Model (SAM) for segmentation tasks. The primary goal of this project is to process images, generate silhouette masks, and crop these silhouettes based on detected contours.








## Features

 - Object Detection: Utilizes YOLOv8 for detecting objects in images.
 -  Segmentation: Employs the Segment Anything Model (SAM) to create detailed segmentation masks.
 -  Silhouette Cropping: Automatically crops silhouettes from the segmented images and saves them in the specified output folder.
 - Flexible Input/Output: Processes images from a specified input folder and saves cropped silhouettes to an output folder.


## Requirements

#### To run this code, you need to install the following dependencies:
```bash
pip install ultralytics
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# Additionally, download the SAM model weights:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Clone this repository:
git clone https://github.com/yourusername/ImageSegmentationWithYOLOandSAM.git
cd ImageSegmentationWithYOLOandSAM

```
## Run the script:
```bash
python sam_yolo.py
```




# 2.DeepLabV3 Semantic Segmentation and Cropping
## Use this code only for segmentation; the data contains person with bag and normal walking person

This repository contains Python code for performing semantic segmentation using the DeepLabV3 model and cropping the segmented images based on contours. It utilizes the PyTorch library and the pre-trained DeepLabV3 model to segment images, and then crops the segmented images along the x-axis to isolate the main parts (e.g., a person).

## Features

- **Semantic Segmentation**: Uses the pre-trained DeepLabV3 model for pixel-wise segmentation of images.
- **Image Cropping**: Crops the segmented images along the x-axis based on the largest contour, which is assumed to be the main part (e.g., a person).
- **Flexible Input/Output**: Processes images from a specified input directory and saves the final cropped results in an output directory.

## Requirements

To run this code, you need to have the following dependencies installed:

- OpenCV (cv2)
- NumPy
- PyTorch
- Pillow (PIL)
- torchvision

You can install these dependencies using pip:

```bash
pip install opencv-python numpy torch pillow torchvision
```
## Run the script:
```bash
python dlseg.py
```
