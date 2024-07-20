import cv2
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms

# Load the pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Define the preprocessing transform
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the input and output directoriessytrewqa211
input_dir = "/NFSDISK2/rakshith/data_babu/"
final_cropped_output_dir = "/NFSDISK2/rakshith/Stereo_Data/Shivashish/babutest"

# Create the output directories if they don't exist
os.makedirs(final_cropped_output_dir, exist_ok=True)

# Create a color palette for segmentation
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Function to process and save the final cropped image
def process_and_save_cropped_image(input_image_path, final_cropped_output_image_path):
    input_image = Image.open(input_image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    
    # Modify color palette: turn the current blue color into white
    blue_index = 15  # Assuming the class index for blue is 15
    colors[blue_index] = [255, 255, 255]  # Change blue to white

    # Create an image with the segmentation predictions
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    # Convert segmented image to numpy array
    seg_img = np.array(r.convert('RGB'))

    # Define the range for non-white colors
    lower_non_white = np.array([0, 0, 0])
    upper_non_white = np.array([254, 254, 254])

    # Create a mask for non-white colors
    mask = cv2.inRange(seg_img, lower_non_white, upper_non_white)

    # Change non-white pixels to black
    seg_img[mask != 0] = [0, 0, 0]

    # Convert back to PIL Image for further processing
    processed_image = Image.fromarray(seg_img)

    # Convert the segmented image to binary (white parts)
    gray_image = processed_image.convert('L')
    binary_image = np.array(gray_image)
    binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)[1]

    # Crop the binary image along the x-axis
    cropped_binary_image = binary_image[:, 700:-600]
    
    # Find contours on the cropped binary image
    contours, _ = cv2.findContours(cropped_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which is assumed to be the person
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_segmented_image = processed_image.crop((x + 700, y, x + w + 700, y + h))  # Adjust x coordinates for the original image
        cropped_segmented_image.save(final_cropped_output_image_path)
    else:
        print(f"No main parts found in {input_image_path}")

# Process all images in the input directory and save the final cropped results
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        input_image_path = os.path.join(input_dir, filename)
        final_cropped_output_image_path = os.path.join(final_cropped_output_dir, filename)
        process_and_save_cropped_image(input_image_path, final_cropped_output_image_path)

print(f"Semantic segmentation and cropping completed. Final results saved in {final_cropped_output_dir}.")
