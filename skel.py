# Import OpenCV library
import cv2
import os

# Set path to image file
image_path = 'images/sampleimage.png'

# Check if image file exists
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found.")
    exit()

# Read image from file
image = cv2.imread(image_path)

# Display original image
cv2.imshow('Original Image', image)

# Wait for keyboard event and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
