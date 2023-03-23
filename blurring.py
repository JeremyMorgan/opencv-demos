# Import OpenCV library
import cv2

# Import numpy library
import numpy as np

# Read image from file
image = cv2.imread('images/sampleimage.png')

# Apply median blur to the image with a kernel size of 11x11
median_blur_image = cv2.medianBlur(image, 11)

# Display original image
cv2.imshow('Original Image', image)

# Display median blurred image
cv2.imshow('Median Blurred Image', median_blur_image)

# Wait for keyboard event and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
