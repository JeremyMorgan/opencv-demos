# Import OpenCV library
import cv2

# Read image from file
image = cv2.imread('images/pexels.jpg')

# Get current image size
height, width = image.shape[:2]

# Print image height and width
print("Height: " + str(height))
print("Width: " + str(width))

# Set desired output size
output_size = (400, 400)

# Calculate scaling factor to fit image within output size
scaling_factor = min(output_size[0]/width, output_size[1]/height)

# Resize image using calculated scaling factor and interpolation method
resized_image = cv2.resize(
    image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# Get new image size after resizing
height2, width2 = resized_image.shape[:2]

# Print resized image height and width
print("Height: " + str(height2))
print("Width: " + str(width2))

# Display original and blurred images
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)

# Wait for keyboard event and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
