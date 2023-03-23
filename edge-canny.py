# Import OpenCV library and matplotlib
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib backend to TkAgg
matplotlib.use('TkAgg')

# Read grayscale image from file
image = cv2.imread('images/apple.jpg', cv2.IMREAD_GRAYSCALE)

# Display original image using matplotlib
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.show()

# Apply Canny edge detection to the image
canny_image = cv2.Canny(image, 100, 200)

# Display Canny edge detection result using matplotlib
plt.imshow(canny_image, cmap='gray')
plt.title("Canny Edge Detection")
plt.show()
