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

# Apply Sobel edge detection to the image
sobel_image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# Display Sobel edge detection result using matplotlib
plt.imshow(sobel_image, cmap='gray')
plt.title("Sobel Edge Detection")
plt.show()

# Wait for keyboard event and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
