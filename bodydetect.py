# Import OpenCV library
import cv2

# Detect Bodies in an image
# Note, we didn't get this to work, and ran out of time.

# Read image from file
image = cv2.imread('images/body2.jpg')

# Display original image
cv2.imshow('Original Image', image)

# Load full body cascade classifier
face_cascade = cv2.CascadeClassifier('models/haarcascade_fullbody.xml')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect bodies in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Print number of bodies detected
print("bodies: " + str(len(faces)))

# Loop through detected bodies and draw green rectangles around them
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display image with detected bodies
cv2.imshow('Body Detected', image)

# Wait for keyboard event and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
