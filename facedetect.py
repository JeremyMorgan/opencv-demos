# Import OpenCV library
import cv2

# Read image from file
image = cv2.imread('images/face.jpg')

# Display the original image
cv2.imshow('Original Image', image)

# Load a face cascade classifier
face_cascade = cv2.CascadeClassifier(
    'models/haarcascade_frontalface_default.xml')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect and store faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Print number of faces detected
print("bodies: " + str(len(faces)))

# Loop through detected faces and draw green rectangles around them
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display image with detected faces
cv2.imshow('Face Detected', image)

# Wait for keyboard event and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
