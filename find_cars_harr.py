import cv2
import streamlink
import numpy as np

# Function to detect cars in a frame


def detect_cars(frame, classifier):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the alpha and beta for brightness and contrast adjustment
    alpha = 1.3  # Contrast control
    beta = 5  # Brightness control

    # Adjust the brightness and contrast of the grayscale frame
    gray_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply Gaussian blur to the image to reduce noise
    final_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Use the classifier to detect cars in the image
    cars = classifier.detectMultiScale(final_image, 1.3, 5)

    return cars

# Function to draw boxes around detected cars


def draw_boxes(image, cars):
    for (x, y, w, h) in cars:
        # Draw a rectangle around the car in the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (11, 252, 3), 2)
    return image

# Function to get the URL of a YouTube stream


def get_youtube_stream_url(video_url):
    # Get all available streams for the given YouTube URL
    streams = streamlink.streams(video_url)
    # Select the best quality stream
    best_stream = streams["best"]
    return best_stream.url


# Main program
if __name__ == "__main__":
    # Load the car detection classifier
    car_cascade = cv2.CascadeClassifier('models/cars.xml')

    # amsterdam stream url
    # you_tube_stream_url = "https://www.youtube.com/watch?v=R3YNscjcJOk"
    # Gebhardt Insurance Traffic Cam Round Trip Bike Shop
    # you_tube_stream_url = "https://www.youtube.com/watch?v=_XBMMTQVj68"
    # Old Central School - Grand Rapids, MN
    # you_tube_stream_url = "https://www.youtube.com/watch?v=b7lsZ-0KiJw"
    # you_tube_stream_url = "https://www.youtube.com/watch?v=1-iS7LArMPA"

    # Set the URL of the YouTube video stream to be processed
    # City of Auburn Toomers Corner
    you_tube_stream_url = "https://www.youtube.com/watch?v=hMYIc5ZPJL4"

    # Get the URL of the best quality stream for the given YouTube URL
    video_source = get_youtube_stream_url(you_tube_stream_url)

    # Open the video stream
    cap = cv2.VideoCapture(video_source)

    # Continuously process frames from the video stream
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        # If there are no more frames, exit the loop
        if not ret:
            break

        # Detect cars in the current frame
        cars = detect_cars(frame, car_cascade)
        # Draw boxes around the detected cars
        result = draw_boxes(frame, cars)

        # Display the processed frame with the car boxes
        cv2.imshow('Car Detection', result)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close all windows
    cap.release()
    cv2.destroyAllWindows()
