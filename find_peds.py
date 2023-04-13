import cv2
import streamlink
import numpy as np
import dlib
import argparse


def get_stream_url(video_url):
    streams = streamlink.streams(video_url)
    return streams['best'].url


# video_url = "https://www.youtube.com/watch?v=b7lsZ-0KiJw"
video_url = "https://www.youtube.com/watch?v=R3YNscjcJOk"
# venice beach
video_url = "https://www.youtube.com/watch?v=w_DfTc7F5oQ"

# too slow joe
# video_url = "https://www.youtube.com/watch?v=e_WBuBqS9h8"
# times square
# video_url = "https://www.youtube.com/watch?v=1-iS7LArMPA"

# video_url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"

stream_url = get_stream_url(video_url)


def detect_pedestrians(frame, classifier_path):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    height, width = frame.shape[:2]
    scale_factor = 0.5

    # resized_frame = cv2.resize(
    #  frame, (int(width * scale_factor), int(height * scale_factor)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    finalimage = gray

    pedestrians, _ = hog.detectMultiScale(
        finalimage, winStride=(4, 4), padding=(8, 8), scale=1.05)

    pedestrians = np.array([[int(x / scale_factor), int(y / scale_factor), int(
        w / scale_factor), int(h / scale_factor)] for (x, y, w, h) in pedestrians])

    return finalimage, pedestrians


def detect_vehicles(frame, classifier_path):

    vehicle_cascade = cv2.CascadeClassifier(classifier_path)

    # pre processing

    # image = frame.resize((450, 250))
    # image_arr = np.array(image)

    # define the alpha and beta
    alpha = 1.5  # Contrast control
    beta = 10  # Brightness control

    # call convertScaleAbs function
    # adjusted = cv2.convertScaleAbs(frame, outimage, alpha=alpha, beta=beta)

    image_arr = np.array(frame)

    adjusted = cv2.convertScaleAbs(
        image_arr, image_arr, alpha=alpha, beta=beta)

    # make it gray!
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    wide = cv2.Canny(blur, 50, 200)
    mid = cv2.Canny(blur, 30, 150)
    tight = cv2.Canny(blur, 210, 250)

    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(blur, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

    # output
    finalimage = blur

    # vehicles = vehicle_cascade.detectMultiScale(finalimage, 1.1, 1)
    vehicles = vehicle_cascade.detectMultiScale(finalimage,
                                                scaleFactor=1.1,
                                                minNeighbors=3,
                                                minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

    return finalimage, vehicles


def main():
    cv2.ocl.setUseOpenCL(False)

    cap = cv2.VideoCapture(stream_url)

    cap.set(cv2.CAP_PROP_FPS, 20)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_FFMPEG, 1900)

    # classifier_path = "models/cars.xml"
    classifier_path = "models/hogcascade_pedestrians.xml"

    trackers = []
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1080 1920
        height, width = frame.shape[:2]
        output_size = (1000, 1000)

        scaling_factor = min(output_size[0]/width, output_size[1]/height)

        resized_image = cv2.resize(
            frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        preprocessed, vehicles = detect_pedestrians(
            resized_image, classifier_path)

        if counter % 10 == 0:
            # trackers.clear()

            for (x, y, w, h) in vehicles:
                tracker = dlib.correlation_tracker()
                counter += 1
                # cv2.rectangle(resized_image, (x, y),
                #              (x + w, y + h), (0, 255, 0), 2)
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(resized_image, rect)
                trackers.append(tracker)

        for tracker in trackers:

            tracker.update(resized_image)
            pos = tracker.get_position()
            x, y, w, h = int(pos.left()), int(pos.top()), int(
                pos.width()), int(pos.height())

            cv2.rectangle(resized_image, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)

        # add text to image

        resized_image = cv2.putText(resized_image, "Pedestrians: " + str(counter), (50, 420),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Pre Processing', preprocessed)
        cv2.imshow('Vehicle Detection', resized_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
