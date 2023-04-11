import cv2
import streamlink
import numpy as np


def get_stream_url(video_url):
    streams = streamlink.streams(video_url)
    return streams['best'].url


video_url = "https://www.youtube.com/watch?v=b7lsZ-0KiJw"
# video_url = "https://www.youtube.com/watch?v=R3YNscjcJOk"
# video_url = "https://www.youtube.com/watch?v=e_WBuBqS9h8"
stream_url = get_stream_url(video_url)


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

    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(blur, kernel, iterations=1)

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

    classifier_path = "models/cars.xml"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

            # 1080 1920
        height, width = frame.shape[:2]

        output_size = (800, 800)

        scaling_factor = min(output_size[0]/width, output_size[1]/height)
        resized_image = cv2.resize(
            frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        processed, vehicles = detect_vehicles(resized_image, classifier_path)

        for (x, y, w, h) in vehicles:
            cv2.rectangle(resized_image, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Pre Processing', processed)
        cv2.imshow('Vehicle Detection', resized_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
