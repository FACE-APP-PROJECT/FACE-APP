# Blur Your Face Automatically with OpenCV and Dlib
# See tutorial at https://www.youtube.com/watch?v=QKggnWdCTNY

import numpy as np
import cv2
import dlib
import argparse
import time

def faceBlurring () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()

    cap=cv2.VideoCapture(args.i if args.i else 0)
    detector=dlib.get_frontal_face_detector()

    blurred = True

    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, frame=cap.read()
        if not ret:
            cv2.waitKey()
            break
        if (ret):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)

            for rect in rects:
                x = rect.left()
                y = rect.top()
                x1 = rect.right()
                y1 = rect.bottom()

                if blurred:
                    frame[y:y1, x:x1] = cv2.blur(frame[y:y1, x:x1], (25, 25))

        # Display the resulting frame
            cv2.imshow('Video Feed', frame)

        print("Time : {:.3f}".format(time.time() - t))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
    print('[INFO] Stopping System')
    cap.release()
    cv2.destroyAllWindows()

