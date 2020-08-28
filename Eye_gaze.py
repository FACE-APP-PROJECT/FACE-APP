# Code KN-TEAM
import cv2
from gaze_tracking import GazeTracking
import argparse
import time
import dlib
import numpy as np

def eyeGaze():
    gaze = GazeTracking()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.i if args.i else 0)

    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, frame=cap.read()
        if not ret:
            cv2.waitKey()
            break

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces=detector(gray)
        if faces is not None:
            i=np.zeros(shape=(frame.shape), dtype=np.uint8)
        for face in faces:
    # We send this frame to GazeTracking to analyze it
            gaze.refresh(frame)

            frame = gaze.annotated_frame()
            text = ""

            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"
            left=face.left()
            top=face.top()
            right=face.right()
            bottom=face.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (147, 58, 31), 2)
            cv2.rectangle(frame, (left, bottom - 10), (right, bottom), (147, 58, 31), cv2.FILLED)
            cv2.putText(frame,text , (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        #cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

        '''left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)'''

        cv2.imshow("Demo", frame)

        print("Time : {:.3f}".format(time.time() - t))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('[INFO] Stopping System')
    cap.release()
    cv2.destroyAllWindows()

