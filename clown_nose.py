import cv2
import numpy as np
import dlib
from math import hypot
import argparse
import time


def clownNoseVideo () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()

    cap=cv2.VideoCapture(args.i if args.i else 0)

    nose_image = cv2.imread("images/fun/nose.png")
    ret, frame = cap.read()
    rows, cols, ret = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")

    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, frame=cap.read()
        if not ret:
            cv2.waitKey()
            break
        nose_mask.fill(0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray_frame)
        for face in faces:
            landmarks = predictor(gray_frame, face)

        # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1])*1.50)
            nose_height = int(nose_width * 0.96)

        # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                              int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                       int(center_nose[1] + nose_height / 2))


        # Adding the new nose
            nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose

        print("Time : {:.3f}".format(time.time() - t))
        cv2.imshow("Frame", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# When everything done, release the capture
    print('[INFO] Stopping System')
    cap.release()
    cv2.destroyAllWindows()

def clownNosePhoto(pathImg) :
    nose_image = cv2.imread("images/fun/nose.png")
    frame=cv2.imread(pathImg)
    rows, cols, ret = frame.shape
    nose_mask = np.zeros((rows, cols), np.uint8)
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
    tickmark=cv2.getTickCount()
    nose_mask.fill(0)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
    faces=detector(gray)
    t = time.time()
    for face in faces:
            landmarks = predictor(gray, face)

        # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1])*1.50)
            nose_height = int(nose_width * 0.96)

        # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                              int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                       int(center_nose[1] + nose_height / 2))


        # Adding the new nose
            nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose
 
# Display the output
    print("Time : {:.3f}".format(time.time() - t))
    cv2.imshow('Frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        print('[INFO] Stopping System')
        cv2.destroyAllWindows() 

def clownNose():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    args = parser.parse_args()
    argument=args.i if args.i else "0"
    List=argument.split(".")
    if List[-1]=='jpg' or List[-1]=='png' or List[-1]=="jpeg":
            clownNosePhoto(argument)
    else :
            clownNoseVideo() 


