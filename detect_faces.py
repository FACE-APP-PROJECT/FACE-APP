import cv2
import dlib
import argparse
import time

def detectFaces() :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()

    cap=cv2.VideoCapture(args.i if args.i else 0)
    detector=dlib.get_frontal_face_detector()

    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, frame=cap.read()
        if not ret:
            cv2.waitKey()
            break
        tickmark=cv2.getTickCount()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces=detector(gray)
        for face in faces:
            x1=face.left()
            y1=face.top()
            x2=face.right()
            y2=face.bottom()
            roi_color = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        fps=cv2.getTickFrequency()/(cv2.getTickCount()-tickmark)
        cv2.imshow("Frame", frame)
        print("Time : {:.3f}".format(time.time() - t))
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            break

    print('[INFO] Stopping System')
    cap.release()
    cv2.destroyAllWindows()

