import dlib
import cv2
from imutils import face_utils
import argparse
import time

def nbFaces() :    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.i if args.i else 0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
    dots_size = 1
    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, frame=cap.read()
        if not ret:
            cv2.waitKey()
            break
    
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
    # loop over the face detections
        for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
 
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # show the face number
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), dots_size, (255, 0, 0), -1)
        cv2.imshow("Output", frame)
        print("Time : {:.3f}".format(time.time() - t))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stopping System')
    cv2.destroyAllWindows()
