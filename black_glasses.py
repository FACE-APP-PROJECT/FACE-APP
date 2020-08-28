# Code KN-TEAM

import dlib
import cv2
import numpy as np
from scipy import ndimage
import argparse
import time

def resize(img, width):
    r = float(width) / img.shape[1]
    dim = (width, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

#Combine an image that has a transparency alpha channel
def blend_transparent(face_img, sunglasses_img):

    overlay_img = sunglasses_img[:,:,:3]
    overlay_mask = sunglasses_img[:,:,3:]
    
    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

#Find the angle between two points
def angle_between(point_1, point_2):
    angle_1 = np.arctan2(*point_1[::-1])
    angle_2 = np.arctan2(*point_2[::-1])
    return np.rad2deg((angle_1 - angle_2) % (2 * np.pi))



def glasses () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args = parser.parse_args()

    cap=cv2.VideoCapture(args.i if args.i else 0)
    glasses = cv2.imread("images/fun/glasses.png", -1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")

#Resize an image to a certain width
#Start main program
    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, img = cap.read()
        if not ret:
            cv2.waitKey()
            break
        img = resize(img, 700)
        img_copy = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
    # detect faces
            faces = detector(gray)
            for face in faces :
        #find face box bounding points
                x = face.left()
                y = face.top()
                w = face.right()
                h = face.bottom()

                dlib_rect = dlib.rectangle(int(x), int(y), int(w), int(h))

        ##############   Find facial landmarks   ##############
                detected_landmarks = predictor(gray, dlib_rect).parts()

                landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
        #landmarks = predictor(gray, face)
                for idx, point in enumerate(landmarks):
                    pos = (point[0, 0], point[0, 1])
                    if idx == 0:
                        eye_left = pos
                    elif idx == 16:
                        eye_right = pos

                    try:
                #cv2.line(img_copy, eye_left, eye_right, color=(0, 255, 255))
                        degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))

                    except:
                        pass

        ##############   Resize and rotate glasses   ##############

        #Translate facial object based on input object.

                eye_center = (eye_left[1] + eye_right[1]) / 2

        #Sunglasses translation
                glass_trans = int(.2 * (eye_center - y))

        #Funny tanslation
        #glass_trans = int(-.3 * (eye_center - y ))

        # Mask translation
        #glass_trans = int(-.8 * (eye_center - y))


        # resize glasses to width of face and blend images
                face_width = w - x

        # resize_glasses
                glasses_resize = resize(glasses, face_width)

        # Rotate glasses based on angle between eyes
                yG, xG, cG = glasses_resize.shape
                glasses_resize_rotated = ndimage.rotate(glasses_resize, (degree+90))
                glass_rec_rotated = ndimage.rotate(img[y + glass_trans:y + yG + glass_trans, x:w], (degree+90))


        #blending with rotation
                h5, w5, s5 = glass_rec_rotated.shape
                rec_resize = img_copy[y + glass_trans:y + h5 + glass_trans, x:x + w5]
                blend_glass3 = blend_transparent(rec_resize , glasses_resize_rotated)
                img_copy[y + glass_trans:y + h5 + glass_trans, x:x+w5 ] = blend_glass3
            print("Time : {:.3f}".format(time.time() - t))
            cv2.imshow('Output', img_copy)
        except:
            cv2.imshow('Output', img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stopping System')
    cap.release()
    cv2.destroyAllWindows()
    
