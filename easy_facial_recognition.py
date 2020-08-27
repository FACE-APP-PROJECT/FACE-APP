# Code Anis - Defend Intelligence
import os
import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import face_recognition 
import argparse
import time  

parser = argparse.ArgumentParser()
parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()


print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importing pretrained model..')


def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces


def encode_face(image):
    face_locations = face_detector(image, 1)
    face_encodings_list = []
    landmarks_list = []
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = frame[:, :, ::-1]
    # ENCODING FACE
    face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)
        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        os.system('espeak "{}"'.format(name))

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)


if __name__ == '__main__':
    #args = parser.parse_args()

    print('[INFO] Importing faces...')
    '''face_to_encode_path = ['Zuckerberg.png','Shahrukh_khan.png','angelina.png','angelina.png']
    known_face_encodings = []
    for face_to_encode_path in face_to_encode_path:
        image = PIL.Image.open(face_to_encode_path)
        image = np.array(image)
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)
    known_face_names=['Zuckerberg','obama','angelina','angelina']'''
    image_of_angelina = face_recognition.load_image_file('./Known_faces/angelina.png')
    angelina_face_encoding = face_recognition.face_encodings(image_of_angelina)[0]

    image_of_erdogan = face_recognition.load_image_file('./Known_faces/erdogan.png')
    erdogan_face_encoding = face_recognition.face_encodings(image_of_erdogan)[0]

    image_of_obama = face_recognition.load_image_file('./Known_faces/obama.png')
    obama_face_encoding = face_recognition.face_encodings(image_of_obama)[0]

    image_of_priyanka = face_recognition.load_image_file('./Known_faces/priyanka.jpg')
    priyanka_face_encoding = face_recognition.face_encodings(image_of_priyanka)[0]

    image_of_trump = face_recognition.load_image_file('./Known_faces/trump.png')
    trump_face_encoding = face_recognition.face_encodings(image_of_trump)[0]

    image_of_Zuckerberg = face_recognition.load_image_file('./Known_faces/Zuckerberg.png')
    Zuckerberg_face_encoding = face_recognition.face_encodings(image_of_Zuckerberg)[0]

    image_of_Saad = face_recognition.load_image_file('./Known_faces/saad.jpg')
    Saad_face_encoding = face_recognition.face_encodings(image_of_Saad)[0]

    image_of_Bill = face_recognition.load_image_file('./Known_faces/bill.jpg')
    Bill_face_encoding = face_recognition.face_encodings(image_of_Bill)[0]

    image_of_Oprah = face_recognition.load_image_file('./Known_faces/oprah.jpg')
    Oprah_face_encoding = face_recognition.face_encodings(image_of_Oprah)[0]

    image_of_Ihssane = face_recognition.load_image_file('./Known_faces/ihssane.jpg')
    Ihssane_face_encoding = face_recognition.face_encodings(image_of_Ihssane)[0]

    image_of_Mohcine = face_recognition.load_image_file('./Known_faces/mohcine.jpg')
    Mohcine_face_encoding = face_recognition.face_encodings(image_of_Mohcine)[0]

    image_of_Taha = face_recognition.load_image_file('./Known_faces/taha.jpeg')
    Taha_face_encoding = face_recognition.face_encodings(image_of_Taha)[0]

    image_of_Omar = face_recognition.load_image_file('./Known_faces/omar.jpeg')
    Omar_face_encoding = face_recognition.face_encodings(image_of_Omar)[0]

    image_of_Rajaa = face_recognition.load_image_file('./Known_faces/rajaa.jpeg')
    Rajaa_face_encoding = face_recognition.face_encodings(image_of_Rajaa)[0]

    image_of_Kawtar_bamou = face_recognition.load_image_file('./Known_faces/kawtar_bamou.jpeg')
    Kawtar_bamou_face_encoding = face_recognition.face_encodings(image_of_Kawtar_bamou)[0]

    image_of_Asma = face_recognition.load_image_file('./Known_faces/asma.jpeg')
    Asma_face_encoding = face_recognition.face_encodings(image_of_Asma)[0]

    image_of_Ihab = face_recognition.load_image_file('./Known_faces/ihab.jpeg')
    Ihab_face_encoding = face_recognition.face_encodings(image_of_Ihab)[0]

    image_of_Safaa = face_recognition.load_image_file('./Known_faces/safaa.jpeg')
    Safaa_face_encoding = face_recognition.face_encodings(image_of_Safaa)[0]

    image_of_Kendji = face_recognition.load_image_file('./Known_faces/kendji.jpg')
    Kendji_face_encoding = face_recognition.face_encodings(image_of_Kendji)[0]

    image_of_Hitler = face_recognition.load_image_file('./Known_faces/hitler.jpg')
    Hitler_face_encoding = face_recognition.face_encodings(image_of_Hitler)[0]

    image_of_Elizabeth = face_recognition.load_image_file('./Known_faces/elizabeth.jpg')
    Elizabeth_face_encoding = face_recognition.face_encodings(image_of_Elizabeth)[0]

    image_of_Harry = face_recognition.load_image_file('./Known_faces/harry.jpg')
    Harry_face_encoding = face_recognition.face_encodings(image_of_Harry)[0]

    image_of_Meghan = face_recognition.load_image_file('./Known_faces/meghan.jpg')
    Meghan_face_encoding = face_recognition.face_encodings(image_of_Meghan)[0]
    
    image_of_Macron = face_recognition.load_image_file('./Known_faces/Macron.jpg')
    Macron_face_encoding = face_recognition.face_encodings(image_of_Macron)[0]

    image_of_Edouard= face_recognition.load_image_file('./Known_faces/edouard.jpg')
    Edouard_face_encoding = face_recognition.face_encodings(image_of_Edouard)[0]


    image_of_Angela_Merkel= face_recognition.load_image_file('./Known_faces/Angela_Merkel.jpg')
    Angela_Merkel_face_encoding = face_recognition.face_encodings(image_of_Angela_Merkel)[0]

    image_of_François_Hollande= face_recognition.load_image_file('./Known_faces/François_Hollande.jpg')
    François_Hollande_face_encoding = face_recognition.face_encodings(image_of_François_Hollande)[0]

    image_of_Vladimir_Putin= face_recognition.load_image_file('./Known_faces/Vladimir_Putin.jpg')
    Vladimir_Putin_face_encoding = face_recognition.face_encodings(image_of_Vladimir_Putin)[0]

    image_of_Boris_Johnson= face_recognition.load_image_file('./Known_faces/Boris_Johnson.jpg')
    Boris_Johnson_face_encoding = face_recognition.face_encodings(image_of_Boris_Johnson)[0]

    image_of_Hillary_Clinton= face_recognition.load_image_file('./Known_faces/Hillary_Clinton.jpg')
    Hillary_Clinton_face_encoding = face_recognition.face_encodings(image_of_Hillary_Clinton)[0]

    image_of_Michelle_Obama= face_recognition.load_image_file('./Known_faces/Michelle_Obama.jpg')
    Michelle_Obama_face_encoding = face_recognition.face_encodings(image_of_Michelle_Obama)[0]

    image_of_Kim_Jong_un= face_recognition.load_image_file('./Known_faces/Kim_Jong_un.jpg')
    Kim_Jong_un_face_encoding = face_recognition.face_encodings(image_of_Kim_Jong_un)[0]

    image_of_Charles= face_recognition.load_image_file('./Known_faces/Charles.jpg')
    Charles_face_encoding = face_recognition.face_encodings(image_of_Charles)[0]

    image_of_Demi_Lovato= face_recognition.load_image_file('./Known_faces/Demi_Lovato.jpeg')
    Demi_Lovato_face_encoding = face_recognition.face_encodings(image_of_Demi_Lovato)[0]

    image_of_Victor_Hugo= face_recognition.load_image_file('./Known_faces/Victor_Hugo.jpg')
    Victor_Hugo_face_encoding = face_recognition.face_encodings(image_of_Victor_Hugo)[0]

    image_of_Sam_Smith= face_recognition.load_image_file('./Known_faces/Sam_Smith.jpg')
    Sam_Smith_face_encoding = face_recognition.face_encodings(image_of_Sam_Smith)[0]

    image_of_Ali_Gatie= face_recognition.load_image_file('./Known_faces/Ali_Gatie.jpg')
    Ali_Gatie_face_encoding = face_recognition.face_encodings(image_of_Ali_Gatie)[0]

    image_of_Selena_Gomez= face_recognition.load_image_file('./Known_faces/Selena_Gomez.jpg')
    Selena_Gomez_face_encoding = face_recognition.face_encodings(image_of_Selena_Gomez)[0]

    image_of_Jeff_Bezos= face_recognition.load_image_file('./Known_faces/jeff-bezos.jpg')
    Jeff_Bezos_face_encoding = face_recognition.face_encodings(image_of_Jeff_Bezos)[0]

    image_of_Maher_Zain= face_recognition.load_image_file('./Known_faces/maher_zain.jpg')
    Maher_Zain_face_encoding = face_recognition.face_encodings(image_of_Maher_Zain)[0]

    image_of_Lionel_Messi= face_recognition.load_image_file('./Known_faces/Lionel_Messi.jpg')
    Lionel_Messi_face_encoding = face_recognition.face_encodings(image_of_Lionel_Messi)[0]

    image_of_Hakim_Ziyech= face_recognition.load_image_file('./Known_faces/Hakim_Ziyech.jpg')
    Hakim_Ziyech_face_encoding = face_recognition.face_encodings(image_of_Hakim_Ziyech)[0]

    image_of_Noureddine_Amrabat = face_recognition.load_image_file('./Known_faces/Noureddine_Amrabat.png')
    Noureddine_Amrabat_face_encoding = face_recognition.face_encodings(image_of_Noureddine_Amrabat)[0]

    image_of_Achraf_Hakimi= face_recognition.load_image_file('./Known_faces/Achraf_Hakimi.jpg')
    Achraf_Hakimi_face_encoding = face_recognition.face_encodings(image_of_Achraf_Hakimi)[0]

    image_of_Adham_Nabulsi= face_recognition.load_image_file('./Known_faces/Adham_Nabulsi.jpg')
    Adham_Nabulsi_face_encoding = face_recognition.face_encodings(image_of_Adham_Nabulsi)[0]

    image_of_Mohamed_Hamaki = face_recognition.load_image_file('./Known_faces/Mohamed_Hamaki.jpg')
    Mohamed_Hamaki_face_encoding = face_recognition.face_encodings(image_of_Mohamed_Hamaki)[0]

    image_of_Balqees_Fathi= face_recognition.load_image_file('./Known_faces/Balqees_Fathi.jpeg')
    Balqees_Fathi_face_encoding = face_recognition.face_encodings(image_of_Balqees_Fathi)[0]

    image_of_Sherine= face_recognition.load_image_file('./Known_faces/Sherine.jpg')
    Sherine_face_encoding = face_recognition.face_encodings(image_of_Sherine)[0]

    image_of_Melania= face_recognition.load_image_file('./Known_faces/Melania.jpg')
    Melania_face_encoding = face_recognition.face_encodings(image_of_Melania)[0]


    #  Create arrays of encodings and names
    known_face_encodings = [
        angelina_face_encoding,erdogan_face_encoding,obama_face_encoding,priyanka_face_encoding,
        trump_face_encoding,Zuckerberg_face_encoding, Saad_face_encoding, Bill_face_encoding, Oprah_face_encoding,
        Ihssane_face_encoding,Mohcine_face_encoding,Taha_face_encoding,Omar_face_encoding,Rajaa_face_encoding,
        Kawtar_bamou_face_encoding, Asma_face_encoding,Ihab_face_encoding,Safaa_face_encoding,Kendji_face_encoding,
        Hitler_face_encoding,Elizabeth_face_encoding,Harry_face_encoding, Meghan_face_encoding, Macron_face_encoding,
        Edouard_face_encoding,Angela_Merkel_face_encoding,François_Hollande_face_encoding,Vladimir_Putin_face_encoding,
        Boris_Johnson_face_encoding, Hillary_Clinton_face_encoding,Michelle_Obama_face_encoding,Kim_Jong_un_face_encoding,
        Charles_face_encoding,Demi_Lovato_face_encoding,Victor_Hugo_face_encoding,Sam_Smith_face_encoding,Ali_Gatie_face_encoding,
        Selena_Gomez_face_encoding,Jeff_Bezos_face_encoding,Maher_Zain_face_encoding,Lionel_Messi_face_encoding,
        Hakim_Ziyech_face_encoding,Noureddine_Amrabat_face_encoding,Achraf_Hakimi_face_encoding,Adham_Nabulsi_face_encoding,
        Mohamed_Hamaki_face_encoding,Balqees_Fathi_face_encoding,Sherine_face_encoding,Melania_face_encoding
    ]

    known_face_names = [
        "Angelina Jolie","Erdogan","Obama","Priyanka ","Trump","Zuckerberg","Saad","Bill","Oprah","Ihssane","Mohcine metouali",
        "Taha","Omar belmir","Rajaa belmir","Kawtar bamou","Asma","Ihab","Safaa","Kendji","Hitler","Elizabeth","Harry","Meghan",
        "Macron","Edouard","Angela Merkel","François_Hollande","Vladimir Putin","Boris Johnson","Hillary Clinton","Michelle Obama",
        "Kim Jong-un","Charles","Demi Lovato","Victor Hugo","Sam Smith","Ali Gatie","Selena Gomez","Jeff Bezos","Maher Zain",
        "Lionel Messi","Hakim Ziyech","Noureddine Amrabat","Achraf Hakimi","Adham Nabulsi","Mohamed Hamaki","Balqees Fathi","Sherine","Melania"
    ]
    print('[INFO] Faces well imported')
    print('[INFO] Starting Webcam or Importing image or video file...')
    video_capture = cv2.VideoCapture(args.i if args.i else 0)
    print('[INFO] Webcam well started or image or video file well imported')
    print('[INFO] Detecting...')
    while cv2.waitKey(1) < 0:
        t = time.time()
        ret, frame=video_capture.read()
        if not ret:
            cv2.waitKey()
            break    
        # Capture frame-by-frame
        easy_face_reco(frame, known_face_encodings, known_face_names)
        cv2.imshow('Easy Facial Recognition App', frame)
        #os.system('espeak "{}"'.format(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()
