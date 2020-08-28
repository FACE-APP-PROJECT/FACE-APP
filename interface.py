# Code KN-TEAM
from tkinter import *
import numpy as np
import cv2
import argparse
import black_glasses
import nbFaces
import clown_nose
import face_blurring
import face_swapping
import Eye_gaze
import emotionDetection
import detect_faces
import sys

root = Tk()
root.title('FACE APP')
root.iconbitmap('icon1.ico')

frame = LabelFrame(root, text="Choices...",padx=100, pady=30)
frame.pack(padx=20, pady=20)

def nbPersons():
	nbFaces.nbFaces()
	
def BlackGlasses () :
	black_glasses.glasses()

def faceDetection() :
	detect_faces.detectFaces()

def facePosition() :
	Eye_gaze.eyeGaze()

def EmotionDetection() :
	emotionDetection.emo()

def faceSwap() :
	face_swapping.faceSwapping()

def clownNose() :
	clown_nose.clownNose()

def faceBlur() :
	face_blurring.faceBlurring()

def close_window():
    root.destroy()

myButton1 = Button(frame, text="Face Detection", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=faceDetection, height = 1, width = 12).grid(row=0, column=0, pady=10)
myButton2 = Button(frame, text="Face position", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=facePosition, height = 1, width = 12).grid(row=1, column=0, pady=10)
myButton3 = Button(frame, text="Number of persons", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=nbPersons, height = 1, width = 12).grid(row=2, column=0, pady=10)
myButton4 = Button(frame, text="Clown's nose", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=clownNose, height = 1, width = 12).grid(row=3, column=0, pady=10)
myButton5 = Button(frame, text="Black Glasses", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=BlackGlasses, height = 1, width = 12).grid(row=4, column=0, pady=10)
myButton6 = Button(frame, text="Face swapping", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=faceSwap, height = 1, width = 12).grid(row=5, column=0, pady=10)
myButton7 = Button(frame, text="Face blurring", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=faceBlur, height = 1, width = 12).grid(row=6, column=0, pady=10)
myButton8 = Button(frame, text="Emotion Detection", padx=50,pady=10,bg="#044c8c",fg="#6eebfb",command=EmotionDetection, height = 1, width = 12).grid(row=7, column=0, pady=10)

bouton_quitter = Button(root, text="Exit", padx=50,pady=10,bg="gray",fg="#dbf6fa" ,command=close_window, height = 1, width = 12).pack()







root.mainloop()

