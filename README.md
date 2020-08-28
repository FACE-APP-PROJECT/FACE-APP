# Face-App

1- install Python : [Download Python](https://www.python.org/downloads/)

2- install Anaconda : [Download Anaconda](https://www.anaconda.com/) 

3- install Espeak : [Download Espeak](https://sourceforge.net/projects/espeak/) 

4- From the Start menu, search for and open "Anaconda Prompt"

5- Create Virtual Environment :Open the command prompt and execute the following command 


```
conda create --name myenv
```

6-  Activate the environment :


```
conda activate myenv
```
7- Install OpenCV, dlib and imutils... :
Continuing from the above prompt, execute the following commands



```
pip install opencv-python
pip install numpy
conda install -c conda-forge dlib
pip install opencv-contrib-python --upgrade
pip install imutils
conda install -c anaconda scikit-image
conda install -c anaconda scikit learn
pip install face-recognition
conda install -c akode face_recognition_models
```


8- Test your installation :
Open the python prompt on the command line by typing python on the command prompt



```
import cv2
import dlib
dlib.__version__
cv2.__version__
import face_recognition
```

9- run the demo :

### image : 

python interface.py -i images/test1.jpg

python easy_facial_recognition.py -i images/obama_michelle.jpg

### video : 

python interface.py -i videos/video.mp4

python easy_facial_recognition.py -i videos/test.mp4

### real time : 

python interface.py

python easy_facial_recognition.py






