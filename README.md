# Face-App

1- install Python : [Download Python](https://www.python.org/downloads/)

2- install Anaconda : [Download Anacoda](https://www.anaconda.com/) 

3- From the Start menu, search for and open "Anaconda Prompt"

4- Create Virtual Environment :Open the command prompt and execute the following command 


```
conda create --name myenv
```

5-  Activate the environment :


```
conda activate myenv
```
6- Install OpenCV, dlib and imutils... :
Continuing from the above prompt, execute the following commands



```
conda install -c conda-forge dlib
pip install opencv-contrib-python --upgrade
pip install imutils
conda install -c anaconda scikit-image
conda install -c anaconda scikit learn
```


7- Test your installation :
Open the python prompt on the command line by typing python on the command prompt



```
import cv2
import dlib
dlib.__version__
cv2.__version__
```

8- run the demo :

image : python interface.py -i images/test1.jpg

video : python interface.py -i videos/video.mp4

real time : python interface.py




