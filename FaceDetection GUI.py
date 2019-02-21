#OpenCV is an image processing Library for C++ and Python. cv2 package is imported for OpenCV framework
import cv2
#Tkinter is a GUI framework for Python similar to swings in Java
import tkinter as tk
#FileDialog class contains methods related to File dialog
from tkinter import filedialog

#Face Recognition function is uses OpenCV framework
def facedetection(name) :
	#Prints version of OpenCV in console
    print(cv2.__version__)
    #imread is used to read image
    myimage=cv2.imread(name)
    #imread grayscale option removes saturation
    myimagegrey=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    #Type of myimage
    print(type(myimage))
    #myimage matrix
    print(myimage)
    #Size of myimage
    print(myimage.size)
    #Shape of myimage
    print(myimage.shape)
    #Dimension of myimage
    print(myimage.ndim)
    
    #Similar to the above
    print(type(myimagegrey))
    print(myimagegrey)
    print(myimagegrey.size)
    print(myimagegrey.shape)
    print(myimagegrey.ndim)
    #mycropimage=myimagegrey[:250]
    #myreverseimage=myimagegrey[::-1,::-1]
    #mycompressimage=myimagegrey[::4,::4]
    #cv2.rectangle(myimage,(194,6),(397,267),(0,0,255),4)
    
    #CascadeClassifier is used to classify using Haarcascade facedetection data i.e, face or not face
    face_learner=cv2.CascadeClassifier(r"C:\Users\PareeKatti\Anaconda3\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
    #detect multiscale is used to detect face, 10% of image is read at a time
    myface=face_learner.detectMultiScale(myimagegrey,1.1,5)
    #Myface matrix
    print(myface)
    #Number of faces
    print("number of faces")
    print(len(myface))
    #For loop is used to draw rectangle around the faces
    for(x,y,w,h) in myface :
        cv2.rectangle(myimage,(x,y),(x+w,y+h),(0,0,255),4)
    #it takes image,coordinate1,coordinate2,RGB,Thickness as arguments
    #display og image with rectangles
    
    #Slightly Profiled Face
    #CascadeClassifier is used to classify using Haarcascade facedetection data i.e, face or not face
    face_learner=cv2.CascadeClassifier(r"C:\Users\PareeKatti\Anaconda3\Library\etc\haarcascades\haarcascade_profileface.xml")
    #detect multiscale is used to detect face, 10% of image is read at a time
    myface=face_learner.detectMultiScale(myimagegrey,1.1,5)
    #Myface matrix
    print(myface)
    #Number of faces
    print("number of faces")
    print(len(myface))
    #For loop is used to draw rectangle around the faces
    for(x,y,w,h) in myface :
        cv2.rectangle(myimage,(x,y),(x+w,y+h),(0,0,255),4)
    
    #Eyes detection
    eye_learner=cv2.CascadeClassifier(r"C:\Users\PareeKatti\Anaconda3\Library\etc\haarcascades\haarcascade_eye.xml")
    #detect multiscale is used to detect eye, 10% of image is read at a time
    myeye=eye_learner.detectMultiScale(myimagegrey,1.6,10,minSize=(30,30))
    #Myeye matrix
    print(myeye)
    print(len(myeye))
    #For loop is used to draw rectangle around the eyes
    for(x,y,w,h) in myeye :
        cv2.rectangle(myimage,(x,y),(x+w,y+h),(0,255,0),2)
        
    #mouth detection
    mouth_learner=cv2.CascadeClassifier(r"C:\Users\PareeKatti\Anaconda3\Library\etc\haarcascades\haarcascade_mcs_mouth.xml")
    #detect multiscale is used to detect eye, 10% of image is read at a time
    mouth=mouth_learner.detectMultiScale(myimagegrey,1.9,9,minSize=(30,30))
    #Mouth matrix
    print(mouth)
    print(len(mouth))
    #For loop is used to draw rectangle around the mouth
    for(x,y,w,h) in mouth :
        cv2.rectangle(myimage,(x,y),(x+w,y+h),(255,0,0),2)     
        
    #nose detection
    nose_learner=cv2.CascadeClassifier(r"C:\Users\PareeKatti\Anaconda3\Library\etc\haarcascades\haarcascade_mcs_nose.xml")
    #detect multiscale is used to detect eye, 10% of image is read at a time
    nose=nose_learner.detectMultiScale(myimagegrey,1.9,5)
    #Nose matrix
    print(nose)
    print(len(nose))
    #For loop is used to draw rectangle around the nose
    for(x,y,w,h) in nose :
        cv2.rectangle(myimage,(x,y),(x+w,y+h),(255,255,0),2) 
    #Show Image
    cv2.imshow("MyImage",myimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#Main Flow
#Name stores the path of image
name=""
#Function to open file using filedialog class
def openfile() :
    global name
    name=filedialog.askopenfilename()
    print(name)
    facedetection(name)
#GUI using Tkinter
#Constructor
a=tk.Tk()
#Content of message box
Message='This a small Face Detection Script using Python and OpenCV library during "Artificial Intelligence and Machine Learning" workshop held at Indian Institute of Science (IISc) Bangalore.'
Message2='The code was revised and a GUI was added using Tkinter Framework for Python. '
Msg=Message+Message2
#Heading
tk.Label(a,text="Face Recognition using OpenCV",font="Verdana 20 bold",bg="#98a453").pack(pady=5,fill="x")
#Message
tk.Message(a,text=Msg,bg="#aaaaaa").pack()
           
#The Following Set of code is used to show the colors used for each facial feature           
tk.Label(a,text="Legend:",font="Verdana 10 bold",bg="#98a453").pack(pady=5,fill="x")
tk.Message(a,text="Face",bg="Red",fg="white").pack(fill='x')
tk.Message(a,text="Eyes",bg="Green",fg="white").pack(fill='x')
tk.Message(a,text="Mouth",bg="Blue",fg="white").pack(fill='x')
tk.Message(a,text="Nose",bg="Cyan").pack(fill='x')
#Button. On Click, it'll open a file dialog to open image
tk.Button(a,text="Browse for the image",font="Verdana 10 bold",bg="#98a453",command=openfile).pack(fill="x",pady=5)
#GUI should be looped till terminated
a.mainloop()