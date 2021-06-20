import cv2
import numpy as np
import finder
import pickle
from keras.models import load_model
model=load_model("./model2-001.model")
import os
import csv
import datetime
from csv import writer
from csv import reader
import pandas as pd

filename='Not_Wearing_mask.csv'
to_add=[]
  
known_face_name,known_face_encoding,known_face_email=finder.load_data()
email_sent=False
Subject= 'Regarding Not Wearing Mask'
msg= ", we find that you are not wearing mask, It's request wear mask. \n Stay Safe Stay Healthy"
labels_dict={0:'without mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('C:/Users/shubham jhalani/Downloads/haarcascade_frontalface_default.xml')

while True:
    (rval, ima) = webcam.read()
    im=cv2.flip(ima,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    #print(im)
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(64,64))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,64,64,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        #print(result)
        
        label=np.argmax(result,axis=1)[0]
        
        namer=str()
        if label==0:
            namer,email=finder.find_face(ima,known_face_encoding,known_face_name,known_face_email)
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(im,namer, (x, y-40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        try:
            if label == 0 and email_sent== False:
                current_time=datetime.datetime.now()
                to_add.extend([namer,email,current_time])
                finder.make_entry(filename,to_add)
                msg= "Dear "+namer+msg
                sent=finder.send_email(Subject,msg,'admin@gmail.com','password','receiver@gmail.com')
                email_sent=True
                if sent== True:
                    print('Email Sent Successfully')
                else:
                    print('Error in Details')
        except:
            print('Check the details carefully')
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()