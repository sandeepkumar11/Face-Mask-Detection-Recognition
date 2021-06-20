# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:25:46 2021

@author: Shubham Jhalani
"""
import cv2
import pickle
import face_recognition
import smtplib
from csv import writer


def take_photo(name):
    """
    This Function takes the photo of user and save it to database
    ----------
    input : String
            Name of the Person going to register
    return: None
    -------
    """
    webcam=cv2.VideoCapture(0)
    ret=False
    while True:
        ret,face=webcam.read()
        font = cv2.FONT_HERSHEY_DUPLEX
        text1="Press 's' take photo"
        cv2.putText(face,text1,(150,150),font, 1, (0,255,255),2)
        cv2.imshow('My Photo',face)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            #ret,face=webcam.read()
            filename = name+'.jpg'
            cv2.imwrite(filename,face)
            webcam.release()
            cv2.destroyAllWindows()
            return
        if cv2.waitKey(1) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break
def take_name():
    """
    To Get the Name of User
    """
    print("Enter Your Name")
    name=input()
    return name

def take_email():
    """
    To Get the Email of User
    """
    print("Enter Your Email")
    email=input()
    return email


def recognise_faces(known_face_encodings,known_face_names): 
    """
    This Function Used to recognise face in realtime webcam.

    Parameters
    ----------
    known_face_encodings : List
        encodings on which model is trained.
    known_face_names : List
        names on which model is trained.

    Returns: None
    -------
    None.

    """
    all_face_locations = []
    all_face_encodings = []
    all_face_names = [] 
    webcam_video_stream=cv2.VideoCapture(0)
    
    #loop through every frame in the video
    while True:
        #get the current frame from the video stream as an image
        ret,current_frame = webcam_video_stream.read()
        #resize the current frame to 1/4 size to proces faster
        current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
        #detect all faces in the image
        #arguments are image,no_of_times_to_upsample, model
        all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
        
        #detect face encodings for all the faces detected
        all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    
    
        #looping through the face locations and the face embeddings
        for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
            #splitting the tuple to get the four position values of current face
            top_pos,right_pos,bottom_pos,left_pos = current_face_location
            
            #change the position maginitude to fit the actual size video frame
            top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4
    
            #find all the matches and get the list of matches
            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
           
            #string to hold the label
            name_of_person = 'Unknown face'
            
            #check if the all_matches have at least one item
            #if yes, get the index number of face that is located in the first index of all_matches
            #get the name corresponding to the index number and save it in name_of_person
            if True in all_matches:
                first_match_index = all_matches.index(True)
                name_of_person = known_face_names[first_match_index]
            
            #draw rectangle around the face    
            cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
            
            #display the name as text in the image
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (0,255,255),1)
        
        #display the video
        cv2.imshow("Webcam Video",current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #release the stream and cam
    #close all opencv windows open
    webcam_video_stream.release()
    cv2.destroyAllWindows()


def find_face(face,known_face_encodings,known_face_names,known_face_emails):
    """
    
    Parameters
    ----------
    face : Image
        Image to recognise face.
    known_face_encodings : List
        List containing known faces encodings.
    known_face_names : List
        List containing known faces names.
    known_face_emails : List
        List containing known emails.

    Returns : List containing two element. First is name_of_person and other is email_of_person if recognised.
                Else list containing two elements Unknown Person and Unkonwn Email.
    ------
    This Function is Used to recognise faces in image.

    """
    #print(face)
    current_frame_small = cv2.resize(face,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    #print(all_face_locations)
    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    try:
        for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
            #splitting the tuple to get the four position values of current face
            top_pos,right_pos,bottom_pos,left_pos = current_face_location
            #change the position maginitude to fit the actual size video frame
            top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4
            
            #find all the matches and get the list of matches
            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
    
            #string to hold the label
            name_of_person = 'Unknown face'
            email= 'Unkonwn Email'            
            temp=[]
            temp.extend([name_of_person,email])
            #check if the all_matches have at least one item
            #if yes, get the index number of face that is located in the first index of all_matches
            #get the name corresponding to the index number and save it in name_of_person
            if True in all_matches:
                first_match_index = all_matches.index(True)
                temp[0] = known_face_names[first_match_index]
                temp[1] =  known_face_emails[first_match_index]
            # print(name_of_person)
            # print(email)
            return temp
        return ['Unknown_face','Unknown_email']
    except:
        return ['Unknown_face','Unknown_email']
    
def save_list(name,add):
    """
    This Function Saves the list into memory using pickle library.
   """
    with open('known_face_'+add, 'wb') as f:
        pickle.dump(name, f)


def extract_list(name,attribute):
    """
    Parameters
    ----------
    name : list
        name of list in which want to extract.
    attribute : string
        string having attribute to extract like encodings, names ,emails etc.

    Returns
    -------
    same as passed in argument name : List
        Return Required List
        
    This Function is Used to Extract the list
    """
    try:
        with open('known_face_'+attribute, 'rb') as f:    
            name = pickle.load(f)
        return name
    except:
        print("Error while in loading data")
        
def load_data():
    """
    Returns: List of List containg three list 
             known_face_names,known_face_encodings,known_face_emails respectively
    -------
    This function load the data into corresponding lists

    """
    known_face_names=[]
    known_face_encodings=[]
    known_face_emails=[]
    try:
        known_face_names=extract_list(known_face_names,'names')
        known_face_encodings=extract_list(known_face_encodings,'encodings')
        known_face_emails=extract_list(known_face_emails,'emails')
    except:
        print('Error in Loading Data')
    return [known_face_names,known_face_encodings,known_face_emails]

def save_data(known_face_names,known_face_encodings,known_face_emails):
    """
    Parameters
    ----------
    known_face_names : List
        List containing known faces names.
    known_face_encodings : List
        List containing known faces encodings.
    known_face_emails : List
        List containing known emails.

    Returns : None
    -------
    This Function save the list into memory.

    """
    save_list(known_face_names,'names')
    save_list(known_face_encodings,'encodings')
    save_list(known_face_emails,'emails')
    

def register():
    """
    This Function is used to register the person with its face.

    Returns: None
    -------

    """
    name=take_name()
    email=take_email()
    take_photo(name)
    known_face_names,known_face_encodings,known_face_emails = load_data()
    try:
        myimage = face_recognition.load_image_file(name+'.jpg')
        my_face_encodings =face_recognition.face_encodings(myimage)[0]
        known_face_encodings.append(my_face_encodings)
        known_face_names.append(name)
        known_face_emails.append(email)
        save_data(known_face_names,known_face_encodings,known_face_emails)
    except:
        print('Error in Finding Image')
        
        
def make_entry(filename,to_add):
    """
    This Function is Used to add data in file

    Parameters
    ----------
    filename : string
        FIlename in which want to add.
    to_add : List
        List of containing elements want to add.

    Returns
    -------
    None.

    """
    try:
        with open(filename, 'a',newline='') as f_object:
        # Pass this file object to csv.writer()
        # and get a writer object
            writer_object = writer(f_object)
          
            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(to_add)
          
            #Close the file object
            f_object.close()
    except:
        print('Error in File')

def send_email(Subject,msg,sender_email_addrs,sender_email_password,receiver_email_addrs):
    
    """
    Input(Subject: Subject of email, 
          msg: Message to be sent,
        sender_email_addrs: Email Address of Sender,
        sender_email_password: Password of Sender Email Address, 
        receiver_email_address: Gmail address of receiver in string)
    
    
    This Function is used to send mail from python using SMTP server using port 465
    """
    try:
        message=f'Subject: {Subject} \n\n {msg}'
        server=smtplib.SMTP_SSL('smtp.gmail.com',465)
        server.login(sender_email_addrs,sender_email_password)
        server.sendmail(sender_email_addrs,receiver_email_addrs,message)
        server.quit()
        return True
    except:
        print('Error in connecting Server')
        return False
