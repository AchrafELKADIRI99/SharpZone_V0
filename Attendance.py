import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# define folder
path = 'ImagesAttendance'

# list of images
images = []

# names of people getting them from images filenames
classNames = []

# getting images
myList = os.listdir(path)

# importing them using opencv function
for cl in myList:
    currentImage = cv2.imread(f'{path}/{cl}')
    images.append(currentImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# function to encode all images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        # print(myDataList)
        for line in myDataList:
            #split them based on ,
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dateString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString}')





# encode all images
encodeListKnown = findEncodings(images)
print("Encoding Completed")

# initialize camera
cap = cv2.VideoCapture(0)

# send every frame (capture)
while True:
    success, img = cap.read()
    # resize pictures
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        # use numpy to know the most close value to encodes
        matchIndex = np.argmin(faceDis)

        # display box with name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('webcam', img)
    cv2.waitKey(1)

# # locate face1
# faceLoc1 = face_recognition.face_locations(imgElon1)[0]
# # encode face
# encodeElon1 = face_recognition.face_encodings(imgElon1)[0]
# # detection rectangle
# cv2.rectangle(imgElon1,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(255,0,255),2)
# # print(faceLoc)
# 
# # locate face
# faceLoc2 = face_recognition.face_locations(imgElon2)[0]
# # encode face
# encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
# # detection rectangle
# cv2.rectangle(imgElon2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)
# # print(faceLoc)
#
#
# #comparing
# results = face_recognition.compare_faces([encodeElon1], encodeElon2)
# faceDistance= face_recognition.face_distance([encodeElon1], encodeElon2)
