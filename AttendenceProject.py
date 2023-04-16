import face_recognition as fr
import cv2
import numpy as np
import os
from datetime import datetime

path = 'Images'                  # Path of known faces folder
images = []                      # Empty list for storing face images path
classNames = []                  # Empty list for storing face names
myList = os.listdir(path)        # List containing known faces
# print(myList)

for img in myList:                                 # Iterating over mylist
    currImg = cv2.imread(f'{path}/{img}')          # Current image
    images.append(currImg)                         # appending image path in images
    classNames.append(os.path.splitext(img)[0])    # appending face names
# print(classNames)


def findEncodings(images):                        # function of get encodings of all known faces
    encodeList = []                               # empty list to store encodings of known faces
    for i in images:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(i)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
# print(encodeListKnown)
print("Encodings Complete.")

cap = cv2.VideoCapture(0)                          # setting up webcam
print("opening webcam")

while True:
    success, img = cap.read()                            # success is boolean and is true if img gets an image
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)     # resizing image to reduce time used
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)         # BRG -> RGB

    facesInCurrFrame = fr.face_locations(imgS)           # getting locations of all the faces in the frame
    encodingsInCurrFrame = fr.face_encodings(imgS, facesInCurrFrame)  # encoding all faces in the frame using the
#                                                                     # locations

    for encodeFace, facesLoc in zip(encodingsInCurrFrame, facesInCurrFrame):
        '''
        takes both face locations and face encodings of each location from facesInCurrFrame and encodingsInCurrFrame 
        respectively.
        '''
        matches = fr.compare_faces(encodeListKnown, encodeFace)  # comparing faces in known list with curr image
        faceDis = fr.face_distance(encodeListKnown, encodeFace)  # finding distance
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = facesLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', img)
    cv2.waitKey(1)
