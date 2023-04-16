import face_recognition as fr
import cv2
import numpy as np
import os

imgAS = fr.load_image_file('img_path/Arijit Singh.jpg')
imgAS = cv2.cvtColor(imgAS, cv2.COLOR_BGR2RGB)
encodeAS = fr.face_encodings(imgAS)[0]
faceLoc = fr.face_locations(imgAS)[0]
cv2.rectangle(imgAS, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (200, 200, 200), 1)
cv2.imshow("Arijit Singh", imgAS)

imgMG = fr.load_image_file('img_path/Satyajeet.jpg')
imgMG = cv2.cvtColor(imgMG, cv2.COLOR_BGR2RGB)
encodeMG = fr.face_encodings(imgMG)[0]
faceLocMG = fr.face_locations(imgMG)[0]
cv2.rectangle(imgMG, (faceLocMG[3], faceLocMG[0]), (faceLocMG[1], faceLocMG[2]), (200, 200, 200), 1)
cv2.imshow("Mahatma Gandhi", imgMG)

distance = fr.face_distance([encodeMG], encodeAS)
result = fr.compare_faces([encodeMG], encodeAS)
print(distance, result)
cv2.putText(imgAS, f'{result} {round(distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

cv2.waitKey(0)
