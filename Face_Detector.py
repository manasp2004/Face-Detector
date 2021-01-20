from random import randrange
import cv2
import os

trained_face_data = cv2.CascadeClassifier('Face_Data_Model.xml')

path = input("Enter path of Image: ")
os.path.normpath(path)

img = cv2.imread(path)

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# print(face_coordinates)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(255), randrange(255), randrange(255)), 2)

cv2.imshow("Face Detector", img)
cv2.waitKey()