import numpy as np
import cv2

cap = cv2.VideoCapture(0)
s, img = cap.read(0)

#img = cv2.imread('./tmp/putin.png')
  
face_cascade = cv2.CascadeClassifier('/Users/geb/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/geb/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/Users/geb/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_mcs_mouth.xml')
leftear_cascade = cv2.CascadeClassifier('/Users/geb/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_mcs_leftear.xml')
rightear_cascade = cv2.CascadeClassifier('/Users/geb/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_mcs_rightear.xml')
nose_cascade = cv2.CascadeClassifier('/Users/geb/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_mcs_nose.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.equalizeHist(gray, gray)
cv2.imwrite('./tmp/cap.png', gray)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

cv2.waitKey(0)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)

    roi_gray =  gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)

    mouth = mouth_cascade.detectMultiScale(roi_gray)
    if len(mouth) > 0:
        (mx,my,mw,mh) = mouth[0]
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),1)

    nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
    if len(nose) > 0:
        (nx,ny,nw,nh) = nose[0]
        cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),1)

lear = leftear_cascade.detectMultiScale(gray, 1.3, 5)
if len(lear) > 0:
    (nx,ny,nw,nh) = lear[0] 
    cv2.rectangle(img,(nx,ny),(nx+nw,ny+nh),(0,0,255),1)

rear = rightear_cascade.detectMultiScale(gray, 1.3, 5)
if len(rear) > 0:
    (nx,ny,nw,nh) = rear[0]
    cv2.rectangle(img,(nx,ny),(nx+nw,ny+nh),(0,0,255),1)
    
cv2.imwrite('./tmp/found.png', img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
