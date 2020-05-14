import cv2
import numpy as np

framewidth = 640
frameheight = 480
cap=cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(3, frameheight)
cap.set(10, 150)
nplateCascade = cv2.CascadeClassifier("C:\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_russian_plate_number.xml")
minarea =200


count = 0
while True:
    success, img=cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Nplates = nplateCascade.detectMultiScale(imgGray, 1.9, 4)
    for (x, y, w, h) in Nplates:
        area = w*h
        if area > minarea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0,255), 2)
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_TRIPLEX,1,(255,0,255),2)
            imgcrop=img[y:y+h,x:x+w]
            cv2.imshow("Cropped Number Plate",imgcrop)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        cv2.imwrite("C:\\Users\\vignesh\\Documents\\Scan_image" + str(count) + ".jpg", imgcrop)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1