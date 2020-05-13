import cv2
import numpy as np


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
'''### the below code is used to find the color for hsv
def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

while True:

    _, img = cap.read()
    imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    print(h_min)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img,mask,result])
    #cv2.imshow('Original', img)
    #cv2.imshow('HSV Color Space', imgHsv)
    #cv2.imshow('Mask', mask)
   #cv2.imshow('Result', result)
    cv2.imshow('Horizontal Stacking', hStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
mycolors = ([87, 203, 71, 113, 255, 255],[0, 125, 91, 10, 248, 298], [47, 97, 90, 88, 255, 255])# In BGR format
myColorvalues = [[255, 51, 51],#Blue
               [0, 0, 255],#Red
               [0, 204, 0]]#Green
mypoints=[]
def findcolor(mycolors, img, myColorvalues):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newpoints = []
    for colors in mycolors:
        lower = np.array(colors[0:3])
        upper = np.array([colors[3:6]])
        mask = cv2.inRange(imghsv, lower, upper)

        x, y = getcontours(mask)
        cv2.circle(imgresult, (x, y), 15, myColorvalues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newpoints.append([x, y, count])
        count += 1
    return newpoints

def getcontours(img):
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgresult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y

def drawoncanvas(mypoints,myColorvalues):
    for point in mypoints:
        cv2.circle(imgresult,(point[0],point[1]),10,myColorvalues[point[2]],cv2.FILLED)

while True:
    success, img = cap.read()
    imgresult = img.copy()
    newpoints = findcolor(mycolors,img,myColorvalues)
    if len(newpoints) != 0:
        for newp in newpoints:
            mypoints.append(newp)
    if len(mypoints) != 0:
        drawoncanvas(mypoints,myColorvalues)
    cv2.imshow("Result", imgresult)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break