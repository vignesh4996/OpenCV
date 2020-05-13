import cv2
import numpy as np
##READING IMAGE
'''print("Packages installed")
img=cv2.imread("C:\\Users\\vicki\\Documents\\Passport photo.jpg")
cv2.imshow("MY PICTURE",img)
cv2.waitKey(0)
###READING VIDEO
vid=cv2.VideoCapture("C:\\Users\\vicki\\Pictures\\Camera Roll\\WIN_20200507_19_28_05_Pro.mp4")
while True:
    success, img=vid.read()
    cv2.imshow("video",img)
    if cv2.waitKey(2) & 0XFF ==ord('c'):
        break

##READING THROUGH WEBCAM
webcam=cv2.VideoCapture(0)
webcam.set(3,640)
webcam.set(4,480)
webcam.set(10,100)
while True:
    success, img=webcam.read()
    cv2.imshow("webacam",img)
    if cv2.waitKey(1) & 0XFF ==ord('c'):
        break
## Changing color
img=cv2.imread("C:\\Users\\vicki\\Documents\\Passport photo.jpg")

imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(imggray,(7,7),0)

cv2.imshow("GRAY IMG",imggray)
cv2.imshow("BLUR IMG",blur)

cv2.waitKey(0)

### EDGE DETECTOR (CANNY EDGE DETECTOR)
img=cv2.imread("C:\\Users\\vicki\\Documents\\Passport photo.jpg")
kernel=np.ones((5,5),np.uint8)
imgcanny=cv2.Canny(img,100,100)
imgdial=cv2.dilate(imgcanny,kernel,iterations=1)
imgerode=cv2.erode(imgdial,kernel,iterations=1)
cv2.imshow("CANNY IMG",imgcanny)
cv2.imshow("DIALATE IMG",imgdial)
cv2.imshow("ERODE IMG",imgerode)
cv2.waitKey(0)


########################  OPENCV CONVENTION ###############
##Image Resize
img=cv2.imread("C:\\Users\\vicki\\Documents\\Passport photo.jpg")
print(img.shape)

imgrs=cv2.resize(img,(200,150))
 imgcrop=img[0:100,50:125]
cv2.imshow("Resized Image",imgrs)
cv2.imshow("CROPPED IMG",imgcrop)
cv2.waitKey(0)
img =np.zeros((500,500,3),np.uint8)
img[:]=200,200,0
cv2.line(img,(5,5),(500,500),(255,0,0),3)
cv2.line(img,(475,5),(0,500),(255,0,0),3)
cv2.rectangle(img,(150,150),(350,350),(0,0,0),cv2.FILLED)
cv2.circle(img,(250,250),70,(255,255,255),3)
cv2.putText(img,"OPENCV",(300,100),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)
#cv2.imshow('test',img)
cv2.waitKey(0)
img=cv2.imread("C:\\Users\\vicki\\Desktop\\cards.jpg")
width,height = 250,350
pts1=np.float32([[111,219],[288,188],[154,482],[352,440]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgoutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("CROPP img",imgoutput)
cv2.imshow("CARDS",img)
#cv2.imshow("CARDS",matrix)
cv2.waitKey(0)

################## JOINING IMAGES ################
img=cv2.imread("C:\\Users\\vicki\\Desktop\\cards.jpg")
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgv=np.vstack((img,img))
imgh=np.hstack((img,imggray,img))

cv2.imshow("HORIZONTAL_CONNECT",imgh)
cv2.imshow("VERTICAL_CONNECT",imgv)
cv2.waitKey(0)

############# COLOR DETECTION ##############
def empty():
    pass
cv2.namedWindow("trackbars")
cv2.resizeWindow("trackbars", 640, 240)
cv2.createTrackbar("Hue Min","trackbars",0,179,empty)
cv2.createTrackbar("Hue Max","trackbars",20 ,179,empty)
cv2.createTrackbar("Sat Min","trackbars",110,255,empty)
cv2.createTrackbar("Sat Max","trackbars",255,255,empty)
cv2.createTrackbar("Val Min","trackbars",0,255,empty)
cv2.createTrackbar("Val Max","trackbars",255,255,empty)

while True:
    img = cv2.imread("C:\\Users\\vicki\\Desktop\\kambhi.jpg")
    imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "trackbars")
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imghsv,lower,upper)
    res=cv2.bitwise_and(img,imghsv,mask=mask)
    cv2.imshow('hsv',imghsv)
    cv2.imshow("HUE SAT VAL IMG",mask)
    cv2.imshow("result",res)

    cv2.imshow("ORIGINAL",img)
    cv2.waitKey(1)

########  Contours and shapes  ####################

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getcontours(img):
    contours,hierarchy =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgcont,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objcor=len(approx)
            x,y,w,h=cv2.boundingRect(approx)
            if objcor==3 : type="Tri"
            elif objcor==4:
                aspratio=w/h
                if aspratio >0.95 and aspratio<1.05: type ="square"
                else: type="Rectangle"
            else : type="None"
            cv2.rectangle(imgcont, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgcont, type,
                        (x + (w // 2), y + (h // 2)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 0), 1)





img = cv2.imread("C:\\Users\\vicki\\Desktop\\shapes.jpg")
imgcont=img.copy()
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgblur=cv2.GaussianBlur(imggray,(7,7),0.7)
imgcanny=cv2.Canny(img,100,75)
imgblank=np.zeros_like(img)
getcontours(imgcanny)
stackimg=stackImages(0.6,([img,imggray,imgblur],
                        [imgcanny,imgcont,imgblank]))
#cv2.imshow("Gray Image",imggray)
#cv2.imshow("Blur Image",imgblur)
cv2.imshow("Stack Image",stackimg)
cv2.waitKey(0)


############### FACE DETECTION   #################
img = cv2.imread("C:\\Users\\vicki\\Desktop\\faces.png")


faceCascade= cv2.CascadeClassifier("C:\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
#img = cv2.imread('Resources/lena.png')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(img,1.9,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("faces",img)
cv2.waitKey(0)

###

cascPath ="C:\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
#eyePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
#smilePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
#eyeCascade = cv2.CascadeClassifier(eyePath)
#smileCascade = cv2.CascadeClassifier(smilePath)

font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.putText(frame,'Face',(x, y), font, 2,(255,0,0),5)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()'''
orange = np.uint8([[[255,166,10 ]]])
hsv_orange = cv2.cvtColor(orange,cv2.COLOR_BGR2HSV)
print(hsv_orange)