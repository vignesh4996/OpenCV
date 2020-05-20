import cv2
import pytesseract
## Tessaract accepts only RGB FORMAT
img = cv2.imread("C:\\Users\\vicki\\Desktop\\text.jpg")
print(img.shape)
imgh,imgw,_ = img.shape
'''
###Detecting Letters

pytesseract.pytesseract.tesseract_cmd= "C:\\Python\\Python38\\Tesseract-OCR\\tesseract.exe"
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#print(pytesseract.image_to_string(img))
boxes = pytesseract.image_to_boxes(img)
for box in boxes.splitlines():
    #print(box)
    box=box.split(' ')
    print(box)

    x,y,w,h=int(box[1]),int(box[2]),int(box[3]),int(box[4])
    print(x, y, w, h)
    cv2.rectangle(img, (x,imgh- y), (w,imgh- h), (50, 50, 255), 2)
    cv2.putText(img,box[0],(x,imgh- y+25), cv2.FONT_HERSHEY_TRIPLEX,1,(50,50,255),2)
    

cv2.imshow("Text", img)
cv2.waitKey(0)


#################  Detecting Words  ###############
pytesseract.pytesseract.tesseract_cmd= "C:\\Python\\Python38\\Tesseract-OCR\\tesseract.exe"
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

boxes = pytesseract.image_to_data(img)
for i,box in enumerate(boxes.splitlines()):
    if i!=0:
        #print(box)
        box=box.split()
        print(box)
        if len(box) == 12:
            x,y,w,h=int(box[6]),int(box[7]),int(box[8]),int(box[9])
            cv2.rectangle(img, (x,y), (w+x,y+ h), (50, 50, 255), 2)
            cv2.putText(img,box[-1],(x,y-20), cv2.FONT_HERSHEY_TRIPLEX,1,(50,50,255),2)


cv2.imshow("Text", img)
cv2.waitKey(0)
'''

############### Detecting Numbers ################
hImg, wImg,_ = img.shape
conf = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_boxes(img,config=conf)
for b in boxes.splitlines():
    print(b)
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
    cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)