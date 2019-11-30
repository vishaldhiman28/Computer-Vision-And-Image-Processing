from transform import four_point_transform
import numpy as np
import cv2
import imutils
import pytesseract
from pytesseract import Output
from PIL import Image

fn="2.JPG"
image = cv2.imread(fn)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#contours
cnts = cv2.findContours(th3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
screenCnt = approx
print(len(approx))


if len(approx)==4:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    deno=cv2.fastNlMeansDenoising(warped,None,9,13)
    img=deno
    overlay=deno.copy()

    # run tesseract, returning the bounding boxes
    data = pytesseract.image_to_data(img,output_type=Output.DICT)

    n_boxes=len(data['level'])
    for i in range(n_boxes):
        if data['text'][i]=='Merchant':
            #change Merchant, blue
            (x,y,w,h)=(data['left'][i],data['top'][i],data['width'][i],data['height'][i])
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(255,0,0), -1)
            img_new=cv2.addWeighted(overlay,0.4,img,0.6,0)
            
            #change id blue
            (x,y,w,h)=(data['left'][i+1],data['top'][i+1],data['width'][i+1],data['height'][i+1])
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(255,0,0), -1)
            img_new=cv2.addWeighted(overlay,0.4,img,0.6,0)
            
            #change Merchant ID value Red 
            (x,y,w,h)=(data['left'][i+2],data['top'][i+2],data['width'][i+2],data['height'][i+2])
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(0,0,255), -1)
            img_new=cv2.addWeighted(overlay,0.4,img,0.6,0)
            

            
        elif data['text'][i]=='Transaction':
            #change Transaction as blue
            (x,y,w,h)=(data['left'][i],data['top'][i],data['width'][i],data['height'][i])
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(255,0,0), -1)
            img_new=cv2.addWeighted(overlay,0.4,img,0.6,0)
            #id
            (x,y,w,h)=(data['left'][i+1],data['top'][i+1],data['width'][i+1],data['height'][i+1])
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(255,0,0), -1)
            img_new=cv2.addWeighted(overlay,0.4,img,0.6,0)
            
            #transaction id value Red
            (x,y,w,h)=(data['left'][i+2],data['top'][i+2],data['width'][i+2],data['height'][i+2])
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(0, 0,255), -1)
            img_new=cv2.addWeighted(overlay,0.4,img,0.6,0)


   
     
    cv2.imwrite("Problem2_modified.JPG",img_new)
    print("\n Modified image for problem 2 is as Problem2_modified.JPG")     


 
