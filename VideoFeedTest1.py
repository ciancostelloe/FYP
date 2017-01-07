import numpy as np
import cv2
import datetime

cap = cv2.VideoCapture(0) # my webcam

template = cv2.imread('counter.png',0)
template2 = cv2.imread('6OClock.png', 0)
template3 = cv2.imread('12OClock.png', 0)
template4 = cv2.imread('3OClock.png', 0)
template5 = cv2.imread('9OClock.png', 0)

while(True):
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Counter
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)
#6 O'Clock
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

    w2, h2 = template2.shape[::-1]

    res = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)

#12 O'Clock
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w2, pt[1] + h2), (0,255,255), 2)

    w3, h3 = template3.shape[::-1]

    res = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w3, pt[1] + h3), (0,0,255), 2)

#3 O'Clock
    w4, h4 = template4.shape[::-1]

    res = cv2.matchTemplate(img_gray,template4,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w4, pt[1] + h4), (255,255,255), 2)


    w5, h5 = template5.shape[::-1]

    res = cv2.matchTemplate(img_gray,template5,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w5, pt[1] + h5), (255,0,255), 2)

    gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    
    #laplacian = cv2.flip(laplacian,1)

    #currTime = datetime.datetime.now()
    #print(currTime)
    test = 'test'
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,test,(50,450), font, 3,(255,255,255),4)

    cv2.imshow('Original', frame)
    #cv2.imshow('frame',laplacian)
    cv2.imshow('GrayScale',img_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
