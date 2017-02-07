import numpy as np
import cv2
import datetime
import time

cap = cv2.VideoCapture(0) # my webcam

template = cv2.imread('counter.png',0)
template2 = cv2.imread('6OClock.png', 0)
template3 = cv2.imread('12OClock.png', 0)
template4 = cv2.imread('3OClock.png', 0)
template5 = cv2.imread('9OClock.png', 0)

while(cap.isOpened()):
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#Counter
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

#12 O'Clock
    w3, h3 = template3.shape[::-1]

    res = cv2.matchTemplate(img_gray,template3,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)

    #time.sleep(1)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w3, pt[1] + h3), (0,0,255), 2)
        ret,frame = cap.read()
        
        flow_meter = frame[120:170, 210:500]
        cv2.imwrite('test_numbers.png', flow_meter)
        #time.sleep(1)

    cv2.imshow('Original', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
execfile('train_and_test.py')


