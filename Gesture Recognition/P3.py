import cv2
import numpy as np
import math

def empty(a):
   pass

def findHands():
   
   contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contours = max(contours, key=lambda x: cv2.contourArea(x))
   cv2.drawContours(frame, contours, -1, (255,255,0), 2)
   
   hull = cv2.convexHull(contours)
   cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
   #cv2.imshow("hull", frame)

   hull = cv2.convexHull(contours, returnPoints=False)
   defects = cv2.convexityDefects(contours, hull)
   

cap= cv2.VideoCapture(0)

cap.set(3,900)
cap.set(4,900)
cv2.namedWindow('controls')

cv2.createTrackbar('hsv_min','controls',0,0,empty)
cv2.createTrackbar('sat_min','controls',21,21,empty)
cv2.createTrackbar('val_min','controls',105,105,empty)
cv2.createTrackbar('hsv_max','controls',73,73,empty)
cv2.createTrackbar('sat_max','controls',130,130,empty)
cv2.createTrackbar('val_max','controls',255,255,empty)

sum1=0
sum2=0
total1=0



while True:
   ret, frame= cap.read()
   frame=cv2.flip(frame,1)
   roi1= frame[100:400,50:350]
   cv2.rectangle(frame,(50,100),(350,400),(255,0,0),1)

   roi2= frame[100:400,550:850]
   cv2.rectangle(frame,(550,100),(850,400),(255,0,0),1)
   
   hsvim1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
   hsvim2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
   kernel = np.ones((5,5))

   hsv_min= cv2.getTrackbarPos('hsv_min','controls')
   sat_min= cv2.getTrackbarPos('sat_min','controls')
   val_min= cv2.getTrackbarPos('val_min','controls')
   hsv_max= cv2.getTrackbarPos('hsv_max','controls')
   sat_max= cv2.getTrackbarPos('sat_max','controls')
   val_max= cv2.getTrackbarPos('val_max','controls')
   
   lower = np.array([hsv_min, sat_min, val_min])
   upper = np.array([hsv_max, sat_max, val_max])
   
   mask1 = cv2.inRange(hsvim1, lower, upper)
   mask2 = cv2.inRange(hsvim2, lower, upper)
   
   dilate1 = cv2.dilate(mask1,kernel)
   dilate2 = cv2.dilate(mask2,kernel)
   
   erode1 = cv2.erode(dilate1,kernel)
   erode2 = cv2.erode(dilate2,kernel)
   
   blurred1 = cv2.blur(erode1, (2,2))
   blurred2 = cv2.blur(erode2, (2,2))
   
   ret,thresh1 = cv2.threshold(blurred1,0,255,cv2.THRESH_BINARY)
   ret,thresh2 = cv2.threshold(blurred2,0,255,cv2.THRESH_BINARY)

   #cv2.imshow('thres',thresh1)

   contours1, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   if contours1 !=[]:
      contours1 = max(contours1, key=lambda x: cv2.contourArea(x))
   else:
      continue
   if contours2 !=[]:
      contours2 = max(contours2, key=lambda x: cv2.contourArea(x))
   else:
      continue
   
   hull1 = cv2.convexHull(contours1)
   cv2.drawContours(roi1, [hull1], -1, (0, 255, 255), 2)
   
   hull1 = cv2.convexHull(contours1, returnPoints=False)  
   defects1 = cv2.convexityDefects(contours1, hull1)
   if defects1 is not None:
      
      count1 = 1
      for i in range(defects1.shape[0]):
         # calculate the angle
         s, e, f, d = defects1[i][0]
         start = tuple(contours1[s][0])
         end = tuple(contours1[e][0])
         far = tuple(contours1[f][0])
         a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
         b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
         c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
         angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))# cosine theorem
         s=(a+b+c)/2
         area=math.sqrt(s*(s-a)*(s-b)*(s-c))
         error= (2*area)/a
         if angle <= np.pi/2 and error > 50:  # angle less than 45 degree, treat as fingers
            count1+= 1
            cv2.circle(roi1, far, 4, [0, 0, 255], -1)
      
      final=frame
      
   else:
      continue
   
   hull2 = cv2.convexHull(contours2)
   cv2.drawContours(roi2, [hull2], -1, (0, 255, 255), 2)
   
   hull2 = cv2.convexHull(contours2, returnPoints=False)  
   defects2 = cv2.convexityDefects(contours2, hull2)
   if defects2 is not None:
      
      count2 = 1
      for i in range(defects2.shape[0]):
         # calculate the angle
         s, e, f, d= defects2[i][0]
         start= tuple(contours2[s][0])
         end= tuple(contours2[e][0])
         far= tuple(contours2[f][0])
         a= np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
         b= np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
         c= np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
         angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))# cosine theorem
         s= (a+b+c)/2
         area= math.sqrt(s*(s-a)*(s-b)*(s-c))
         error= (2*area)/a
         if angle <= np.pi/2 and error > 50:  # angle less than 90 degree, treat as fingers
            count2+= 1
            cv2.circle(roi2, far, 4, [0, 0, 255], -1)
      
      final=frame
      
   else:
      continue
   
   #print(count1,count2)
   
   if cv2.waitKey(10) & 0xFF == 49:
      
      if count1 == count2:
         print("player 1 OUT!!")
         print("player 1 score-",sum1)
         total1= sum1
         sum1=0
      else:
         sum1+=count1
         
   
   if cv2.waitKey(10) & 0xFF == 50:

      if count1 == count2:
         
         if sum2== total1:
            print("draw match")
         else:
            print("player 2 score is ",sum2)
            print("player 1 won by ",total1-sum2," runs")
            break
         
      elif sum2 > total1:
            print("player 1 score is ",total1)
            print("player 2 won by ",sum2-total1," runs")
            break
      else:
         sum2+=count2
   
   cv2.imshow('final',final)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
