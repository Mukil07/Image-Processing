import cv2 
import numpy as np


# creating the kalman filter algo
kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

def Estimate( x, y):
        
   #To estimate the position of the object
   measured = np.array([[np.float32(x)], [np.float32(y)]])
   kf.correct(measured)
   predicted = kf.predict()
      
   return predicted

      
def findingContours(frame):
       
   x,y,w,h=0,0,0,0
   contours,_=cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
      area= cv2.contourArea(contour)

      if area>500:

         #cv2.drawContours(res,contour,-1,(255,0,0),3)
         peri=cv2.arcLength(contour,True)
         approx=cv2.approxPolyDP(contour,0.02*peri,True)
         x,y,w,h= cv2.boundingRect(approx)
         
   # Returning mid point of the blob
   return x+w/2,y+h/2,w,h
      

def Masking( frame):

   #Set threshold to filter only blue color
   lower= np.array([130,30,0])
   upper= np.array([255,255,90])
   mask= cv2.inRange( frame, lower, upper)
   
   # Dilate
   k= np.ones((10, 10))
   maskDilated = cv2.dilate(mask, k)
   maskC= cv2.bitwise_and(frame,frame,mask=maskDilated)
   maskC= cv2.resize(maskC,(800,800))
   
   return maskDilated,maskC

#Performs required image processing to get ball coordinated 
def drawing(frame,points):
   
   for i in range (1,len(points)):
      cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)

vid = cv2.VideoCapture('Adv-IP.mp4')
points=[]
predicted= np.zeros((2, 1), np.float32)
count =0

while(vid.isOpened()):
   ret, frame= vid.read()
   
   if(ret == True):
               
      mask,_= Masking(frame)
      _,maskC= Masking(frame)
      
      if cv2.waitKey(1) & 0xFF == ord('a'):
         count+=1   
         cv2.imwrite('ball'+str(count)+'.jpg',maskC)
         
      nextPoint=[]
      x,y,w,h= findingContours(mask)
      nextPoint.append((int(x),int(y)))
      for i in nextPoint:
         points.append(i)
                  
      predicted = Estimate(x, y)

      # avoiding the drawing of lines when ball disappears from the screen
      if (0,0) in points:
         points.remove((0,0))
         
      # disappearing of tracking line
      if len(points)>15:
         del points[0]
      drawing(frame,points)
               

      #Draw actual coords 
      cv2.circle(frame, (int(x), int(y)), 20, [0,0,255], 2, 7)
      cv2.line(frame, (int(x), int(y + 20)), (int(x + 50), int(y )), [0,0,0], 2,7)
      cv2.putText(frame, "Actual", (int(x + 50), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX,0.5, [0,0,255])

      #Draw Kalman Filter Predicted output
      cv2.circle(frame, (predicted[0], predicted[1]), 20, [255,0,0], 2, 7)
      cv2.line(frame, (predicted[0] + 16, predicted[1] - 15), (predicted[0] + 50, predicted[1] - 30), [0,0,0], 2, 7)
      cv2.putText(frame, "Predicted", (int(predicted[0] + 50), int(predicted[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,0,0])
      frame= cv2.resize(frame,(800,800))
      
      
         
      
      cv2.imshow('final',frame)

      if cv2.waitKey(10) & 0xFF == ord('q'):
         break

   else:
      break

initial= cv2.imread('initial.jpg')
pic1= cv2.imread('1.jpg')
pic2= cv2.imread('2.jpg')
pic3= cv2.imread('3.jpg')
pic4= cv2.imread('6.jpg')
pic5= cv2.imread('8.jpg')
pic= cv2.imread('cool12.jpg')
pic6= cv2.imread('10.jpg')
pic7= cv2.imread('11.jpg')
pic8= cv2.imread('cool13.jpg')

  
ret,thresh= cv2.threshold(pic1,3,255,cv2.THRESH_BINARY)
thresh_inv1= cv2.bitwise_not(thresh)
ret,thresh= cv2.threshold(pic2,3,255,cv2.THRESH_BINARY)
thresh_inv2= cv2.bitwise_not(thresh)
ret,thresh= cv2.threshold(pic3,3,255,cv2.THRESH_BINARY)
thresh_inv3= cv2.bitwise_not(thresh)

ret,thresh= cv2.threshold(pic4,5,255,cv2.THRESH_BINARY)
thresh_inv6= cv2.bitwise_not(thresh)

ret,thresh= cv2.threshold(pic5,5,255,cv2.THRESH_BINARY)
thresh_inv8= cv2.bitwise_not(thresh)

ret,thresh= cv2.threshold(pic,5,255,cv2.THRESH_BINARY)
thresh_inv9= cv2.bitwise_not(thresh)

ret,thresh= cv2.threshold(pic8,5,255,cv2.THRESH_BINARY)
thresh_inv7= cv2.bitwise_not(thresh)

ret,thresh= cv2.threshold(pic6,5,255,cv2.THRESH_BINARY)
thresh_inv10= cv2.bitwise_not(thresh)

ret,thresh= cv2.threshold(pic7,5,255,cv2.THRESH_BINARY)
thresh_inv11= cv2.bitwise_not(thresh)


img1= cv2.bitwise_and(initial,thresh_inv1)
img2= cv2.bitwise_and(img1,thresh_inv2)
img3= cv2.bitwise_and(img2,thresh_inv3)
img4= cv2.bitwise_and(img3,thresh_inv6)
img5= cv2.bitwise_and(img4,thresh_inv8)
img6= cv2.bitwise_and(img5,thresh_inv10)
img7= cv2.bitwise_and(img6,thresh_inv11)
img8= cv2.bitwise_and(img7,thresh_inv9)
img9= cv2.bitwise_and(img8,thresh_inv7)



final=img9+pic1+pic2+pic3+pic4+pic5+pic6+pic7+pic+pic8
#for i in range(1,12):
   #final= initial+cv2.imread(r'i')


cv2.imshow('final',final)
cv2.waitKey(0)


      
vid.release()
cv2.destroyAllWindows()
   


   

      
 




