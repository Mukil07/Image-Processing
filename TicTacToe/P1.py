import cv2
import numpy as np
def empty(a):
   pass

paint= np.zeros((690,690,3))+255
cv2.namedWindow('paint')

# for knowing the points 
def click_event(event,x,y,flags,param):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(x,y)

# for drawing the lines
for i in range(2):
   a=230
   cv2.line(paint,(a,0),(a,690),(0,0,0),2)
   cv2.line(paint,(0,a),(690,a),(0,0,0),2)
   a+=230
   cv2.line(paint,(0,a),(690,a),(0,0,0),2)
   cv2.line(paint,(a,0),(a,690),(0,0,0),2)

cv2.imshow('points',paint)
# masking
cv2.namedWindow('controls1')
cv2.namedWindow('controls2')

#for yellow colour (Player 1)
cv2.createTrackbar('Hue_min','controls1',0,0,empty)
cv2.createTrackbar('Sat_min','controls1',129,129,empty)
cv2.createTrackbar('Val_min','controls1',183,183,empty)
cv2.createTrackbar('Hue_max','controls1',23,23,empty)
cv2.createTrackbar('Sat_max','controls1',255,255,empty)
cv2.createTrackbar('Val_max','controls1',255,255,empty)

#for pink colour (Player 2)
cv2.createTrackbar('Hue_min','controls2',158,158,empty)
cv2.createTrackbar('Sat_min','controls2',32,32,empty)
cv2.createTrackbar('Val_min','controls2',127,127,empty)
cv2.createTrackbar('Hue_max','controls2',195,195,empty)
cv2.createTrackbar('Sat_max','controls2',255,255,empty)
cv2.createTrackbar('Val_max','controls2',239,239,empty)

def findingContours(img):
   x,y,w,h=0,0,0,0
   contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
      area= cv2.contourArea(contour)
      if area>500:
         peri=cv2.arcLength(contour,True)
         approx=cv2.approxPolyDP(contour,0.02*peri,True)
         x,y,w,h= cv2.boundingRect(approx)  
   return x+w//2,y

win=[[(108,116),(346,109),(578,113)],[(110,345),(349,344),(579,347)],[(107,578),(347,581),(577,579)],[(108,116),(110,345),(107,578)],
     [(346,109),(349,344),(347,581)],[(578,113),(579,347),(577,579)],[(108,116),(349,344),(577,579)],[(578,113),(349,344),(107,578)]]
centres= [(108,116),(346,109),(578,113),(110,345),(349,344),(579,347),(107,578),(347,581),(577,579)] 
points1=[]
points2=[]
empty=[]
p1=[]
p2=[]
def drawing(points1,points2):
  
   for i in points1:
      w,h=-100,-100
      cv2.line(paint,(i[0]-50,i[1]-50),(i[0]+50,i[1]+50),(0,0,255),2)
      cv2.line(paint,(i[0]-50,i[1]+50),(i[0]+50,i[1]-50),(0,0,255),2)
   for i in points2:
      cv2.circle(paint,(i[0],i[1]),50,(255,0,0),3)

   
   
cap= cv2.VideoCapture(0)
while True:
   nextPoint1=[]
   nextPoint2=[]
   nextP1=[]
   nextP2=[]
   
   ret,frame= cap.read()
   frame= cv2.resize(frame,(690,690))
   hsvf= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   gray= cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
   
   h_min1=cv2.getTrackbarPos('Hue_min','controls1')
   h_max1=cv2.getTrackbarPos('Hue_max','controls1')
   s_min1=cv2.getTrackbarPos('Sat_min','controls1')

   s_max1=cv2.getTrackbarPos('Sat_max','controls1')
   v_min1=cv2.getTrackbarPos('Val_min','controls1')
   v_max1=cv2.getTrackbarPos('Val_max','controls1')

   h_min2=cv2.getTrackbarPos('Hue_min','controls2')
   h_max2=cv2.getTrackbarPos('Hue_max','controls2')
   s_min2=cv2.getTrackbarPos('Sat_min','controls2')

   s_max2=cv2.getTrackbarPos('Sat_max','controls2')
   v_min2=cv2.getTrackbarPos('Val_min','controls2')
   v_max2=cv2.getTrackbarPos('Val_max','controls2')


   

   lower1=np.array([h_min1,s_min1,v_min1])
   upper1=np.array([h_max1,s_max1,v_max1])

   lower2=np.array([h_min2,s_min2,v_min2])
   upper2=np.array([h_max2,s_max2,v_max2])
   
   maskf1=cv2.inRange(hsvf,lower1,upper1)
   resf1=cv2.bitwise_and(frame, frame, mask= maskf1)

   maskf2=cv2.inRange(hsvf,lower2,upper2)
   resf2=cv2.bitwise_and(frame, frame, mask= maskf2)
   
   x1,y1=findingContours(maskf1)
   x2,y2=findingContours(maskf2)
   
   r= 15
   a=0
   b=0

   # for filling the space once
   #for colour 1
   for i in centres:
      a=int(i[0])
      b=int(i[1])
      
      d= (x1-a)**2+(y1-b)**2-r**2
      if d<0:
         nextPoint1.append((x1,y1))
         nextP1.append((a,b))
         break
      else:
         a=0
         b=0
                
   if (a,b) in centres:  
      centres.remove((a,b))

   #for colour 2
   for i in centres:
      a=int(i[0])
      b=int(i[1])
      
      d= (x2-a)**2+(y2-b)**2-r**2
      if d<0:
         nextPoint2.append((x2,y2))
         nextP2.append((a,b))
         break
      else:
         a=0
         b=0
                
   if (a,b) in centres:  
      centres.remove((a,b))
      
   # appending the next point
   for i in nextPoint1:
      points1.append(i)
   for i in nextPoint2:
      points2.append(i)

   for i in nextP1:
      p1.append(i)
   for i in nextP2:
      p2.append(i)
   

      
   drawing(points1,points2)
     
   
   resf1= cv2.resize(resf1,(690,690))
   resf1= cv2.cvtColor(resf1,cv2.COLOR_HSV2BGR)
   resf2= cv2.resize(resf2,(690,690))
   resf2= cv2.cvtColor(resf2,cv2.COLOR_HSV2BGR)
   
   
   flag=0
   for i in win:
      
      if set(i).issubset((set(p1))):
         print('yes')
         cv2.destroyAllWindows()
         flag=1
         print("player 1 wins")
      if set(i).issubset((set(p2))):
         cv2.destroyAllWindows()
         flag=1
         print("player 2 wins")
   
   f=paint+resf1+resf2
   cv2.circle(f,(x1,y1),6,(0,255,0),2)
   cv2.circle(f,(x2,y2),6,(255,0,0),2)
   final= cv2.flip(f,1)

   
   #cv2.imshow('mask',maskf)
   cv2.imshow('img',final)
   cv2.setMouseCallback('img',click_event)

   if flag==1:
      break

   if cv2.waitKey(1) & 0xFF== ord('q'):
      break

cv2.waitKey(0)
cv2.destroyAllWindows()
