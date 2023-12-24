import cv2
import numpy as np

cap= cv2.VideoCapture(1)
Timg= cv2.imread('pad.jpg')
vid= cv2.VideoCapture('Adv-IP.mp4')

ret, vidF= vid.read()
Timg=cv2.resize(Timg,(640,480))
vidF= cv2.resize(vidF,(640,480))
h,w,c=Timg.shape
#vidF= cv2.resize(vidF,(w,h))
cv2.imshow('img2',vidF)
cv2.waitKey(0)

orb= cv2.ORB_create(nfeatures=500)
kp1, des1= orb.detectAndCompute(Timg,None)
Timg= cv2.drawKeypoints(Timg,kp1,None)
#cv2.imshow('img',Timg)



while True:
   ret, frame= cap.read()
   
   
   final = frame.copy()
   
   kp2, des2= orb.detectAndCompute(frame,None)
   frame= cv2.drawKeypoints(frame,kp2,None)

   bf= cv2.BFMatcher()
   matches= bf.knnMatch(des1,des2,k=2)
   good=[]
   for m,n in matches:
      if m.distance< 0.75*n.distance:
         good.append(m)

   features= cv2.drawMatches(Timg,kp1,frame,kp2,good,None,flags=2)
   cv2.imshow('final',final)
   
   if len(good)>15:
      
      suc, vidF= vid.read()
      if vidF is not None:
         
         vidF=cv2.resize(vidF,(640,480))   
         src= np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
         dst= np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                            
         matrix,_= cv2.findHomography(src,dst,cv2.RANSAC,5)
         
         pts= np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
         null= []
         dstf= cv2.perspectiveTransform(pts,matrix)
         img= cv2.polylines(frame,[np.int32(dstf)],True,(255,0,0),3)
            
         warp= cv2.warpPerspective(vidF,matrix,(frame.shape[1],frame.shape[0]))

         mask= np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
         cv2.fillPoly(mask,[np.int32(dstf)],(255,255,255))
         maskInv= cv2.bitwise_not(mask)
               
         final= cv2.bitwise_and(final,final,mask=maskInv)
         final= cv2.bitwise_or(final,warp)

         #cv2.imshow('img',maskInv)
         cv2.imshow('final',final)
      else:
         break

   if cv2.waitKey(10) & 0xFF == ord('q'):
         break

cv2.destroyAllWindows()
                   
