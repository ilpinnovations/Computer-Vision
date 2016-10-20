# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from collections import deque

#functions for calculating convex hull
def minDistance(contour, contourOther):
    distanceMin = 99999999
    for xA, yA in contour[0]:
        for xB, yB in contourOther[0]:
            distance = ((xB-xA)**2+(yB-yA)**2)**(1/2) # distance formula
            if (distance < distanceMin):
                distanceMin = distance
    return distanceMin


def find_if_close(cnt1,cnt2):

    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 40 :
                return True
            elif i==row1-1 and j==row2-1:
                return False



fgbg = cv2.BackgroundSubtractorMOG();
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="/home/tcs/Desktop/video5.mp4")
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
ap.add_argument("-b", "--buffer", type=int, default=5, help="max buffer size")
args = vars(ap.parse_args())

count=0;
count2=0;
counter=0

area = 0;

#case for reading from webcams
if args.get("video", None) is None:
	camera = cv2.VideoCapture(2)
	time.sleep(0.25)
 
#case for reading from file
else:
	camera = cv2.VideoCapture(args["video"])
 	
# initialize the first frame to store background as it is static
firstFrame = None
len1=1;


(grabbed, firstFrame) = camera.read();

firstFrame = imutils.resize(firstFrame, width=500)
gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (21, 21), 0)
firstFrame=gray;


framecount=0;
pts = [deque(maxlen=4) for i in range(1, args["buffer"])]
countup=0;
countdown=0;
[row,col]=firstFrame.shape[:2];



downcount=0; upcount=0;
temp_cY = 0; temp_cY2 = 0;



temp_prev=(0,0);
temp_cx=0;
temp_cy=0;

#looping over the framess
while True:
	
	framecount=framecount+1;
	flag=[0]*args["buffer"]
	
	unified = []

	
	(grabbed, frame) = camera.read()
	text = "Unoccupied"
	framecount=framecount+1;
	
 	#drawing line of indication
 	cv2.line(frame,(0,2*row-row/3),(col+1000,2*row-row/3),(0,0,255),2)
 	
 	(dx,dy) = (0,0)	
	
		

	frame = imutils.resize(frame, width=500)
	r,c=frame.shape[:2];
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 	thresh = fgbg.apply(frame)
	'''
	# compute the absolute difference between the current frame and
	# first frame
	'''
	frameDelta = cv2.absdiff(firstFrame, gray)
	print frameDelta;
	thresh = cv2.threshold(frameDelta, 75, 255, cv2.THRESH_BINARY)[1]
 
	
	thresh = cv2.dilate(thresh, None, iterations=2)
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	# loop over the contours
	LENGTH = len(cnts)
	
	numbercounters=0;
	for i,cnt1 in enumerate(cnts):
		if cv2.contourArea(cnt1) > 10000:
			numbercounters=numbercounters+1;
	
	if(numbercounters==0):
		
		firstFrame = frame
		firstFrame = imutils.resize(firstFrame, width=500)
		gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		firstFrame=gray;
	
	
	#if an object is present
	if(LENGTH!=0):

		status = np.zeros((LENGTH,1))
		
		for i,cnt1 in enumerate(cnts):
	    		x = i    
	    		if i != LENGTH-1:
				for j,cnt2 in enumerate(cnts[i+1:]):
		    			x = x+1
		    			dist = find_if_close(cnt1,cnt2)
		    			if dist == True:
		        			val = min(status[i],status[x])
		        			status[x] = status[i] = val
		    			else:
		        			if status[x]==status[i]:
		            				status[x] = i+1
	

		
		
		maximum = int(status.max())+1
		
		
		#to find convex hull
		for i in xrange(maximum):
	    		pos = np.where(status==i)[0]
	    		if pos.size != 0:
				cont = np.vstack(cnts[i] for i in pos)
				hull = cv2.convexHull(cont)
				unified.append(hull)

	
	
		cv2.drawContours(frame,unified,-1,(0,255,0),2)
		cv2.drawContours(thresh,unified,-1,255,-1)
		
	
	else:
		
		k=0;
		while(k<len(pts)):
			i=0;
			while(i< len(pts[k])):
				i = i+1;
			k=k+1;
	
	key=0; i=0;
	
	ic=0;
	
	for c in unified:
		
		ic=ic+1;
		# ignore small contours
		if cv2.contourArea(c) < args["min_area"]:
			continue 	  
		
 		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
 		
 		
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		
		center= cX,cY;
		#storing the tracked points
		pts[ic].append([center])
		
		cv2.circle(frame, (cX, cY), 1, (255, 255, 255), -1)
		
		
		
		
	flag=[0]*50;
	
	#looping over the tracked points
	
	for k in np.arange(1, len(pts)):
		flagcount=0;
		flag[k]=0;2
		if(k<len(unified)):
			area=cv2.contourArea(unified[k]);
		for i in np.arange(2, len(pts[k])):
			
			if pts[k][i - 1] is None or pts[k][i] is None:
				continue
	 
			
			if counter >= 10 and i == 1 and pts[k][-10] is not None:
				
				dX = pts[k][-10][0] - pts[k][i][0]
				dY = pts[k][-10][1] - pts[k][i][1]
				(dirX, dirY) = ("", "")
	 
				
				if np.abs(dY) > 20:
					dirY = "North" if np.sign(dY) == 1 else "South"
	 

				
	
	
	
	
			
			thickness=1;2
			tuplepoint=tuple(pts[k][i]);
			
			if(flagcount==0):
 				prev=tuplepoint[0];
 			flagcount=flagcount+1;
 			
 			
 			
 			
 			cX,cY=tuplepoint[0]
 			
			
			(cX2,cY2)=prev;
			dist=((cX-cX2)*(cX-cX2) + (cY-cY2)*(cY-cY2))**(0.5)
		
			
			if dist<=90:
				
				cv2.line(frame,(cX,cY),prev,(0,0,255),5)
				

				if(flag[k]==0):
						#checking whether the person crosses a line upwards
					if cY>2*row/3-1 and cY2<2*row/3+1:
						
						
						downcount=downcount+1;
						if(temp_cY != cY and temp_cY2 != cY2):
							temp_cY = cY
							temp_cY2 = cY2
							
							upcount = 0;
							
							cv2.line(frame,(0,2*row-row/3),(col+1000,2*row-row/3),(0,255,0),2)
							count=count+1;
							if area>5500:
								count=count+1;
							
						
							
						flag[k]=1;
						
						#checking whether the person crosses a line upwards
					elif cY2>2*row/3-1 and cY<2*row/3+1:
						
						
						upcount=upcount+1;
						if(temp_cY != cY and temp_cY2 != cY2):
							temp_cY = cY
							temp_cY2 = cY2
						
							downcount = 0;
							
							cv2.line(frame,(0,2*row-row/3),(col+1000,2*row-row/3),(0,255,0),2)
							count2=count2+1;
							if area>5500:
								count2=count2+1;
							
						flag[k]=1;
						
		
		
			prev=(cX,cY);
				
		
		
		text = "Occupied"
		
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		cv2.putText(frame, "downcount : {}".format(count), (125,125),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
		cv2.putText(frame, "upcount : {}".format(count2), (125,200),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
	# show the frame and record if the user presses a key
		cv2.imshow("Security Feed", frame)
		cv2.imshow("Thresh", thresh)
		cv2.imshow("Frame Delta", frameDelta)
		key = cv2.waitKey(27) & 0xFF
 		
	# if the `q` key is pressed, break from the lop
	
	if key == ord("q"):
		break
	
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
