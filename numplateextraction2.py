import numpy as np
import cv2



def numplateExtract(s1,intensity):
	char_list=[];
	
	
	img = s1;
	
	factor = 0.00;
	#Generalized intensity depending upon the amount of light present
	if(intensity < 50.00):
		factor = 5.00
	elif(intensity >=50.00 and intensity<=75.00):
		factor = 2.5
	else:
		factor = 1.65
	r,c=img.shape[:2];

	ver=[0]*5000;
	i=1
	
	
	#thresholding background of numberplate
	for i in range(1,c):
		for j in range(1,r):
			if(img[j,i]>=intensity*factor):
				img[j,i]=255;
			else:
				img[j,i]=0;

	
	cv2.imshow("image_original",img)

	cv2.waitKey(0)
	
	
	
	
	areaplate=np.zeros(shape=(r*2,c*2))
	r1=r*2;
	c1=c*2;
	
	
	
	
	
	areaplate[r1/4:(r1/4)+r-3,c1/4:(c1/4)+c-3]=img[3:r,3:c];
	cv2.imshow("AreaPlate",areaplate);
	cv2.waitKey(0);
	
	
	#finding 4 points of the numberplate boundary for perspective projection
	
	j_min=10000000; i=0;
	while(i<=((c1-c)/2)+30):
		j=1;
		while(j<=r1-1):
			if(areaplate[j,i]!=255):
				j=j+1;
				continue;
			else:
				break;
		if(j<j_min):
			j_min=j;
			i_min=i;
		i=i+1;
	
	
	
	area_xf=i_min;
	area_yf=j_min;
	
	
	
	
	
	
	
	j_min=10000000; i=c1-1;
	while(i>=((c1)/2)):
		j=1;
		while(j<=r1-1):
			if(areaplate[j,i]!=255):
				j=j+1;
				continue;
			else:
				break;
		if(j<j_min):
			j_min=j;
			i_min=i;
		i=i-1;
	
	
	
	area_xf2=i_min;
	area_yf2=j_min;
	
	
	
	
	
	
	j_max=-10000000; i=0;
	while(i<=c1-1):
		j=r1-1;
		while(j>=0):
			if(areaplate[j,i]!=255):
				j=j-1;
				continue;
			else:
				break;
		if(j>j_max):
			j_max=j;
			i_max=i;
		i=i+1;
	
	area_xf3=i_max;
	area_yf3=j_max;
	
	
	
	
	j_max=-10000000; i=c1-1;
	print "CHECK ",(c1+c)/2-16;
	while(i>=(c1+c)/2-c1/10):
		j=r1-1;
		while(j>=0):
			
			if(areaplate[j,i]!=255):
				j=j-1;
				continue;
			else:
				break;
		if(j>j_max):
			j_max=j;
			print "JMAX ",j_max;
			i_max=i;
		i=i-1;
	
	
	area_xf4=i_max;
	area_yf4=j_max;
	
	print "area points 4", area_xf4,area_yf4;
	
	
	#perspective projection
	pts1 = np.float32([[area_xf,area_yf],[area_xf2,area_yf2],[area_xf3,area_yf3],[area_xf4,area_yf4]])
	pts2 = np.float32([[0,0],[c1,0],[0,r1],[c1,r1]])
	M = cv2.getPerspectiveTransform(pts1,pts2)
	 
	dst = cv2.warpPerspective(areaplate,M,(c1,r1))
	
	#final image showing the numberplate
	cv2.imshow("trans ", dst);
	cv2.waitKey(0);



img=cv2.imread("/home/tcs/Desktop/samplenumplate3.jpg",1);
cv2.imshow("Image",img);
cv2.waitKey(0);
r,c=img.shape[:2];
i=0;
j=0;
sum1 = 0;
count1=0;

kernel = np.ones((30,30),np.uint8)

#converting image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(c):
	for j in range(r):
		sum1 = sum1+gray[j,i]
		count1 = count1+1
	
intensity = float(float(sum1)/float(count1));

gray1 = cv2.GaussianBlur(gray, (21, 21), 0)


tophat1 = cv2.morphologyEx(gray1, cv2.MORPH_TOPHAT, kernel)

thresh1 = cv2.threshold(tophat1, float(float(intensity)/8), 255, cv2.THRESH_BINARY)[1]


#finding contours 

(cnts, hierarchy) = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

i=0;

max_area=-1;
for i in range(len(cnts)):
	
			
	
	
	(x, y, w, h) = cv2.boundingRect(cnts[i])
	
	
	#checking whether the specific area and aspect ratio is that of number plate
	if w*h >= 1000 and w*h <= 750000 and float(float(w)/float(h))>1.25 and float(float(w)/float(h))<5.00:
		
		
			if((w*h) > max_area):
				max_area = w*h;
				
				xf=x;
				yf=y;
				wf=w;
				hf=h;

		
cv2.rectangle(img, (xf, yf), (xf + wf, yf + hf), (0, 0, 255), 2)			
cv2.imshow("After Contours ",img[yf:yf+hf,xf:xf+wf]);
cv2.waitKey(0)





plate=img[yf:yf+hf,xf:xf+wf];


plate_gray=cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY);
cv2.imshow("Number Plate Gray ",plate_gray);
cv2.waitKey(0)

r,c=plate_gray.shape[:2];
i=1; k=1;
 












r,c=plate_gray.shape[:2];






img_final=numplateExtract(plate_gray,intensity);





