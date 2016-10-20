
import numpy as np
import cv2
#import scipy

char_list_final=[];
def charExtract(s1):
	char_list=[];
	img = cv2.imread(s1,0)
	
	
	#Cropping the contact number part
	
	#img=img1[740:760,110:400])
	
	cv2.imshow("image_original",img)
	cv2.waitKey(0)
	r,c=img.shape[:2];

	ver=[0]*5000;
	i=1
	
	
	# Thresholding the background and intensifying the written part
	for i in range(1,c):
		for j in range(1,r):
			if(img[j,i]>220):
				img[j,i]=255;
			else:
				img[j,i]=0;

	
	countwhite=0;
	i=1;
	
	#Finding Vertical Projection
	for i in range(1,c):
		for j in range(1,r):
			ver[i]=ver[i]+255-img[j,i];


	
	chars=[0]*5000;
	
	
	#Extraction of Digits
	
	start=1; posc=1; i=1;
	while(i<=c):
		if(ver[i]>0):
			start=i;
			k=i;
			while k<=c:
		
				if(ver[k]>0):	
					k=k+1;
				else:
					break;
		
		
			end=k;
	
			chars[posc]=start;
			chars[posc+1]=end;
	
	
			i=end+1;
			posc=posc+2;
	
		else:
			i=i+1;


	
	i=1
	cnt=1
	
	
	#Counting number of digits ignoring small strips
	
	i=2; charcount=0;
	while(i<=posc):
		if(chars[i]-chars[i-1]<=10):
			i=i+2;
			continue;
		else:
			charcount=charcount+1;
			i=i+2;

	
	img_final=np.zeros(shape=(28,28*(charcount)))
	

	l=0
	




	BLACK = [0,0,0]

	
	itposc=1;
	while itposc<=posc-1:
		
		res_new=np.zeros(shape=(28,28))
		res_new.fill(0);
		if(chars[itposc+1]-chars[itposc]<=10):
			itposc=itposc+2;
			continue;
		img_demo = img[:,chars[itposc]:chars[itposc+1]];
		
		
		rchar,cchar=img_demo.shape[:2];
		
		
		rnew=20;
		cnew=2*(rnew*cchar/rchar);
		if(cnew <20):
			cnew = 20;
		
		
		
		
		
		
		
		
		i=1;j=1;j_min = 100000;

		while(i<=cchar-1):
			j=1;
			while(j<=rchar-1):
				if(img_demo[j,i]>0):
					j=j+1;
					continue;
				else:
					break;
			if(j<j_min):
				j_min=j;
			i=i+1;

		i=cchar-1;j=rchar-1;j_max = -100000;
		while(i>=1):
			j=rchar-1;
			while(j>=1):
				if(img_demo[j,i]>0):
					j=j-1;
					continue;
				else:
					break;
			if(j>j_max):
				j_max=j;
			i=i-1;
		img_demo = img_demo[j_min:j_max,:]
		
		
		
		#Resizing the image into 28 X 28
		img_demo = cv2.resize(img_demo,(cnew, rnew), interpolation = cv2.INTER_AREA)
		
		pi=0;
		pk=0;
		
		i=((28-rnew)/2);
		
		while(i<=((28+rnew)/2)-1):
			pi=pi+1;
			k=((28-cnew)/2);
			pk=0;
			while(k<=((28+cnew)/2)-1):
				pk=pk+1;
				
				res_new[i-1,k-1]=255-img_demo[pi-1,pk-1];
				k=k+1;
			i=i+1;
				
		
		
		#Concatenating to the final image
		while(k <=27):
			j=1
	
			while(j<=27):
				img_final[k,j+l] = res_new[k,j];
				j=j+1;
		
		
			k=k+1;
		
		
		
		
	
		#Conversion into list and appending into final list which will be returned finally
		char1 = res_new.tolist()
		
		it=0;
		
		
		char_list.append(char1);
		
		
		i=i+100
		k=1

		while(k <=27):
			j=1
	
			while(j<=27):
				img_final[k,j+l] = res_new[k,j];
				j=j+1;
		
		
			k=k+1;
	
		l=l+28;
		itposc=itposc+2;
		

	img_final_demo = img_final

	cv2.imshow("image_final",((img_final_demo).astype(np.uint8)))
	cv2.waitKey(0);
	cv2.destroyAllWindows();
	
	return char_list;
	
	


char_list_final=charExtract('/home/tcs/Desktop/digits15.jpg');







			
	
		
