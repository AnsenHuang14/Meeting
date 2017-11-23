import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from skimage import exposure
from skimage.transform import rescale, resize
import os
import cv2
import skimage.measure
import dicom,pylab
np.set_printoptions(threshold=np.inf)

# get color and grayscale image matrix by coordinate in dicom
# crop image by setting weight and height refer to roi four nodes
def dcm_read(path1='./dcm_data/003.dcm',path2='./dcm_data/003.dcm',w=260,h=180,num=0):
	loc = pd.read_csv(path2,sep=' ',header = None).as_matrix()
	loc[:,1]=loc[:,1]-60
	ds = dicom.read_file(path1) 
	im = ds.pixel_array.reshape(768,1024,3)
	# print (ds)
	crop_w = w
	crop_h = h

	color = im[60:618,377:743]
	gray = im[60:618,0:366]
	t = 0
	for image in [gray,color]:
		a = image

		left = loc[np.argmin(loc[:,0])]
		right = loc[np.argmax(loc[:,0])]
		top = loc[np.argmax(loc[:,1])]
		bottom = loc[np.argmin(loc[:,1])]
		weight = right[0]-left[0]
		height = top[1]-bottom[1]

		if crop_w==0 & crop_h==0:
			plus_w = 0
			remaind_w = 0
			plus_h = 0
			remaind_h = 0
		else:
			plus_w = int((crop_w-weight)/2)
			remaind_w = int((crop_w-weight)%2)
			plus_h = int((crop_h-height)/2)
			remaind_h = int((crop_h-height)%2)
		
		# print(left,right,top,bottom,crop_w,crop_h,weight,height)
		# for i in range(bottom[1]-plus_h,top[1]+plus_h+remaind_h):
		# 	a[i,left[0]-plus_w]=255
		# 	a[i,right[0]+plus_w+remaind_w]=255
		# for i in range(left[0]-plus_w,right[0]+plus_w+remaind_w):
		# 	a[bottom[1]-plus_h,i]=255
		# 	a[top[1]+plus_h+remaind_h,i]=255
		# for ind in loc:
		# 	a[ind[1],ind[0]]=255

		crop_row0 = (bottom[1]-plus_h)
		crop_row1 = (top[1]+plus_h+remaind_h)
		crop_col0 = (left[0]-plus_w)
		crop_col1 =  (right[0]+plus_w+remaind_w)

		if crop_row0<0:
			crop_row1 -= crop_row0
			crop_row0 = 0
		if crop_col0<0:
			crop_col1 -= crop_col0
			crop_col0 = 0
		
		a =a[crop_row0:crop_row1,crop_col0:crop_col1]
		if t == 0:scipy.misc.imsave('./dcm_data_image/gray_'+str(num)+'.png', a)
		else :scipy.misc.imsave('./dcm_data_image/color_'+str(num)+'.png', a)
		t+=1
		# plt.title(path2)
		# plt.imshow(a)
		# plt.show()

# crop all images  	
def crop_roi():
	a = 0 
	for i in range(1,267):
		if os.path.exists('./dcm_data/'+'{0:03}'.format(i)+'.dcm') &os.path.exists('./dcm_data/'+'{0:03}'.format(i)+'.dcm_Nodes.txt') :
			a+=1
			path1 = './dcm_data/'+'{0:03}'.format(i)+'.dcm'
			path2 = path1+'_Nodes.txt'
			dcm_read(path1,path2,300,300,i)

def get_label():
	y = pd.read_csv('./y.txt',sep='\t',header=None)
	y.columns = ['id','label']
	print(len(y))
# convert image to npy and range to 0,1
def construct_roi_data():
	t = 0
	data = np.array([0.0]*(300*300*263*4)).reshape(263,300,300,4)
	for i in range(1,267):
		print(i)
		if os.path.exists('./dcm_data_image/gray_'+str(i)+'.png'):
			im1 = scipy.misc.imread('./dcm_data_image/gray_'+str(i)+'.png', flatten=False,mode='L')
			im1 = resize(im1,(300,300)).reshape(300,300,1)
			im2 = scipy.misc.imread('./dcm_data_image/color_'+str(i)+'.png', flatten=False,mode='RGB')
			im2 = resize(im2,(300,300))
			im = np.concatenate([im1,im2],2)
			data[t] = im
			# plt.imshow(data[t,:,:,1:4].reshape(300,300,3),cmap='gray')
			# plt.show()
			t+=1
	data[:,:,:,0]/=255
	data[:,:,:,1]/=np.max(data[:,:,:,1])
	data[:,:,:,2]/=np.max(data[:,:,:,2])
	data[:,:,:,3]/=np.max(data[:,:,:,3])

	data = data.astype('float32')
	np.save('./dcm_data.npy',data)

def remove_dot_save(t):
	if os.path.exists('./dcm_data_image/gray_'+str(t)+'.png'):
			im1 = scipy.misc.imread('./dcm_data_image/gray_'+str(t)+'.png', flatten=False,mode='L')
			im1 = resize(im1,(300,300)).reshape(300,300,1)
			im2 = scipy.misc.imread('./dcm_data_image/color_'+str(t)+'.png', flatten=False,mode='RGB')
			im2 = resize(im2,(300,300))
			im = np.concatenate([im1,im2],2)
			threshold = 0.8
			threshold2 = 0.9
			# for row -------------------------------
			changeindex = list()
			for i  in range(300):
				im_tmp = im[i:(i+1),:,:]
				gray = 0.2989 * im[i:(i+1),:,1] + 0.5870 * im[i:(i+1),:,2] + 0.1140 * im[i:(i+1),:,3]
				if np.sum(im_tmp[:,:,0]-gray<0.03)/300<threshold :
					print(i,'row dotted line')
					changeindex.append(i)
			# -------------------
			# for i in range(changeindex[1]+2):
			# 		if i == changeindex[1] or i == changeindex[1]+1 :
			# 			im[i,:,0] = 0.2989 * im[i-2,:,1] + 0.5870 * im[i-2,:,2] + 0.1140 * im[i-2,:,3]
			# 		else: im[i,:,0] = 0.2989 * im[i,:,1] + 0.5870 * im[i,:,2] + 0.1140 * im[i,:,3]
			# -------------------
			if len(changeindex)==2 and changeindex[1]<150 :
				for i in range(changeindex[1]+2):
					if i == changeindex[1] or i == changeindex[1]+1 :
						im[i,:,0] = 0.2989 * im[i-2,:,1] + 0.5870 * im[i-2,:,2] + 0.1140 * im[i-2,:,3]
					else: im[i,:,0] = 0.2989 * im[i,:,1] + 0.5870 * im[i,:,2] + 0.1140 * im[i,:,3]
			if len(changeindex)==2 and changeindex[1]>150 :
				for i in range(changeindex[1],300):
					if i == changeindex[1] or i == changeindex[1]+1 :
						if i+2 >=299:
							im[i,:,0] = im[i-2,:,0] 
						else:
							im[i-1,:,0] = im[i+2,:,0] 
							im[i,:,0] = im[i+2,:,0] 
							# im[i,:,0] = 0.2989 * im[i+2,:,1] + 0.5870 * im[i+2,:,2] + 0.1140 * im[i+2,:,3]
					else: im[i,:,0] = 0.2989 * im[i,:,1] + 0.5870 * im[i,:,2] + 0.1140 * im[i,:,3]

			if len(changeindex)==4:
				for i in range(changeindex[1]+2):
					if i == changeindex[1] or i == changeindex[1]+1 :
						im[i,:,0] = 0.2989 * im[i-2,:,1] + 0.5870 * im[i-2,:,2] + 0.1140 * im[i-2,:,3]
					else: im[i,:,0] = 0.2989 * im[i,:,1] + 0.5870 * im[i,:,2] + 0.1140 * im[i,:,3]
				for i in range(changeindex[2],300):
					if i == changeindex[2] or i == changeindex[2]+1 :
						if i+2 >=299:
							im[i,:,0] = im[i-2,:,0] 
						else:
							im[i,:,0] = 0.2989 * im[i+2,:,1] + 0.5870 * im[i+2,:,2] + 0.1140 * im[i+2,:,3]
					else: im[i,:,0] = 0.2989 * im[i,:,1] + 0.5870 * im[i,:,2] + 0.1140 * im[i,:,3]
			
			
			# for column -----------------------------
			changeindex = list()
			for i  in range(300):
				im_tmp = im[:,i:(i+1),:]
				gray = 0.2989 * im[:,i:(i+1),1] + 0.5870 * im[:,i:(i+1),2] + 0.1140 * im[:,i:(i+1),3]
				if np.sum(im_tmp[:,:,0]-gray<0.03)/300<threshold2 :
					print(i,'column dotted line')
					changeindex.append(i)
			#--------------------
			# if i == changeindex[1] or i == changeindex[1]+1 :
			# 			if i+2>=299:
			# 				im[:,i,0] = im[:,i-2,0] 
			# 			else:
			# 				im[:,i,0] = 0.2989 * im[:,i+2,1] + 0.5870 * im[:,i+2,2] + 0.1140 * im[:,i+2,3]
			# 		else: im[:,i,0] = 0.2989 * im[:,i,1] + 0.5870 * im[:,i,2] + 0.1140 * im[:,i,3]
			#-------------------
			if len(changeindex)==2 and changeindex[1]<150 :
				for i in range(changeindex[1]+2):
					if i == changeindex[1] or i == changeindex[1]+1 :
						im[:,i,0] = 0.2989 * im[:,i-2,1] + 0.5870 * im[:,i-2,2] + 0.1140 * im[:,i-2,3]
					else: im[:,i,0] = 0.2989 * im[:,i,1] + 0.5870 * im[:,i,2] + 0.1140 * im[:,i,3]

			if len(changeindex)==2 and changeindex[1]>150 :
				for i in range(changeindex[1],300):
					if i == changeindex[1] or i == changeindex[1]+1 :
						if i+2>=299:
							im[:,i,0] = im[:,i-2,0] 
						else:
							im[:,i,0] = 0.2989 * im[:,i+2,1] + 0.5870 * im[:,i+2,2] + 0.1140 * im[:,i+2,3]
					else: im[:,i,0] = 0.2989 * im[:,i,1] + 0.5870 * im[:,i,2] + 0.1140 * im[:,i,3]


			if len(changeindex)==4:
				for i in range(changeindex[1]+2):
					if i == changeindex[1] or i == changeindex[1]+1 :
						im[:,i,0] = 0.2989 * im[:,i-2,1] + 0.5870 * im[:,i-2,2] + 0.1140 * im[:,i-2,3]
					else: im[:,i,0] = 0.2989 * im[:,i,1] + 0.5870 * im[:,i,2] + 0.1140 * im[:,i,3]
				for i in range(changeindex[2],300):
					if i == changeindex[2] or i == changeindex[2]+1 :
						if i+2>=299:
							im[:,i,0] = im[:,i-2,0] 
						else:
							im[:,i,0] = 0.2989 * im[:,i+2,1] + 0.5870 * im[:,i+2,2] + 0.1140 * im[:,i+2,3]
					else: im[:,i,0] = 0.2989 * im[:,i,1] + 0.5870 * im[:,i,2] + 0.1140 * im[:,i,3]
			gray_image = im[:,:,0].reshape(im.shape[0],im.shape[1])
			color_image = im[:,:,1:4].reshape(im.shape[0],im.shape[1],3)
			scipy.misc.imsave('./image_remove_dot/gray_'+str(t)+'.png',gray_image )
			scipy.misc.imsave('./image_remove_dot/color_'+str(t)+'.png', color_image)
			# print('save',t)
			# plt.imshow(gray_image,cmap='gray')
			# plt.show()
			# plt.imshow(color_image,cmap='gray')
			# plt.show()
if __name__ == '__main__':
	remove_dot_save(236)