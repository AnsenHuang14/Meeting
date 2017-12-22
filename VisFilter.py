import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
import scipy
from keras import backend as K

def deprocessimage(x):
	"""
	Hint: Normalize and Clip
	"""
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	if K.image_data_format() == 'channels_first':
		x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	#x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x

def sensitivity(y_true, y_pred):
	true_number = K.sum(y_true)
	pred = K.round(y_pred)
	TP = K.sum(pred*y_true)
	return TP/true_number

def specificity(y_true, y_pred):
	pred = K.round(y_pred)
	pred = K.abs(pred-1)
	transpose_false = K.abs(y_true-1)
	false_number = K.sum(transpose_false)
	TN = K.sum(pred*transpose_false)
	return TN/false_number

def ss(y_true, y_pred):
	true_number = K.sum(y_true)
	pred = K.round(y_pred)
	TP = K.sum(pred*y_true)
	sens = TP/true_number
	pred2 = K.abs(pred-1)
	transpose_false = K.abs(y_true-1)
	false_number = K.sum(transpose_false)
	TN = K.sum(pred2*transpose_false)
	spec = TN/false_number
	return K.max([sens,spec])*0.2+K.min([sens,spec])*0.8

def vis_img_in_filter(img_dim,layer_dict,model,
					  layer_name ,channel,img=None,usingimg=False,):
	layer_output = layer_dict[layer_name].output
	img_ascs = list()
	loss_list = list()
	for filter_index in range(layer_output.shape[3]):
		# build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		loss = K.mean(layer_output[:, :, :, filter_index])

		# compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, model.input)[0]

		# normalization trick: we normalize the gradient
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

		# this function returns the loss and grads given the input picture
		iterate = K.function([model.input], [loss, grads])

		# step size for gradient ascent
		step = 5.

		if usingimg==False:
			img_asc = np.random.random((1, img_dim[0],img_dim[1],channel))
			# img_asc = (img_asc - 0.5) * 20 + img_dim[0]
		else: img_asc = np.array(img)
		
		
		# run gradient ascent for 20 steps
		print('filter:',filter_index)
		for i in range(20):
			# print('step',i,filter_index)
			loss_value, grads_value = iterate([img_asc])
			img_asc = img_asc.astype('float32')
			img_asc += grads_value.astype('float32') * step
			
		img_asc = img_asc[0]
		img_ascs.append(deprocess_image(img_asc).reshape((img_dim[0],img_dim[1],channel)))
		loss_list.append(loss_value)
	# print(loss_list)
	idx   = np.flip(np.argsort(loss_list),0)
	# print(type(img_ascs),idx)
	img_ascs = [ img_ascs[i] for i in idx]
	if layer_output.shape[3] >= 35:
		plot_x, plot_y = 6, 6
	elif layer_output.shape[3] >= 23:
		plot_x, plot_y = 4, 6
	elif layer_output.shape[3] == 16:
		plot_x, plot_y = 4, 4	
	elif layer_output.shape[3] >= 11:
		plot_x, plot_y = 2, 6
	else:
		plot_x, plot_y = int(int(layer_output.shape[3])/2), 2
	fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
	if usingimg:
		#--gray mode---
		ax[0, 0].imshow(img[:,:,:,0].reshape((img_dim[0], img_dim[1])),cmap='gray')
		#---elastography
		# ax[0, 0].imshow(img[:,:,:,1:4].reshape((img_dim[0], img_dim[1],3)))
		# ax[0, 0].imshow(img.reshape((img_dim[0], img_dim[1],channel)))
		ax[0, 0].set_title('Input image')
	fig.suptitle('Input image and %s filters' % (layer_name,))
	fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
	t = 0
	for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
		if usingimg and x == 0 and y == 0:
		    continue
		if t<int(layer_output.shape[3]):
			#--gray mode---
			ax[x, y].imshow(img_ascs[x * plot_y + y ][:,:,0],cmap='gray')
			#---elastography
			# ax[x, y].imshow(img_ascs[x * plot_y + y ][:,:,1:4])
			# ax[x, y].imshow(img_ascs[x * plot_y + y ])
			ax[x, y].set_title('filter %d' % (x * plot_y + y ))
		# else :
		# 	ax[x, y].imshow(img_ascs[x * plot_y + y-1 ])
		# 	ax[x, y].set_title('filter %d' % (x * plot_y + y-1 ))
		t+=1
	fig.savefig(layer_name+'.png', dpi=100)

def get_layer_output(input_X,input_layer,output_layer):
		get_output = K.function([input_layer.input],[output_layer.output])
		layer_output = get_output([input_X])[0]
		return layer_output

def plotoutput(img,model,num):
	out_list = get_layer_output(img,model.layers[0],model.layers[11])[0]
	plot_x = 4
	plot_y = 4
	fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))

	fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
	t = 0
	for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
		#--gray mode---
		ax[x, y].imshow(out_list[:,:,t],cmap='gray')
		#---elastography
		# ax[x, y].imshow(img_ascs[x * plot_y + y ][:,:,1:4])
		ax[x, y].set_title('filter %d' % (x * plot_y + y ))
		t+=1
	fig.savefig('./Meeting12-22/forward_path_output/'+str(num)+'.png', dpi=100)

def main():
	X = np.load('./Data/dcm_data.npy')
	img = X[1,:,:,:].reshape(1,300,300,4)
	K.set_learning_phase(1)

	model_name = "model"
	model_path = "./model/model_trad_acc_f4.h5"

	model = load_model(model_path,{'ss':ss,'sensitivity':sensitivity,'specificity':specificity})
	model.summary()
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	# ------------ vis filter -------------
	# i = 1
	# for layer in model.layers: 
	# 	if i != 1:
	# 		vis_img_in_filter((300,300),layer_dict,model,layer.name,4,img=img,usingimg=True)
	# 	i+=1
	# ------------ forward path -------------
	for i in range(len(X)):
		img = X[i,:,:,:].reshape(1,300,300,4)
		plotoutput(img,model,i+1)

if __name__ == "__main__":
	main()