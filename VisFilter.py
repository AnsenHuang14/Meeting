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


def vis_img_in_filter(img_dim,layer_dict,model,
					  layer_name ,channel,img=None,):
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

		if img==None:
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
	print(loss_list)
	idx   = np.flip(np.argsort(loss_list),0)
	print(type(img_ascs),idx)
	img_ascs = [ img_ascs[i] for i in idx]
	if layer_output.shape[3] >= 35:
		plot_x, plot_y = 6, 6
	elif layer_output.shape[3] >= 23:
		plot_x, plot_y = 4, 6
	elif layer_output.shape[3] >= 11:
		plot_x, plot_y = 2, 6
	else:
		plot_x, plot_y = 1, 2
	fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
	if img!=None:
		ax[0, 0].imshow(img.reshape((img_dim[0], img_dim[1],channel)))
		ax[0, 0].set_title('Input image')
	fig.suptitle('Input image and %s filters' % (layer_name,))
	fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
	for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
		if img!=None and x == 0 and y == 0:
		    continue
		ax[x, y].imshow(img_ascs[x * plot_y + y - 1])
		ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))

	fig.savefig(layer_name+'.png', dpi=100)

def main():
	
	K.set_learning_phase(1)

	model_name = "model"
	model_path = "./model/model_loss.h5"
	# model = VGG16(weights='imagenet', include_top=False)
	model = load_model(model_path,{'sensitivity':sensitivity,'specificity':specificity})
	model.summary()
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	# vis_img_in_filter((300,300),layer_dict,model,'block1_conv1',3,img=None)
	# # img =  scipy.misc.imread('./image_remove_dot/color_4.png', flatten=False,mode='RGB').reshape(1,300,300,3)
	i = 1
	for layer in model.layers: 
		# if i != 1:
		vis_img_in_filter((300,300),layer_dict,model,layer.name,4,img=None)
		i+=1

if __name__ == "__main__":
	main()