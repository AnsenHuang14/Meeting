import numpy as np
from keras.applications.vgg16 import VGG16
from keras import regularizers,applications,Model
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten,concatenate,Input
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.pooling import MaxPooling2D,MaxPooling1D,AveragePooling2D,AveragePooling1D
from keras.models import model_from_json
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
from vis.utils import utils
from keras.models import load_model
import os
import skimage.measure
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import itertools
checkpoint_loss = ModelCheckpoint('./model/model_loss.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
checkpoint_acc = ModelCheckpoint('./model/model_acc.h5', monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')

def sensitivity(y_true, y_pred):
	true_number = K.sum(y_true)
	pred = K.round(y_pred)
	TP = K.sum(pred[y_true==1]==1)
	return TP/true_number

def specificity(y_true, y_pred):
	false_number = K.sum(y_true==0)
	pred = K.round(y_pred)
	TN = K.sum(pred[y_true==0]==0)
	return TN/false_number

def load():
	X = np.load('./Data/dcm_data.npy')
	y = pd.read_csv('./Data/y.txt',sep='\t',header=None).as_matrix()
	y = y[:,1]
	print('malignant:',np.sum(y),'benign:',263-np.sum(y))
	print(X.shape,y.shape)
	return X,y

def shuffle(X,y):
	malignant_index = [i for i,x in enumerate(y.reshape(1,263)[0].tolist()) if x == 1]
	benign_index = [i for i,x in enumerate(y.reshape(1,263)[0].tolist()) if x == 0]	
	random.shuffle(malignant_index)
	random.shuffle(benign_index)
	test_index = malignant_index[0:7]+benign_index[0:20]
	val_index = malignant_index[7:21]+benign_index[20:60]
	train_index = malignant_index[21:]+benign_index[60:]
	
	random.shuffle(test_index)
	random.shuffle(train_index)
	random.shuffle(val_index)
	print('tmp_test_index =',test_index)
	print('tmp_val_index =',val_index)
	print('tmp_train_index =',train_index)
	np.save('./tmp_index/test_index',test_index)
	np.save('./tmp_index/train_index',train_index)
	np.save('./tmp_index/val_index',val_index)
	return test_index,val_index,train_index

def prepare_for_train(X,y):
	X = X.astype('float32')
	# X/=255
	X[:,:,:,0] /=255
	X[:,:,:,1]/=np.max(X[:,:,:,1])
	X[:,:,:,2]/=np.max(X[:,:,:,2])
	X[:,:,:,3]/=np.max(X[:,:,:,3])
	# plt.imshow(X[0,:,:,1:4],cmap='gray')
	# plt.show()
	
	test_index,val_index,train_index = shuffle(X,y)
	# test_index = np.load('./tmp_index/test_index.npy') 
	# val_index = np.load('./tmp_index/val_index.npy')
	# train_index = np.load('./tmp_index/train_index.npy')

	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]
	return X,y,test_X,test_y,val_X,val_y


if __name__ == '__main__':
	X,y = load()
	X,y,test_X,test_y,val_X,val_y = prepare_for_train(X,y)
	
	base_model = VGG16(weights='imagenet', include_top=False)
	layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

	inputlayer = Input(shape=(X.shape[1],X.shape[2],X.shape[3]))
	conv1 = layer_dict['block1_conv1'](inputlayer)
	conv1.trainable = False
	conv2 = layer_dict['block1_conv2'](conv1)
	conv2.trainable = False
	max1 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(12, kernel_size=(3,3),activation='relu')(max1)
	conv4 = Conv2D(12, kernel_size=(3,3),activation='relu')(conv3)
	max2 =  MaxPooling2D(pool_size=(2, 2))(conv4)

	output = Flatten()(max2)
	output = Dense(32, activation='relu',kernel_regularizer=regularizers.l1(0.01))(output)
	output = Dense(1, activation='sigmoid')(output)
	Model = Model(input = inputlayer, output = output)

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',sensitivity,specificity])
	model.fit(X, y,
			  batch_size=20,
			  epochs=150,
			  verbose=1,
			  validation_split=0.0,validation_data=(val_X,val_y),class_weight={0:65/263,1:(200/263)},callbacks=[checkpoint_loss,checkpoint_acc])
	
