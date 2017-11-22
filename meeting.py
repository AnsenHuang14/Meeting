import numpy as np
import pandas as pd
import keras
from keras import regularizers,applications
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
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
def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall
def LoadModel(path,c=None):
	model = load_model(path,custom_objects=c)
	model.summary()
	
	return model
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
	print('test_index =',test_index)
	print('val_index =',val_index)
	print('train_index =',train_index)
	return test_index,val_index,train_index

def cnn(X,y):
	earlyStopping_acc = EarlyStopping( monitor = 'val_acc',patience = 10)# monitor-> val_loss,val_acc...
	earlyStopping_loss = EarlyStopping( monitor = 'val_loss',patience = 10)
	checkpoint_loss = ModelCheckpoint('./model/model_loss.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_acc = ModelCheckpoint('./model/model_acc.h5', monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')
	# X = X[:,:,:,0].reshape(len(X),300,300,1)
	X = X.astype('float32')
	# X/=255
	X[:,:,:,0] /=255
	X[:,:,:,1]/=np.max(X[:,:,:,1])
	X[:,:,:,2]/=np.max(X[:,:,:,2])
	X[:,:,:,3]/=np.max(X[:,:,:,3])
	# plt.imshow(X[0,:,:,1:4],cmap='gray')
	# plt.show()
	
	# test_index,val_index,train_index = shuffle(X,y)
	test_index = np.load('./index/test_index.npy') 
	val_index = np.load('./index/val_index.npy')
	train_index = np.load('./index/train_index.npy')

	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]
	np.save('./index/test_index',test_index)
	np.save('./index/train_index',train_index)
	np.save('./index/val_index',val_index)
	
	model = Sequential()
	model.add(Conv2D(48, kernel_size=(3,3),activation='relu',input_shape=(300,300,4)))
	model.add(Conv2D(48, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(48, (3, 3), activation='relu'))
	model.add(Conv2D(96, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	# model.add(Conv2D(64, (3, 3), activation='relu'))
	# model.add(Conv2D(64, (3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l1(0.005)))
	model.add(Dropout(0.25))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',precision,recall])

	model.fit(X, y,
	          batch_size=20,
	          epochs=150,
	          verbose=1,
	          validation_split=0.0,validation_data=(val_X,val_y),class_weight={0:65/263,1:(200/263)},callbacks=[checkpoint_loss,checkpoint_acc])


if __name__ == '__main__':
	X,y = load()
	cnn(X,y)
	model = LoadModel('./model/model_loss.h5',{'precision':precision,'recall':recall})

	test_index = np.load('./index/test_index.npy')
	val_index = np.load('./index/val_index.npy')
	train_index = np.load('./index/train_index.npy')
	# print(np.sort(test_index) ,'\n',np.sort(val_index) ,'\n',np.sort(train_index) )
	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]

	print(model.predict(test_X))
	print(model.evaluate(test_X,test_y))
	print(model.evaluate(val_X,val_y))
	print(model.evaluate(X,y))
	
	