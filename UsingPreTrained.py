import numpy as np
from keras.applications.vgg16 import VGG16
from keras import regularizers,applications,Model
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten,concatenate,Input
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.pooling import MaxPooling2D,MaxPooling1D,AveragePooling2D,AveragePooling1D
from keras.models import model_from_json
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras.optimizers import Adam
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
	TP = K.sum(pred*y_true)
	if true_number==0:return 0 
	return TP/true_number

def specificity(y_true, y_pred):
	pred = K.round(y_pred)
	pred = K.abs(pred-1)
	transpose_false = K.abs(y_true-1)
	false_number = K.sum(transpose_false)
	TN = K.sum(pred*transpose_false)
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
	
	# test_index,val_index,train_index = shuffle(X,y)
	test_index = np.load('./tmp_index/test_index.npy') 
	val_index = np.load('./tmp_index/val_index.npy')
	train_index = np.load('./tmp_index/train_index.npy')

	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]
	return X,y,test_X,test_y,val_X,val_y
def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
def get_performance(X,y):
	model = load_model('./model/model_loss.h5',{'sensitivity':sensitivity,'specificity':specificity})
	model.summary()
	test_index = np.load('./tmp_index/test_index.npy')
	val_index = np.load('./tmp_index/val_index.npy')
	train_index = np.load('./tmp_index/train_index.npy')
	# print(np.sort(test_index) ,'\n',np.sort(val_index) ,'\n',np.sort(train_index) )
	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]

	# print(model.predict(test_X))
	print(model.evaluate(test_X,test_y))
	print(model.evaluate(val_X,val_y))
	print(model.evaluate(X,y))
	cnf_matrix = confusion_matrix(test_y, [ 1 if i>0.5 else 0  for i in model.predict(test_X)], labels=[0,1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['benign','malignant'],
                      title='Testing Confusion matrix')
	cnf_matrix = confusion_matrix(val_y, [ 1 if i>0.5 else 0  for i in model.predict(val_X)], labels=[0,1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['benign','malignant'],
                      title='Validation Confusion matrix')
	cnf_matrix = confusion_matrix(y, [ 1 if i>0.5 else 0  for i in model.predict(X)], labels=[0,1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['benign','malignant'],
                      title='Training Confusion matrix')
	plt.show()

if __name__ == '__main__':
	X,y = load()
	X,y,test_X,test_y,val_X,val_y = prepare_for_train(X,y)
	# get_performance(X[:,:,:,0:3],y)
	base_model = VGG16(weights='imagenet', include_top=False)
	# base_model.summary()
	layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

	inputlayer = Input(shape=(X.shape[1],X.shape[2],X.shape[3]-1))
	conv1 = layer_dict['block1_conv1'](inputlayer)
	conv1.trainable = False
	conv2 = layer_dict['block1_conv2'](conv1)
	conv2.trainable = False
	max1 = MaxPooling2D(pool_size=(2, 2))(conv2)

	# conv1 = Conv2D(24, kernel_size=(3,3),activation='relu')(inputlayer)
	# conv2 = Conv2D(24, kernel_size=(3,3),activation='relu')(conv1)
	# max1 =  MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(24, kernel_size=(3,3),activation='relu')(conv2)
	conv4 = Conv2D(24, kernel_size=(3,3),activation='relu')(conv3)
	max2 =  MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(48, kernel_size=(3,3),activation='relu')(max2)
	conv6 = Conv2D(48, kernel_size=(3,3),activation='relu')(conv5)
	max3 =  MaxPooling2D(pool_size=(2, 2))(conv6)

	output = Flatten()(max3)
	# output = Dense(64, activation='relu',kernel_regularizer=regularizers.l1(0.00001))(output)
	output = Dense(128, activation='relu',kernel_regularizer=regularizers.l1(0.00001))(output)
	output = Dense(1, activation='sigmoid')(output)
	model = Model(input = inputlayer, output = output)

	model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.000001),metrics=['accuracy',sensitivity,specificity])
	model.fit(X[:,:,:,0:3], y,
			  batch_size=20,
			  epochs=150,
			  verbose=1,
			  validation_split=0.0,validation_data=(val_X[:,:,:,0:3],val_y),class_weight={0:1,1:1.1},callbacks=[checkpoint_loss,checkpoint_acc])
	
