import numpy as np
import pandas as pd
import keras
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

def ROI_mapping(X):
	t = 1
	for i in range(X.shape[0]):
		if os.path.exists('./dcm_data_binaryMap/binMap_'+str(t)+'.png'):
			binmap = scipy.misc.imread('./dcm_data_binaryMap/binMap_'+str(t)+'.png', flatten=False,mode='L')
		else :
			t+=1
			binmap = scipy.misc.imread('./dcm_data_binaryMap/binMap_'+str(t)+'.png', flatten=False,mode='L')
		for j in range(X.shape[3]):
			X[i,:,:,j]*=binmap
		t+=1
		
	return X

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

	model = Sequential()
	model.add(Conv2D(12, kernel_size=(3,3),activation='relu',input_shape=(300,300,4)))
	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l1(0.005)))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',precision,recall])

	model.fit(X, y,
			  batch_size=20,
			  epochs=25,
			  verbose=1,
			  validation_split=0.0,validation_data=(val_X,val_y),class_weight={0:65/263,1:(200/263)},callbacks=[checkpoint_loss,checkpoint_acc])

def cnn_ROI(X,y):
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

	model = Sequential()
	model.add(Conv2D(12, kernel_size=(1,1),activation='relu',input_shape=(300,300,4)))
	model.add(Conv2D(12, (1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(12, (2, 2), activation='relu'))
	model.add(Conv2D(12, (2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(Conv2D(12, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l1(0.005)))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',precision,recall])

	model.fit(X, y,
			  batch_size=20,
			  epochs=20,
			  verbose=1,
			  validation_split=0.0,validation_data=(val_X,val_y),class_weight={0:65/263,1:(200/263)},callbacks=[checkpoint_loss,checkpoint_acc])

def plot_confusion_matrix(cm, classes,
						  normalize=True,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
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


def cnn_inception(X,ROI_X,y):
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
	ROI_X[:,:,:,0] /=255
	ROI_X = ROI_X[:,:,:,0].reshape(ROI_X.shape[0],ROI_X.shape[1],ROI_X.shape[2])
	ROI_X_T = np.array([0.0]*ROI_X.shape[0]*ROI_X.shape[1]*ROI_X.shape[2]) .reshape(ROI_X.shape[0],ROI_X.shape[1],ROI_X.shape[2])
	for i in range(ROI_X.shape[0]):ROI_X_T[i] = np.transpose(ROI_X[i])
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

	ROI_test_X = ROI_X[test_index]
	ROI_val_X = ROI_X[val_index]
	ROI_X = ROI_X[train_index]

	ROIT_test_X = ROI_X_T[test_index]
	ROIT_val_X = ROI_X_T[val_index]
	ROI_X_T = ROI_X_T[train_index]
	

	input_img = Input(shape=(X.shape[1],X.shape[2],X.shape[3]))
	input_roi = Input(shape=(ROI_X.shape[1],ROI_X.shape[2]))
	input_roit = Input(shape=(ROI_X_T.shape[1],ROI_X_T.shape[2]))

	con2d1 = Conv2D(12, kernel_size=(3,3),activation='relu')(input_img)
	con2d1 = Conv2D(12, kernel_size=(3,3),activation='relu')(con2d1)
	con2d1 = Dropout(0.5)(con2d1)
	maxp1 = MaxPooling2D(pool_size=(2, 2))(con2d1)

	con2d2 = Conv2D(12, kernel_size=(3,3),activation='relu')(maxp1)
	con2d2 = Conv2D(12, kernel_size=(3,3),activation='relu')(con2d2)
	con2d2 = Dropout(0.5)(con2d2)
	maxp2 = MaxPooling2D(pool_size=(2, 2))(con2d2)

	con2d3 = Conv2D(12, kernel_size=(3,3),activation='relu')(maxp2)
	con2d3 = Conv2D(12, kernel_size=(3,3),activation='relu')(con2d3)
	con2d3 = Dropout(0.5)(con2d3)
	maxp3 = MaxPooling2D(pool_size=(2, 2))(con2d3)

	con2d4 = Conv2D(12, kernel_size=(3,3),activation='relu')(maxp3)
	con2d4 = Conv2D(12, kernel_size=(3,3),activation='relu')(con2d4)
	con2d4 = Dropout(0.5)(con2d4)
	maxp4 = MaxPooling2D(pool_size=(2, 2))(con2d4)
	f1 = Flatten()(maxp4)

	conv1d1 = Conv1D(10,kernel_size=5,strides=1, padding='valid')(input_roi)
	conv1d1 = Dropout(0.5)(conv1d1)
	conv1d2 = Conv1D(10,kernel_size=10,strides=1, padding='valid')(input_roi)
	conv1d2 = Dropout(0.5)(conv1d2)
	conv1d3 = Conv1D(10,kernel_size=15,strides=1, padding='valid')(input_roi)
	conv1d3 = Dropout(0.5)(conv1d3)
	conv1d4 = Conv1D(10,kernel_size=20,strides=1, padding='valid')(input_roi)
	conv1d4 = Dropout(0.5)(conv1d4)
	conv1d1 = Conv1D(10,kernel_size=5,strides=1, padding='valid')(conv1d1)
	conv1d1 = Dropout(0.5)(conv1d1)
	conv1d2 = Conv1D(10,kernel_size=10,strides=1, padding='valid')(conv1d2)
	conv1d2 = Dropout(0.5)(conv1d2)
	conv1d3 = Conv1D(10,kernel_size=15,strides=1, padding='valid')(conv1d3)
	conv1d3 = Dropout(0.5)(conv1d3)
	conv1d4 = Conv1D(10,kernel_size=20,strides=1, padding='valid')(conv1d4)
	conv1d4 = Dropout(0.5)(conv1d4)
	concate1 = concatenate([conv1d1,conv1d2,conv1d3,conv1d4],axis=1)
	f2 = Flatten()(concate1)

	conv1d5 = Conv1D(10,kernel_size=5,strides=1, padding='valid')(input_roit)
	conv1d5 = Dropout(0.5)(conv1d5)
	conv1d6 = Conv1D(10,kernel_size=10,strides=1, padding='valid')(input_roit)
	conv1d6 = Dropout(0.5)(conv1d6)
	conv1d7 = Conv1D(10,kernel_size=15,strides=1, padding='valid')(input_roit)
	conv1d7 = Dropout(0.5)(conv1d7)
	conv1d8 = Conv1D(10,kernel_size=20,strides=1, padding='valid')(input_roit)
	conv1d8 = Dropout(0.5)(conv1d8)
	conv1d5 = Conv1D(10,kernel_size=5,strides=1, padding='valid')(conv1d5)
	conv1d5 = Dropout(0.5)(conv1d5)
	conv1d6 = Conv1D(10,kernel_size=10,strides=1, padding='valid')(conv1d6)
	conv1d6 = Dropout(0.5)(conv1d6)
	conv1d7 = Conv1D(10,kernel_size=15,strides=1, padding='valid')(conv1d7)
	conv1d7 = Dropout(0.5)(conv1d7)
	conv1d8 = Conv1D(10,kernel_size=20,strides=1, padding='valid')(conv1d8)
	conv1d8 = Dropout(0.5)(conv1d8)
	concate2 = concatenate([conv1d5,conv1d6,conv1d7,conv1d8],axis=1)
	f3 = Flatten()(concate2)

	concate3 = concatenate([f1,f2,f3],axis=1)
	d1 = Dropout(0.5)(concate3)
	output = Dense(64, activation='relu')(concate3)
	output = Dropout(0.5)(output)
	output = Dense(32, activation='relu',kernel_regularizer=regularizers.l1(0.01))(output)
	output = Dense(1, activation='sigmoid')(output)

	model = Model([input_img,input_roi,input_roit],output)
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',precision,recall])
	model.fit({'input_1':X,'input_2':ROI_X,'input_3':ROI_X_T}, y,
			  batch_size=20,
			  epochs=150,
			  verbose=1,
			  validation_split=0.0,validation_data=({'input_1':val_X,'input_2':ROI_val_X,'input_3':ROIT_val_X},val_y),class_weight={0:65/263,1:(200/263)},callbacks=[checkpoint_loss,checkpoint_acc])
	# print(np.sort(test_index) ,'\n',np.sort(val_index) ,'\n',np.sort(train_index) )

def get_performance_inception(X,ROI_X,y):
	model = LoadModel('./model/model_loss.h5',{'precision':precision,'recall':recall})

	test_index = np.load('./tmp_index/test_index.npy')
	val_index = np.load('./tmp_index/val_index.npy')
	train_index = np.load('./tmp_index/train_index.npy')
	# print(np.sort(test_index) ,'\n',np.sort(val_index) ,'\n',np.sort(train_index) )
	X[:,:,:,0] /=255
	X[:,:,:,1]/=np.max(X[:,:,:,1])
	X[:,:,:,2]/=np.max(X[:,:,:,2])
	X[:,:,:,3]/=np.max(X[:,:,:,3])
	ROI_X[:,:,:,0] /=255
	ROI_X = ROI_X[:,:,:,0].reshape(ROI_X.shape[0],ROI_X.shape[1],ROI_X.shape[2])
	ROI_X_T = np.array([0.0]*ROI_X.shape[0]*ROI_X.shape[1]*ROI_X.shape[2]) .reshape(ROI_X.shape[0],ROI_X.shape[1],ROI_X.shape[2])
	for i in range(ROI_X.shape[0]):ROI_X_T[i] = np.transpose(ROI_X[i])

	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]
	ROI_test_X = ROI_X[test_index]
	ROI_val_X = ROI_X[val_index]
	ROI_X = ROI_X[train_index]
	ROIT_test_X = ROI_X_T[test_index]
	ROIT_val_X = ROI_X_T[val_index]
	ROI_X_T = ROI_X_T[train_index]
	
	print(model.evaluate({'input_1':test_X,'input_2':ROI_test_X,'input_3':ROIT_test_X},test_y))
	print(model.evaluate({'input_1':val_X,'input_2':ROI_val_X,'input_3':ROIT_val_X},val_y))
	print(model.evaluate({'input_1':X,'input_2':ROI_X,'input_3':ROI_X_T},y))
	cnf_matrix = confusion_matrix(test_y, [ 1 if i>0.5 else 0  for i in model.predict({'input_1':test_X,'input_2':ROI_test_X,'input_3':ROIT_test_X})], labels=[0,1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['benign','malignant'],
                      title='Testing Confusion matrix')
	cnf_matrix = confusion_matrix(val_y, [ 1 if i>0.5 else 0  for i in model.predict({'input_1':val_X,'input_2':ROI_val_X,'input_3':ROIT_val_X})], labels=[0,1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['benign','malignant'],
                      title='Validation Confusion matrix')
	cnf_matrix = confusion_matrix(y, [ 1 if i>0.5 else 0  for i in model.predict({'input_1':X,'input_2':ROI_X,'input_3':ROI_X_T})], labels=[0,1])
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=['benign','malignant'],
                      title='Training Confusion matrix')
	plt.show()
def get_performance(X,y):
	model = LoadModel('./model/model_loss.h5',{'precision':precision,'recall':recall})

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
	# plt.imshow(X[5,:,:,1:4],cmap='gray')
	# plt.show()
if __name__ == '__main__':
	X,y = load()
	ROI_X = ROI_mapping(load()[0])
	cnn_inception(X,ROI_X,y)

	X,y = load()
	ROI_X = ROI_mapping(load()[0])
	get_performance_inception(X,ROI_X,y)
	# cnn(X,y)
	# get_performance(ROI_X,y)
	# get_performance(X,y)
	# cnn_ROI(ROI_X,y)
	# get_performance(ROI_X,y)
	# get_performance(X,y)
	
	