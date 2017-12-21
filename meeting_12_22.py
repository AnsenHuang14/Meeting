import numpy as np
import pandas as pd
import keras,os,skimage.measure,scipy.misc,itertools,random
from keras import regularizers,applications,Model
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Flatten,concatenate,Input
from keras.layers.convolutional import Conv2D,Conv1D,SeparableConv2D
from keras.layers.pooling import MaxPooling2D,MaxPooling1D,AveragePooling2D,AveragePooling1D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from vis.utils import utils
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=True,fill_mode='wrap')
def Datagen(X,y,test_index,val_index,train_index):
	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]
	i=0
	for batch in datagen.flow(X[:,:,:,0].reshape(182,300,300,1), batch_size=182,
	                      save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
		i += 1
		if i > 2:
			break  # otherwise the generator would loop indefinitely
	# print(datagen.flow(X, y,182)[0][0].shape)

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

def load(Xpath='./Data/dcm_data_366_4channel.npy',ypath='./Data/y.txt'):
	X = np.load(Xpath)
	y = pd.read_csv(ypath,sep='\t',header=None).as_matrix()
	y = y[:,1]
	print('malignant: ',np.sum(y),'benign:',263-np.sum(y))
	print('X,y shape: ',X.shape,y.shape)
	return X,y

def fold(X,y,fold=1,save = True,shuf = True):
	if shuf:
		malignant_index = [i for i,x in enumerate(y.reshape(1,263)[0].tolist()) if x == 1]
		benign_index = [i for i,x in enumerate(y.reshape(1,263)[0].tolist()) if x == 0]	
		random.shuffle(malignant_index)
		random.shuffle(benign_index)
		test_index1 = malignant_index[0:7]+benign_index[0:20]
		val_index1 = malignant_index[7:21]+benign_index[20:60]
		train_index1 = malignant_index[21:]+benign_index[60:]
		
		test_index2 = malignant_index[7:14]+benign_index[20:40]
		val_index2 = malignant_index[14:28]+benign_index[40:80]
		train_index2 = malignant_index[0:7]+malignant_index[28:]+benign_index[0:20]+benign_index[80:]
		
		test_index3 = malignant_index[14:21]+benign_index[40:60]
		val_index3 = malignant_index[21:35]+benign_index[60:100]
		train_index3 = malignant_index[0:14]+malignant_index[35:]+benign_index[0:40]+benign_index[100:]
		
		test_index4 = malignant_index[21:28]+benign_index[60:80]
		val_index4 = malignant_index[28:42]+benign_index[80:120]
		train_index4 = malignant_index[0:21]+malignant_index[42:]+benign_index[0:60]+benign_index[120:]
		if save:
			np.save('./tmp_index/test_index1',test_index1)
			np.save('./tmp_index/train_index1',train_index1)
			np.save('./tmp_index/val_index1',val_index1)

			np.save('./tmp_index/test_index2',test_index2)
			np.save('./tmp_index/train_index2',train_index2)
			np.save('./tmp_index/val_index2',val_index2)

			np.save('./tmp_index/test_index3',test_index3)
			np.save('./tmp_index/train_index3',train_index3)
			np.save('./tmp_index/val_index3',val_index3)

			np.save('./tmp_index/test_index4',test_index4)
			np.save('./tmp_index/train_index4',train_index4)
			np.save('./tmp_index/val_index4',val_index4)
	
	test_index = np.load('./tmp_index/test_index'+str(fold)+'.npy')
	val_index = np.load('./tmp_index/val_index'+str(fold)+'.npy')
	train_index = np.load('./tmp_index/train_index'+str(fold)+'.npy')
		
	return test_index,val_index,train_index
	

def shuffle(X,y,save = True,shuf = True):
	if  shuf==False:
		test_index = np.load('./tmp_index/test_index.npy')
		val_index = np.load('./tmp_index/val_index.npy')
		train_index = np.load('./tmp_index/train_index.npy')
	else:
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
	if save:
		print('------saved temp index------')
		np.save('./tmp_index/test_index',test_index)
		np.save('./tmp_index/train_index',train_index)
		np.save('./tmp_index/val_index',val_index)
	return test_index,val_index,train_index

# so far best testing structure
def cnn(X,y,fold,test_index,val_index,train_index):
	checkpoint_loss = ModelCheckpoint('./model/model_trad_loss_f'+str(fold)+'.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_acc = ModelCheckpoint('./model/model_trad_acc_f'+str(fold)+'.h5', monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')
	checkpoint_ss = ModelCheckpoint('./model/model_trad_ss_f'+str(fold)+'.h5', monitor = 'val_ss',verbose = 1,save_best_only = True,mode = 'max')
 
	X = X.astype('float32')
	# plt.imshow(X[0,:,:,1:4],cmap='gray')
	# plt.show()
	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]


	input_img = Input(shape=(X.shape[1],X.shape[2],X.shape[3]))

	con2d1 = Conv2D(4, kernel_size=(3,3), padding='valid',dilation_rate=(2, 2),kernel_initializer='glorot_uniform',activation='relu')(input_img)
	con2d1 = Conv2D(4, kernel_size=(3,3), padding='valid',dilation_rate=(2, 2),kernel_initializer='glorot_uniform',activation='relu')(con2d1)
	maxp1 = MaxPooling2D(pool_size=(2, 2))(con2d1)

	con2d2 = Conv2D(8, kernel_size=(3,3), padding='valid',dilation_rate=(2, 2),kernel_initializer='glorot_uniform',activation='relu')(maxp1)
	con2d2 = Conv2D(8, kernel_size=(3,3), padding='valid',dilation_rate=(2, 2),kernel_initializer='glorot_uniform',activation='relu')(con2d2)
	maxp2 = MaxPooling2D(pool_size=(2, 2))(con2d2)

	con2d3 = Conv2D(12, kernel_size=(3,3), padding='valid',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(maxp2)
	con2d3 = Conv2D(12, kernel_size=(3,3), padding='valid',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu')(con2d3)
	maxp3 = MaxPooling2D(pool_size=(2, 2))(con2d3)

	con2d4 = Conv2D(16, kernel_size=(3,3), padding='valid',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.00))(maxp3)
	con2d4 = Conv2D(16, kernel_size=(3,3), padding='valid',dilation_rate=(1, 1),kernel_initializer='glorot_uniform',activation='relu',kernel_regularizer=regularizers.l2(0.00))(con2d4)
	maxp4 = MaxPooling2D(pool_size=(2, 2))(con2d4)


	output = Dense(128,activation='relu')(Flatten()(maxp4))
	output = Dropout(0.5)(output)
	output = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(25))(output)
	model = Model(input_img,output)
	model.summary()
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',ss,sensitivity,specificity])
	
	# model.fit_generator(datagen.flow(X, y,32),
	# 		steps_per_epoch=len(X) / 32,
	# 		epochs=50,
	# 		verbose=1,
	# 		validation_data=(val_X,val_y),
	# 		class_weight={0:(65/263),1:(275/263)},callbacks=[checkpoint_loss,checkpoint_acc])

	model.fit(X, y,
			  batch_size=61,
			  epochs=1000,
			  verbose=1,
			  validation_split=0.0,validation_data=(val_X,val_y),class_weight={0:1,1:10},callbacks=[checkpoint_ss,checkpoint_loss,checkpoint_acc])

def sep_cnn(X,y,fold,test_index,val_index,train_index):
	checkpoint_loss = ModelCheckpoint('./model/model_sep_loss_f'+str(fold)+'.h5', monitor = 'val_loss',verbose = 1,save_best_only = True,mode = 'min')
	checkpoint_acc = ModelCheckpoint('./model/model_sep_acc_f'+str(fold)+'.h5', monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')
	checkpoint_ss = ModelCheckpoint('./model/model_sep_ss_f'+str(fold)+'.h5', monitor = 'val_ss',verbose = 1,save_best_only = True,mode = 'max')
 
	X = X.astype('float32')
	# plt.imshow(X[0,:,:,1:4],cmap='gray')
	# plt.show()
	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]


	input_img = Input(shape=(X.shape[1],X.shape[2],X.shape[3]))

	con2d1 = SeparableConv2D(4,depth_multiplier =2, kernel_size=(3,3),activation='relu')(input_img)
	con2d1 = SeparableConv2D(4,depth_multiplier =2, kernel_size=(3,3),activation='relu')(con2d1)
	maxp1 = MaxPooling2D(pool_size=(2, 2))(con2d1)

	con2d2 = SeparableConv2D(8,depth_multiplier =4, kernel_size=(5,5),activation='relu')(maxp1)
	con2d2 = SeparableConv2D(8,depth_multiplier =4, kernel_size=(5,5),activation='relu')(con2d2)
	maxp2 = MaxPooling2D(pool_size=(2, 2))(con2d2)

	con2d3 = SeparableConv2D(12,depth_multiplier =4, kernel_size=(5,5),activation='relu')(maxp2)
	con2d3 = SeparableConv2D(12,depth_multiplier =4, kernel_size=(5,5),activation='relu')(con2d3)
	maxp3 = MaxPooling2D(pool_size=(2, 2))(con2d3)


	output = Dense(256,activation='relu')(Flatten()(maxp3))
	output = Dropout(0.5)(output)
	output = Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(25))(output)
	model = Model(input_img,output)
	model.summary()
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',ss,sensitivity,specificity])
	
	# model.fit_generator(datagen.flow(X, y,32),
	# 		steps_per_epoch=len(X) / 32,
	# 		epochs=50,
	# 		verbose=1,
	# 		validation_data=(val_X,val_y),
	# 		class_weight={0:(65/263),1:(275/263)},callbacks=[checkpoint_loss,checkpoint_acc])

	model.fit(X, y,
			  batch_size=61,
			  epochs=200,
			  verbose=1,
			  validation_split=0.0,validation_data=(val_X,val_y),class_weight={0:1,1:10},callbacks=[checkpoint_ss,checkpoint_loss,checkpoint_acc])


if __name__ == '__main__':
	X,y = load(Xpath='./Data/dcm_data.npy',ypath='./Data/y.txt')
	# f = 0
	# test_index,val_index,train_index = fold(X,y,f,save = False,shuf = False)
	f='4'
	test_index,val_index,train_index = shuffle(X,y,False,False)
	cnn(X,y,f,test_index,val_index,train_index)
	# sep_cnn(X,y,f,test_index,val_index,train_index)
	# Datagen(X,y,test_index,val_index,train_index)