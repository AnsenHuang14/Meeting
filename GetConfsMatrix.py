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
from vis.utils import utils
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
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

def get_performance(X,y,fold,metric='acc'):
	model = load_model('./model/model_trad_'+metric+'_f'+str(fold)+'.h5',{'ss':ss,'sensitivity':sensitivity,'specificity':specificity})

	test_index = np.load('./tmp_index/test_index'+str(fold)+'.npy')
	val_index = np.load('./tmp_index/val_index'+str(fold)+'.npy')
	train_index = np.load('./tmp_index/train_index'+str(fold)+'.npy')
	# print(np.sort(test_index) ,'\n',np.sort(val_index) ,'\n',np.sort(train_index) )
	test_X = X[test_index]
	test_y = y[test_index]
	val_X = X[val_index]
	val_y = y[val_index]
	X = X[train_index]
	y = y[train_index]
	print(test_y)
	# print(model.predict(test_X))
	print('Testing',model.evaluate(test_X,test_y))
	print('Validation',model.evaluate(val_X,val_y))
	print('Training',model.evaluate(X,y))
	# print(test_y,'\n',[ 1 if i>0.5 else 0  for i in model.predict(test_X)])
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
	X,y = load(Xpath='./Data/dcm_data.npy',ypath='./Data/y.txt')
	f='3'
	get_performance(X,y,f,'loss')
	get_performance(X,y,f,'acc')
	get_performance(X,y,f,'ss')
	# get_performance(X,y,'./model/model_sep_loss.h5')