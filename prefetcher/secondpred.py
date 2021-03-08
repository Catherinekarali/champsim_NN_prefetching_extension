#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#import sys
#sys.path.append('~/Documents/ChampSim/prefetcher')
#import tensorflow as tf
#tf.device("cpu:0")
from tensorflow.keras.backend import clear_session
from numpy import array
from keras.models import model_from_json
#from keras.models import load_model
from keras import backend as K

def pref_init():
	Model = '649_33_cells32'
	global model
	K.clear_session()
#	model=load_model(Model+"new")
#	tf.keras.backend.clear_session()
#	model = tf.keras.models.load_model(Model+"new", compile=False)
	clear_session()
	json_file = open(Model + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights(Model + ".h5")
	model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
	return 0;

def pref_operate(h_list):
	steps = 33
	for i in range(1, len(h_list)):
		if (h_list[i]<-64 or h_list[i]>64):
			h_list[i] = 0
	h_list[0] = (h_list[0]&127)
	x = array(h_list)
	XX = x.reshape(1,steps,1)

	a = model.predict(XX)
	pred1 = list(a[0]).index(max(list(a[0])))-64
	a[0][list(a[0]).index(max(list(a[0])))] = 0
	pred2 = list(a[0]).index(max(list(a[0])))-64
	return (pred1, pred2) 
