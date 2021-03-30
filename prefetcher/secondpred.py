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
from tensorflow.keras.models import load_model
from keras import backend as K
import os

def pref_init(Model):
	global model
	global transformer
	K.clear_session()
#	model=load_model(Model+"new")
#	tf.keras.backend.clear_session()
#	model = tf.keras.models.load_model(Model+"new", compile=False)
	clear_session()
	if os.path.exists(Model + '.json'):
		json_file = open(Model + '.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights(Model + ".h5")
		model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
		transformer = False
	elif os.path.exists(Model):
		model=load_model(Model, compile = False)
		transformer = True
	return 0;

def pref_operate(h_list):
	steps = 33
	for i in range(1, len(h_list)):
		if (h_list[i]<-64 or h_list[i]>64):
			h_list[i] = 0
	h_list[0] = (h_list[0]&127)
	x = array(h_list)
	if (transformer == False):	
		XX = x.reshape(1,steps,1)
	else:
		x = x + 64
		XX = x.reshape(1,33)

	a = model.predict(XX)
	pred1 = list(a[0]).index(max(list(a[0])))-64
	a[0][list(a[0]).index(max(list(a[0])))] = 0
	pred2 = list(a[0]).index(max(list(a[0])))-64
	return (pred1, pred2) 
