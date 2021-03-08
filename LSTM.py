from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from numpy import array
from numpy import reshape
from numpy import argmax
import pandas as pd
import csv
import matplotlib.pyplot as plt
from keras import backend as K

classes = 129
cells,features,steps = 128,1,33
Model = "628_33_cells{}".format(cells)

def create_model(cells, features, steps):
    model = Sequential()
    model.add(LSTM(cells,input_shape=(steps, features), dropout=0.2, recurrent_dropout=0.2, return_sequences=False, unroll=True))
#    model.add(Attention(return_sequences=False))
#   model.add(LSTM(cells,input_shape=(steps, features), dropout=0.2, recurrent_dropout=0.2, return_sequences=False, unroll=True))
    model.add(BatchNormalization())
    model.add(Dense(classes, activation='softmax'))
    #model.add(Activation('softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = create_model(cells,features,steps)

data = pd.read_csv("628_50_200_33_s2000.txt", header = None)
data.columns = ['ip','lsb8', 'lsb7', 'lsb6', 'x1','x2', 'x3', 'x4','x5', 'x6', 'x7','x8', 'x9', 'x10','x11', 'x12', 'x13','x14', 'x15', 'x16','x17', 'x18', 'x19','x20', 'x21', 'x22','x23', 'x24', 'x25','x26', 'x27', 'x28','x29', 'x30','x31', 'x32', 'y']
x = data.loc[:,'lsb6':'x32']
y = data.loc[:, 'y']
yy = array(y) + 64 
YY = to_categorical(yy,num_classes=129)
YY.shape
XX = array(x).reshape(x.shape[0],steps,1)

history = model.fit(XX, YY, epochs = 5, verbose=2,batch_size=64,callbacks=[],validation_split=0.2) #,class_weight=d)

model_json = model.to_json()
with open(Model + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(Model + ".h5")