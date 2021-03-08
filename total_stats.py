from keras.models import model_from_json
from keras.utils import to_categorical
from numpy import array
from numpy import reshape
import pandas as pd
import csv

classes,features = 129,1
cells,steps = 8,33
t1, t2, t3, t4, f, ts, fs, tn, fn = 0, 0, 0, 0, 0, 0, 0, 0, 0

data = pd.read_csv("total_data/628_warmup50_10_33.txt", header = None)
data.columns = ['ip','lsb8', 'lsb7', 'lsb6', 'x1','x2', 'x3', 'x4','x5', 'x6', 'x7','x8', 'x9', 'x10','x11', 'x12', 'x13','x14', 'x15', 'x16','x17', 'x18', 'x19','x20', 'x21', 'x22','x23', 'x24', 'x25','x26', 'x27', 'x28','x29', 'x30','x31', 'x32', 'y']
x = data.loc[:,'lsb6':'x32']
y = data.loc[:, 'y']
yy = array(y) + 64 

#load model
Model = "628_{}_cells{}".format(steps,cells)
json_file = open("models/" + Model + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("models/" + Model + ".h5")
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


for j in range(0, x.shape[0] - 100, 100): 
    YY = to_categorical(yy[j:j+100],num_classes=129)
    XX = array(x[j:j+100]).reshape(100,steps,1)


#total stats
    for i in range(0,XX.shape[0]):
        print(i)
        a = model.predict(XX[i].reshape(1,XX.shape[1],XX.shape[2]))
        b = YY[i]  
        c = XX[i][-1]
        pred1 = list(a[0]).index(max(list(a[0])))-64
        a[0][list(a[0]).index(max(list(a[0])))] = 0
        pred2 = list(a[0]).index(max(list(a[0])))-64
        a[0][list(a[0]).index(max(list(a[0])))] = 0
        pred3 = list(a[0]).index(max(list(a[0])))-64
        a[0][list(a[0]).index(max(list(a[0])))] = 0
        pred4 = list(a[0]).index(max(list(a[0])))-64
        right = list(b).index(max(list(b)))-64
        last_delta = int(c[0])
        if (pred1 == right):
            t1 +=1
        elif (pred2 == right):
            t2 +=1
        elif (pred3 == right):
            t3 +=1
        elif (pred4 == right):
            t4 +=1
        else:
            f +=1
        if (last_delta == right):
            ts +=1
        else:
            fs +=1
        if (right == 1):
            tn +=1
        else:
            fn +=1
        s = t1+t2+t3+t4+f    

out = open('stats.txt', 'a')
out.write(Model + "    LSTM t1={},t2={},t3={},t4={},f={}, acc1={}, acc2={}, acc3={} ,acc4={}, ts={}, fs={}, acc={}, tn={}, fn={}, acc={}\n".format(t1, t2, t3, t4, f, t1/s, (t1+t2)/s, (t1+t2+t3)/s, (t1+t2+t3+t4)/s, ts, fs, ts/(ts+fs), tn, fn, tn/(tn+fn)))
out.close()