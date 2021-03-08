from numpy import reshape, array
import pandas as pd
import csv
from tensorflow.keras.models import load_model

t1, t2, t3, t4, f, ts, fs, tn, fn = 0, 0, 0, 0, 0, 0, 0, 0, 0

data = pd.read_csv("/media/sf_ChampSim_shared/total_data/623_warmup50_60_33.txt", header = None)
data.columns = ['ip','lsb8', 'lsb7', 'lsb6', 'x1','x2', 'x3', 'x4','x5', 'x6', 'x7','x8', 'x9', 'x10','x11', 'x12', 'x13','x14', 'x15', 'x16','x17', 'x18', 'x19','x20', 'x21', 'x22','x23', 'x24', 'x25','x26', 'x27', 'x28','x29', 'x30','x31', 'x32', 'y']
x = data.loc[:,'lsb6':'x32']
y = data.loc[:, 'y']
yy = array(y) + 64 
xx = array(x) + 64

Model = "623_transformer"
model = load_model("/media/sf_ChampSim_shared/models/" + Model, compile = False)

for i in range(0,xx.shape[0]):
    print(i)
    a = model.predict(xx[i].reshape(1,33))
    b = yy[i]  
    c = array(x)[i][-1]
    pred1 = list(a[0]).index(max(list(a[0])))-64
    a[0][list(a[0]).index(max(list(a[0])))] = 0
    pred2 = list(a[0]).index(max(list(a[0])))-64
    a[0][list(a[0]).index(max(list(a[0])))] = 0
    pred3 = list(a[0]).index(max(list(a[0])))-64
    a[0][list(a[0]).index(max(list(a[0])))] = 0
    pred4 = list(a[0]).index(max(list(a[0])))-64
    right = yy[i]-64
    last_delta = c
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
out.write("{} t1={},t2={},t3={},t4={},f={}, acc1={}, acc2={}, acc3={} ,acc4={}, ts={}, fs={}, acc={}, tn={}, fn={}, acc={}\n".format(Model, t1, t2, t3, t4, f, t1/s, (t1+t2)/s, (t1+t2+t3)/s, (t1+t2+t3+t4)/s, ts, fs, ts/(ts+fs), tn, fn, tn/(tn+fn)))
out.close()