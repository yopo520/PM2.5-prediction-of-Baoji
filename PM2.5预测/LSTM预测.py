from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mglearn.tools
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence


mpl.rcParams['font.sans-serif'] = ['stsong']#显示中文
mpl.rcParams['font.size']=30
plt.rcParams['axes.unicode_minus'] = False#显示负号

data_train=pd.read_excel('train.xlsx',index_col=0)
data_text=pd.read_excel('test.xlsx',index_col=0)

X_train=data_train.iloc[:,0:-1]
X_text=data_text.iloc[:,0:-1]
y_train=data_train.iloc[:,-1]
y_text=data_text.iloc[:,-1]

y_train=np.array(y_train)
y_text=np.array(y_text)

X_train=np.array(X_train)
X_text=np.array(X_text)

mean=X_train.mean()
X_train=X_train.astype('float64')
X_train -=mean
std=X_train.std()
X_train /=std

mean=X_text.mean()
X_text=X_text.astype('float64')
X_text -=mean
std=X_text.std()
X_text /=std


print(len(np.unique(X_train)))#统计不同值的数量
print(len(np.unique(X_text)))



from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Bidirectional

model=Sequential()
model.add(Embedding(100,16))
model.add(Bidirectional(LSTM(16)))
model.add(Dense(6,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
history=model.fit(X_train,y_train,epochs=8,batch_size=32,validation_data=(X_text,y_text))

results=model.evaluate(X_text,y_text)
print(results)

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()
prdiction=model.predict(X_text)
for i in range(0,364):
    print(np.argmax(prdiction[i]))





