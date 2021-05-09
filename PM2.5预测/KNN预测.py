from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mglearn.tools
from pylab import *

mpl.rcParams['font.sans-serif'] = ['stsong']#显示中文
mpl.rcParams['font.size']=30#14
plt.rcParams['axes.unicode_minus'] = False#显示负号




data_train=pd.read_excel('train.xlsx',index_col=0)
data_text=pd.read_excel('test.xlsx',index_col=0)

print(data_train.corr())
X_train=data_train.iloc[:,0:-1]
X_text=data_text.iloc[:,0:-1]
y_train=data_train.iloc[:,-1]
y_text=data_text.iloc[:,-1]

PM_dataframe=pd.DataFrame(X_train,columns=data_train.iloc[:,0:-1].columns)
grr=pd.plotting.scatter_matrix(PM_dataframe,c=y_train,figsize=(15,15),marker='o',
                               hist_kwds={'bins':20},s=60,alpha=0.8,cmap=mglearn.cm3)
plt.show()
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
pred=knn.predict(X_text)
print(knn.score(X_text,y_text))#测试精度
print(confusion_matrix(y_text,pred))#混淆矩阵
scores=mglearn.tools.heatmap(
    confusion_matrix(y_text,pred),xlabel='预测等级',
    ylabel='真实等级',xticklabels=[1,2,3,4,5],yticklabels=[1,2,3,4,5],cmap=plt.cm.gray_r,fmt="%d")
#plt.title("混淆矩阵")
plt.gca().invert_yaxis()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("预测等级",fontsize=27)
plt.ylabel("真实等级",fontsize=27)
plt.show()
print(classification_report(y_text,pred))






