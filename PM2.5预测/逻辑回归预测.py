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

mpl.rcParams['font.sans-serif'] = ['stsong']#显示中文
mpl.rcParams['font.size']=30
plt.rcParams['axes.unicode_minus'] = False#显示负号

data_train=pd.read_excel('train.xlsx',index_col=0)
data_text=pd.read_excel('test.xlsx',index_col=0)

X_train=data_train.iloc[:,0:-1]
X_text=data_text.iloc[:,0:-1]
y_train=data_train.iloc[:,-1]
y_text=data_text.iloc[:,-1]


scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_text_scaled=scale.transform(X_text)
poly=PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly=poly.transform(X_train_scaled)
X_text_poly=poly.transform(X_text_scaled)

Log=LogisticRegression(C=100).fit(X_train_poly,y_train)
pred=Log.predict(X_text_poly)
print(Log.score(X_text_poly,y_text))#测试精度
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


