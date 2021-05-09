from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import mglearn.tools
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score


mpl.rcParams['font.sans-serif'] = ['stsong']#显示中文
mpl.rcParams['font.size']=30
plt.rcParams['axes.unicode_minus'] = False#显示负号

data_train=pd.read_excel('train.xlsx',index_col=0)
data_text=pd.read_excel('test.xlsx',index_col=0)

X_train=data_train.iloc[:,0:-1]
X_text=data_text.iloc[:,0:-1]
y_train=data_train.iloc[:,-1]
y_text=data_text.iloc[:,-1]


gcf=GradientBoostingClassifier(max_depth=2,n_estimators=20,learning_rate=0.1,random_state=0).fit(X_train,y_train)
pred=gcf.predict(X_text)
print(gcf.score(X_text,y_text))#测试精度
print(gcf.score(X_train,y_train))#训练精度
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
print(gcf.feature_importances_)

