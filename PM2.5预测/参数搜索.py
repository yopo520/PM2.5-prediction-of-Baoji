import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    data_train=pd.read_excel('train.xlsx',index_col=0)
    X_train=data_train.iloc[:,0:-1]
    y_train=data_train.iloc[:,-1]>=4
    param_grid={
        'n_neighbors':[1,2,3,4,5,6,7]
    }
    loo=LeaveOneOut()#留一法交叉验证
    grid_search=GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring='roc_auc',cv=8)
    grid=grid_search.fit(X_train,y_train)
    #print(grid.best_params_)#模型评估n_neighbors选择2最好

    #逻辑回归模型参数选择
    pipe=make_pipeline(
        StandardScaler(),
        PolynomialFeatures(),
        LogisticRegression()
    )
    param_grid1={
        'polynomialfeatures__degree': [1, 2, 3],
        'logisticregression__C':[0.001,0.01,0.1,1,10,100]
    }
    grid_search1 = GridSearchCV(pipe, param_grid=param_grid1, scoring='roc_auc',cv=8,n_jobs=-1)
    grid1=grid_search1.fit(X_train,y_train)
    #print(grid1.best_params_)#找到C为100，degree为2最好


    param_grid2={
        'n_estimators':(10,20,50,100),
        'max_features':[1,2,3]
    }
    grid_search2=GridSearchCV(RandomForestClassifier(random_state=0),param_grid=param_grid2,scoring='roc_auc',cv=8)
    grid2=grid_search2.fit(X_train,y_train)
    #print(grid2.best_params_)#选用n_estimators为50，max_features为1最好
    param_grid3 = {
        'n_estimators': (10, 20, 50, 100),
        'max_depth': (1,2,3,4,5),
        'learning_rate': [0.001,0.01,0.1,1,10,100]
    }
    grid_search3 = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid=param_grid3, scoring='roc_auc', cv=8)
    grid3 = grid_search3.fit(X_train, y_train)
    print(grid3.best_params_)#learning_rate为0.1，max_depth为2,n_estimators设置20最好




