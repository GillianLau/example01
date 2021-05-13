import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR

from sklearn.datasets import load_boston
from sklearn.model_selection import StratifiedKFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from time import time
import datetime
from sklearn.metrics import mean_squared_error

import pymysql
import sqlalchemy
from sqlalchemy import create_engine


#df = inchsql.csv')
df = pd.read_excel('10inchdataset.xlsx',sheet_name = 2)
x,y = np.split(df,(10,),axis = 1)

print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)
X_DF.info()
X_DF.describe().T
X_DF.head()
#X_DF.hist(bins = 50,figsize = (30,20))
x_train, x_test, y_train, y_test = TTS(x, y, train_size=0.75, random_state=75)
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))


model = xgb.XGBRegressor(max_depth=4, learning_rate=0.08, n_estimators=160, silent=True, objective='reg:gamma')
model.fit(x_train, y_train)


# kfold = StratifiedKFold(n_splits = 10,random_state = 7)
# Kresults = CVS(model,x_train,y_train,cv = kfold)
# print(Kresults)
# print("CV Accuracy:%.2f%%(%.2f%%)"%(Kresults.mean()*100,Kresults.std()*100))

# 对测试集进行预测
y_pred = model.predict(x_test)
print(mean_squared_error(y_test, y_pred))
error =y_pred.reshape(y_pred.shape[0],1) - y_test
Abs = abs(error)


def LowerCount(a,b):
    num = 0
    for i in a:
        if i<b: #可依需要修改条件
            num+=1
    percent = num/len(a)
    return num,percent
#调用
#LowerCount(Abs,10)
print(LowerCount(Abs.values,1.27))

plt.figure(figsize=(12,6), facecolor='w')
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
plt.plot(ln_x_test, y_pred, 'g-', lw=2, label=u'predict value')
plt.plot(ln_x_test,error,'b-',lw=2,label=u'error')
plt.xlabel(u'num')
plt.ylabel(u'depth')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.title(u'prediction of depth')
plt.show()