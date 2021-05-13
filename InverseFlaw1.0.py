import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
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
#x_train, x_test, y_train, y_test = TTS(x, y, train_size=0.75, random_state=75)
#print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))


#model = xgb.XGBRegressor(max_depth=4, learning_rate=0.08, n_estimators=160, silent=True, objective='reg:gamma')
#model.fit(x_train, y_train)
#cv_params = {'n_estimators':[600,800,1000,1200,1400,1600]}
#other_params = {'max_depth':6,'learning_rate':0.07,'min_child_weight':1,'seed':0,'subsample':0.8,
#                'colsample_tree':0.8,'gamma':0,'reg_alpha':0,'reg_lambda':1,'objective':'reg:gamma'}
kf = KFold(n_splits = 4,random_state = 42)
allpred = []
allerror = []
for train_index,test_index in kf.split(x):
    model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=1600, silent=True, objective='reg:gamma').fit(x.iloc[train_index,:], y.iloc[train_index,:])
    #model.fit(x[train_index], y[train_index])
    y_pred1 = model.predict(x.iloc[test_index,:])
    actuals = y.iloc[test_index,:]
    error = y_pred1.reshape(y_pred1.shape[0],1)-actuals
    Abs = abs(error)
    def LowerCount(a, b):
        num = 0
        for i in a:
            if i < b:  # 可依需要修改条件
                num += 1
        percent = num / len(a)
        return num, percent

    ans2 = LowerCount(Abs.values, 1.27)[1]
    print(LowerCount(Abs.values, 1.27))
    print(mean_squared_error(actuals,y_pred1))
    allpred.append(y_pred1.reshape(y_pred1.shape[0],1))
    allerror.append(ans2)

meanerror = np.mean(allerror)
print(meanerror)

#Kresults = CVS(model,x,y,cv = kfold)
#print(Kresults)
#print("CV Accuracy:%.2f%%(%.2f%%)"%(Kresults.mean()*100,Kresults.std()*100))

# 对测试集进行预测
# y_pred = model.predict(x_test)
# print(mean_squared_error(y_test, y_pred))
#
# error =y_pred.reshape(y_pred.shape[0],1) - y_test
# Abs = abs(error)


#def LowerCount(a,b):
#    num = 0
#   for i in a:
#       if i<b: #可依需要修改条件
#           num+=1
#   percent = num/len(a)
#   return num,percent
#调用
#LowerCount(Abs,10)
#print(LowerCount(Abs.values,1.27))

# plt.figure(figsize=(12,6), facecolor='w')
# ln_x_test = range(len(x_test))
#
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'prediction of depth')
# plt.show()