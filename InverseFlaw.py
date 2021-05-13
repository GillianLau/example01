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
df = pd.read_excel('10inchdataset.xlsx',sheet_name = 4)
x,y = np.split(df,(31,),axis = 1)

print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)
X_DF.info()
X_DF.describe().T
X_DF.head()
X_DF.hist(bins = 50,figsize = (30,20))
x_train, x_test, y_train, y_test = TTS(x, y, train_size=0.75, random_state=36)
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))


dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test,label = y_train)
# XGBoost模型构建
# 1. 参数构建
params = {'booster':'gbtree','max_depth':5, 'eta':0.78, 'silent':1, 'objective':'reg:linear','n_estimators': 1000}
num_round = 2
# 2. 模型训练
bst = xgb.train(params, dtrain, num_round)
#def xg_eval_mae(yhat, dtrain):
#    y = dtrain.get_label()
#    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))
#bst_cv1 = xgb.cv(params, dtrain, num_boost_round=50, nfold=3, seed=0,
#                feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)
#print ('CV score:', bst_cv1.iloc[-1,:]['test-mae-mean'])
#plt.figure()
#bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()
#交叉验证
# kfold = StratifiedKFold(n_splits = 10,random_state = 7)
# Kresults = CVS(bst,x_train,y_train,cv = kfold)
# print(Kresults)
# print("CV Accuracy:%.2f%%(%.2f%%)"%(Kresults.mean()*100,Kresults.std()*100))
# 3. 模型保存
bst.save_model('xgb.model')

y_pred = bst.predict(dtest)
print(mean_squared_error(y_test, y_pred))

# 4. 加载模型
bst2 = xgb.Booster()
bst2.load_model('xgb.model')
# 5 使用加载模型预测
y_pred2 = bst2.predict(dtest)
print(mean_squared_error(y_test, y_pred2))
error =y_pred2.reshape(y_pred2.shape[0],1) - y_test
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
plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')


plt.plot(ln_x_test,error,'b-',lw=2,label=u'error')
plt.xlabel(u'num')
plt.ylabel(u'depth')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.title(u'prediction of depth')
plt.show()

from xgboost import plot_importance
from matplotlib import pyplot
# 找出最重要的特征
plot_importance(bst,importance_type = 'cover')
pyplot.show()

#输出计数和百分比

import matplotlib.pyplot as plt
from xgboost import plot_tree
from graphviz import Digraph
import pydot
xgb.plot_tree(bst,num_trees = 0)
plt.rcParams['figure.figsize'] = [80,60]
plt.show()
print(y_pred2)