import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import gradient_boosting

from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
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
df = pd.read_excel('201910inch.xlsx',sheet_name = 4)
x,y = np.split(df,(41,),axis = 1)

print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)
X_DF.info()
X_DF.describe().T
X_DF.head()
#X_DF.hist(bins = 50,figsize = (30,20))
x_train, x_test, y_train, y_test = TTS(x, y, train_size=0.6, random_state=14)
#print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))


#model = xgb.XGBRegressor(max_depth=4, learning_rate=0.08, n_estimators=160, silent=True, objective='reg:gamma')
#model.fit(x_train, y_train)
cv_params = {'n_estimators':[100,400,600,800,1000,1200,1400,1600]}
other_params = {'max_depth':6,'learning_rate':0.05,'min_child_weight':1,'seed':0,'subsample':0.8,
                'colsample_tree':0.8,'gamma':0,'reg_alpha':0,'reg_lambda':1,'objective':'reg:gamma'}
# kf = KFold(n_splits = 4,random_state = 24)
# allpred = []
# allerror = []
# for train_index,test_index in kf.split(x):
#     model = xgb.XGBRegressor(**other_params)
#     #model.fit(x[train_index], y[train_index])
#     #optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,scoring = 'r2',cv = 5,verbose = 1,n_jobs = 4)
#     model.fit(x.iloc[train_index,:], y.iloc[train_index,:])
#     #evalute_result = optimized_gbm.grid_scores_
#     y_pred1 = model.predict(x.iloc[test_index,:])
#     #print('每轮迭代运行结果：{0}'.format(evalute_result ))
#     #print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
#     #print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
#     actuals = y.iloc[test_index,:]
#     error = y_pred1.reshape(y_pred1.shape[0],1)-actuals
#     Abs = abs(error)
#     def LowerCount(a, b):
#         num = 0
#         for i in a:
#             if i < b:  # 可依需要修改条件
#                 num += 1
#         percent = num / len(a)
#         return num, percent
#
#     ans = LowerCount(Abs.values, 1.27)
#     print(LowerCount(Abs.values, 1.27))
#     print(mean_squared_error(actuals,y_pred1))
#     allpred.append(y_pred1.reshape(y_pred1.shape[0],1))
#     allerror.append(ans[1])
#
# meanerror = np.mean(allerror)
# print(meanerror)

## 寻优 ##
print("Params Optimization:")
model = xgb.XGBRegressor(**other_params)
optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,verbose = 1)
optimized_gbm.fit(x_train,y_train)
y_pred2 = optimized_gbm.predict(x_test)
error2 = y_pred2.reshape(y_pred2.shape[0],1)-y_test
Abs = abs(error2)
def LowerCount(a, b):
    num = 0
    for i in a:
        if i < b:  # 可依需要修改条件
            num += 1
    percent = num / len(a)
    return num, percent
ans2 = LowerCount(Abs.values, 1.27)
print(ans2)
print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))

#Kresults = CVS(model,x,y,cv = kfold,
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

plt.figure(1)
ln_x_test = range(len(x_test))
ax1 = plt.subplot(211)
plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
plt.xlabel(u'num')
plt.ylabel(u'depth')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.title(u'xgb prediction of depth')
plt.show()
#
# ## 随机森林
# print('随机森林结果：')
# modelrf = RFR()
# cv_paramsrf = {'n_estimators':[600,800,1000,1200,1400,1600]}
# optimized_rf =GridSearchCV(estimator = modelrf,param_grid = cv_params,verbose = 1)
# optimized_rf.fit(x_train,y_train)
# y_pred3 = optimized_rf.predict(x_test)
# error3 = y_pred3.reshape(y_pred3.shape[0],1)-y_test
# Abs = abs(error3)
# def LowerCount(a, b):
#     num = 0
#     for i in a:
#         if i < b:  # 可依需要修改条件
#             num += 1
#     percent = num / len(a)
#     return num, percent
# ans3 = LowerCount(Abs.values, 1.27)
# print(ans3)
# print('参数的最佳取值：{0}'.format(optimized_rf.best_params_))
# print('最佳模型得分：{0}'.format(optimized_rf.best_score_))
# print('最佳参数运行时间：{0}'.format(optimized_rf.refit_time_))
#
# ax2 = plt.subplot(212)
# ln_x_test = range(len(x_test))
#
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred3, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error3,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'rf prediction of depth')
# plt.show()