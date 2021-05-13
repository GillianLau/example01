import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import gradient_boosting as GB

from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
import time
import datetime
from sklearn.metrics import mean_squared_error
## 数据导入
df = pd.read_excel('10inchdataset.xlsx',sheet_name = 7)
x,y = np.split(df,(29,),axis = 1)

print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)
X_DF.info()
X_DF.describe().T
X_DF.head()
#X_DF.hist(bins = 50,figsize = (30,20))
x_train, x_test, y_train, y_test = TTS(x, y, train_size=0.75, random_state=75)

# cv_params = {'max_depth':[3,4,5,6,7,8,9,10],'min_child_weight':[1,2,3,4,5,6]}
# other_params = {'max_depth':4,'n_estimators': 1400,'min_child_weight':1,'seed':0,'subsample':0.8,
#                 'colsample_tree':0.8,'gamma':0,'reg_alpha':0,'reg_lambda':1,'objective':'reg:gamma'}
# print("Params Optimization:")
# model = xgb.XGBRegressor()
# optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,verbose = 1)
# optimized_gbm.fit(x_train,y_train)
# y_pred2 = optimized_gbm.predict(x_test)
# error2 = y_pred2.reshape(y_pred2.shape[0],1)-y_test
# Abs = abs(error2)
# def LowerCount(a, b):
#     num = 0
#     for i in a:
#         if i < b:  # 可依需要修改条件
#             num += 1
#     percent = num / len(a)
#     return num, percent
# ans2 = LowerCount(Abs.values, 1.27)
# print(ans2)
# print(mean_squared_error(y_test,y_pred2))
# evaluate_result = optimized_gbm.cv_results_['std_test_score']
# print('每次运行结果：{0}'.format(evaluate_result))
# print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
# print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
# print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))
#
# plt.figure()
# ln_x_test = range(len(x_test))
# #ax1 = plt.subplot(211)
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'xgb prediction of depth')
# plt.show()
# # 寻优可视化分析
# #list = [600,800,1000,1200,1400,1600,1800]
# plt.figure()
# plt.subplot(221)
# plt.plot(evaluate_result,'go-')
# plt.grid(True)
# # for a, b in zip( evaluate_result):
# #     plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_test_score')
# plt.subplot(222)
# plt.plot(optimized_gbm.cv_results_['mean_test_score'],'y*-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['mean_test_score']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_test_score')
# plt.subplot(223)
# plt.plot(optimized_gbm.cv_results_['std_score_time'],'go-')
# plt.grid(True)

# gamma
# cv_params = {'gamma':[0,0.1,0.2,0.3,0.4,0.5,0.6]}
# other_params = {'max_depth':8,'min_child_weight':4}
# print("Params Optimization:")
# model = xgb.XGBRegressor(**other_params)
# optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,verbose = 1)
# optimized_gbm.fit(x_train,y_train)
# y_pred2 = optimized_gbm.predict(x_test)
# error2 = y_pred2.reshape(y_pred2.shape[0],1)-y_test
# Abs = abs(error2)
# def LowerCount(a, b):
#     num = 0
#     for i in a:
#         if i < b:  # 可依需要修改条件
#             num += 1
#     percent = num / len(a)
#     return num, percent
# ans2 = LowerCount(Abs.values, 1.27)
# print(ans2)
# print(mean_squared_error(y_test,y_pred2))
# evaluate_result = optimized_gbm.cv_results_['std_test_score']
# print('每次运行结果：{0}'.format(evaluate_result))
# print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
# print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
# print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))
#
# plt.figure()
# ln_x_test = range(len(x_test))
# #ax1 = plt.subplot(211)
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'xgb prediction of depth')
# plt.show()
# # 寻优可视化分析
# list = [0,0.1,0.2,0.3,0.4,0.5,0.6]
# plt.figure()
# plt.subplot(221)
# plt.plot(list,evaluate_result,'go-')
# plt.grid(True)
# for a, b in zip(list, evaluate_result):
#     plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_test_score')
# plt.subplot(222)
# plt.plot(list,optimized_gbm.cv_results_['mean_test_score'],'y*-')
# plt.grid(True)
# for a, b in zip(list,optimized_gbm.cv_results_['mean_test_score']):
#     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_test_score')
# plt.subplot(223)
# plt.plot(list,optimized_gbm.cv_results_['std_score_time'],'go-')
# plt.grid(True)
# for a, b in zip(list,optimized_gbm.cv_results_['std_score_time']):
#     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_score_time')
# plt.subplot(224)
# plt.plot(list,optimized_gbm.cv_results_['mean_score_time'],'y*-')
# plt.grid(True)
# for a, b in zip(list,optimized_gbm.cv_results_['mean_score_time']):
#     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_score_time')
# plt.tight_layout()
# plt.show()


#subsample colsample_bytree
# cv_params = {'subsample':[0.5,0.6,0.7,0.8,0.9],'colsample_bytree':[0.5,0.6,0.7,0.8,0.9]}
# other_params = {'max_depth':8,'min_child_weight':4,
#                 'gamma':0}
# print("Params Optimization:")
# model = xgb.XGBRegressor(**other_params)
# optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,verbose = 1)
# optimized_gbm.fit(x_train,y_train)
# y_pred2 = optimized_gbm.predict(x_test)
# error2 = y_pred2.reshape(y_pred2.shape[0],1)-y_test
# Abs = abs(error2)
# def LowerCount(a, b):
#     num = 0
#     for i in a:
#         if i < b:  # 可依需要修改条件
#             num += 1
#     percent = num / len(a)
#     return num, percent
# ans2 = LowerCount(Abs.values, 1.27)
# print(ans2)
# print(mean_squared_error(y_test,y_pred2))
# evaluate_result = optimized_gbm.cv_results_['std_test_score']
# print('每次运行结果：{0}'.format(evaluate_result))
# print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
# print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
# print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))
#
# plt.figure()
# ln_x_test = range(len(x_test))
# #ax1 = plt.subplot(211)
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'xgb prediction of depth')
# plt.show()
# # 寻优可视化分析
# #list = [600,800,1000,1200,1400,1600,1800]
# plt.figure()
# plt.subplot(221)
# plt.plot(evaluate_result,'go-')
# plt.grid(True)
# # for a, b in zip( evaluate_result):
# #     plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_test_score')
# plt.subplot(222)
# plt.plot(optimized_gbm.cv_results_['mean_test_score'],'y*-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['mean_test_score']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_test_score')
# plt.subplot(223)
# plt.plot(optimized_gbm.cv_results_['std_score_time'],'go-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['std_score_time']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_score_time')
# plt.subplot(224)
# plt.plot(optimized_gbm.cv_results_['mean_score_time'],'y*-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['mean_score_time']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_score_time')
# plt.tight_layout()
# plt.show()

# reg_alpha&reg_lambda
# cv_params = {'reg_alpha':[0,0.05,0.1,1,2,3],'reg_lambda':[0,0.05,0.1,1,2,3]}
# other_params = {'max_depth':8,'min_child_weight':4,'subsample':0.5,'colsample_bytree':0.8,
#                 'gamma':0,}
# print("Params Optimization:")
# model = xgb.XGBRegressor(**other_params)
# optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,verbose = 1)
# optimized_gbm.fit(x_train,y_train)
# y_pred2 = optimized_gbm.predict(x_test)
# error2 = y_pred2.reshape(y_pred2.shape[0],1)-y_test
# Abs = abs(error2)
# def LowerCount(a, b):
#     num = 0
#     for i in a:
#         if i < b:  # 可依需要修改条件
#             num += 1
#     percent = num / len(a)
#     return num, percent
# ans2 = LowerCount(Abs.values, 1.27)
# print(ans2)
# print(mean_squared_error(y_test,y_pred2))
# evaluate_result = optimized_gbm.cv_results_['std_test_score']
# print('每次运行结果：{0}'.format(evaluate_result))
# print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
# print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
# print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))
#
# plt.figure()
# ln_x_test = range(len(x_test))
# #ax1 = plt.subplot(211)
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'xgb prediction of depth')
# plt.show()
# # 寻优可视化分析
# #list = [600,800,1000,1200,1400,1600,1800]
# plt.figure()
# plt.subplot(221)
# plt.plot(evaluate_result,'go-')
# plt.grid(True)
# # for a, b in zip( evaluate_result):
# #     plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_test_score')
# plt.subplot(222)
# plt.plot(optimized_gbm.cv_results_['mean_test_score'],'y*-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['mean_test_score']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_test_score')
# plt.subplot(223)
# plt.plot(optimized_gbm.cv_results_['std_score_time'],'go-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['std_score_time']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_score_time')
# plt.subplot(224)
# plt.plot(optimized_gbm.cv_results_['mean_score_time'],'y*-')
# plt.grid(True)
# # for a, b in zip(optimized_gbm.cv_results_['mean_score_time']):
# #     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_score_time')
# plt.tight_layout()
# plt.show()

# learning_rate
# cv_params = {'learning_rate':[0.01,0.05,0.1,0.2]}
# other_params = {'max_depth':8,'min_child_weight':4,'subsample':0.5,'colsample_bytree':0.8,
#                  'gamma':0,'reg_alpha':0.1,'reg_lambda':2}
# print("Params Optimization:")
# model = xgb.XGBRegressor(**other_params)
# optimized_gbm =GridSearchCV(estimator = model,param_grid = cv_params,verbose = 1)
# optimized_gbm.fit(x_train,y_train)
# y_pred2 = optimized_gbm.predict(x_test)
# error2 = y_pred2.reshape(y_pred2.shape[0],1)-y_test
# Abs = abs(error2)
# def LowerCount(a, b):
#     num = 0
#     for i in a:
#         if i < b:  # 可依需要修改条件
#             num += 1
#     percent = num / len(a)
#     return num, percent
# ans2 = LowerCount(Abs.values, 1.27)
# print(ans2)
# print(mean_squared_error(y_test,y_pred2))
# evaluate_result = optimized_gbm.cv_results_['std_test_score']
# print('每次运行结果：{0}'.format(evaluate_result))
# print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
# print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
# print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))
#
# plt.figure()
# ln_x_test = range(len(x_test))
# #ax1 = plt.subplot(211)
# plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
# plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
# plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
# plt.xlabel(u'num')
# plt.ylabel(u'depth')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.title(u'xgb prediction of depth')
# plt.show()
# # 寻优可视化分析
# list = [0.01,0.05,0.1,0.2]
# plt.figure()
# plt.subplot(221)
# plt.plot(list,evaluate_result,'go-')
# plt.grid(True)
# for a, b in zip(list, evaluate_result):
#     plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_test_score')
# plt.subplot(222)
# plt.plot(list,optimized_gbm.cv_results_['mean_test_score'],'y*-')
# plt.grid(True)
# for a, b in zip(list,optimized_gbm.cv_results_['mean_test_score']):
#     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_test_score')
# plt.subplot(223)
# plt.plot(list,optimized_gbm.cv_results_['std_score_time'],'go-')
# plt.grid(True)
# for a, b in zip(list,optimized_gbm.cv_results_['std_score_time']):
#     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'std_score_time')
# plt.subplot(224)
# plt.plot(list,optimized_gbm.cv_results_['mean_score_time'],'y*-')
# plt.grid(True)
# for a, b in zip(list,optimized_gbm.cv_results_['mean_score_time']):
#     plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
# plt.title(u'mean_score_time')
# plt.tight_layout()
# plt.show()

#
cv_params = {'n_estimators':[1250,1300,1350,1400,1450,1500,1550]}
# other_params = {'max_depth':4,'learning_rate':0.03,'min_child_weight':1,'seed':0,'subsample':0.8,
#                 'colsample_tree':0.8,'gamma':0,'reg_alpha':0,'reg_lambda':1,'objective':'reg:gamma'}
other_params = {'max_depth':8,'min_child_weight':4,'subsample':0.5,'colsample_bytree':0.8,
                 'gamma':0,'reg_alpha':0.1,'reg_lambda':2,'learning_rate':0.1,'objective':'reg:gamma','seed':0}
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
print(mean_squared_error(y_test,y_pred2))
evaluate_result = optimized_gbm.cv_results_['std_test_score']
print('每次运行结果：{0}'.format(evaluate_result))
print('参数的最佳取值：{0}'.format(optimized_gbm.best_params_))
print('最佳模型得分：{0}'.format(optimized_gbm.best_score_))
print('最佳参数运行时间：{0}'.format(optimized_gbm.refit_time_))

plt.figure()
ln_x_test = range(len(x_test))
#ax1 = plt.subplot(211)
plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
plt.plot(ln_x_test, y_pred2, 'g-', lw=2, label=u'predict value')
plt.plot(ln_x_test,error2,'b-',lw=2,label=u'error')
plt.xlabel(u'num')
plt.ylabel(u'depth')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.title(u'xgb prediction of depth')
plt.show()
# 寻优可视化分析
list = [1250,1300,1350,1400,1450,1500,1550]
plt.figure()
plt.subplot(221)
plt.plot(list,evaluate_result,'go-')
for a, b in zip(list, evaluate_result):
    plt.text(a, b, round(b,4), ha='center', va='bottom', fontsize=5)
plt.title(u'std_test_score')
plt.grid(True)
plt.subplot(222)
plt.plot(list,optimized_gbm.cv_results_['mean_test_score'],'y*-')
for a, b in zip(list,optimized_gbm.cv_results_['mean_test_score']):
    plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
plt.title(u'mean_test_score')
plt.grid(True)
plt.subplot(223)
plt.plot(list,optimized_gbm.cv_results_['std_score_time'],'go-')
for a, b in zip(list,optimized_gbm.cv_results_['std_score_time']):
    plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
plt.title(u'std_score_time')
plt.grid(True)
plt.subplot(224)
plt.plot(list,optimized_gbm.cv_results_['mean_score_time'],'y*-')
for a, b in zip(list,optimized_gbm.cv_results_['mean_score_time']):
    plt.text(a, b,round(b,4), ha='center', va='bottom', fontsize=5)
plt.title(u'mean_score_time')
plt.grid(True)
plt.tight_layout()
plt.show()