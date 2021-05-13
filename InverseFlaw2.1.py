import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import GradientBoostingRegressor as GB
#from sklearn.externals import joblib
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
import time
import datetime
import pickle
from sklearn.metrics import mean_squared_error


# def read_xlsx(filename,n):

#     data = pd.read_excel(filename,sheet_name = n)
#     return data
#命名
#names = ['L','W','D']
for i in range(1,6):
    locals()['dfset1_{0}'.format(i)+'L'] = pd.read_excel('C1_0.5.xlsx',sheet_name=i-1)
    locals()['dfset1_{0}'.format(i) + 'W'] = pd.read_excel('C1_0.5.xlsx',sheet_name=i+4)
    locals()['dfset1_{0}'.format(i) + 'D'] = pd.read_excel('C1_0.5.xlsx',sheet_name=i+9)
for i in range(1,5):
    locals()['dfset2_{0}'.format(i)+'L'] = pd.read_excel('C1_1.0.xlsx',sheet_name=i-1)
    locals()['dfset2_{0}'.format(i) + 'W'] = pd.read_excel('C1_1.0.xlsx',sheet_name=i+3)
    locals()['dfset2_{0}'.format(i) + 'D'] = pd.read_excel('C1_1.0.xlsx',sheet_name=i+7)
for i in range(1,6):
    locals()['dfset3_{0}'.format(i)+'L'] = pd.read_excel('C1_1.5.xlsx',sheet_name=i-1)
    locals()['dfset3_{0}'.format(i) + 'W'] = pd.read_excel('C1_1.5.xlsx',sheet_name=i+4)
    locals()['dfset3_{0}'.format(i) + 'D'] = pd.read_excel('C1_1.5.xlsx',sheet_name=i+9)
#不同测试情况#
# dfset_org= pd.read_excel('all.xlsx',sheet_name = 4) #原始数据库#

# dfset1_1L = pd.read_excel('C1_1.5.xlsx',sheet_name = 0) #0.5m/s 1st length
# dfset1_2L = pd.read_excel('C1_1.5.xlsx',sheet_name = 1) #0.5m/s 2nd length
# dfset1_3L = pd.read_excel('C1_1.5.xlsx',sheet_name = 2) #0.5m/s 3rd length
# dfset1_4L = pd.read_excel('C1_1.5.xlsx',sheet_name = 3) #0.5m/s 4th length
# dfset1_5L = pd.read_excel('C1_1.5.xlsx',sheet_name = 4) #0.5m/s 5th length
# dfset2_1L = pd.read_excel('C1_1.0.xlsx',sheet_name = 0) #0.5m/s 1st length
# dfset2_2L = pd.read_excel('C1_1.0.xlsx',sheet_name = 1) #0.5m/s 2nd length
# dfset2_3L = pd.read_excel('C1_1.0.xlsx',sheet_name = 2) #0.5m/s 3rd length
# dfset2_4L = pd.read_excel('C1_1.0.xlsx',sheet_name = 3) #0.5m/s 4th length

# dfset1_1W = pd.read_excel('C1_0.5.xlsx',sheet_name = 5) #0.5m/s 1st width
# dfset1_2W = pd.read_excel('C1_0.5.xlsx',sheet_name = 6) #0.5m/s 2nd width
# dfset1_3W = pd.read_excel('C1_0.5.xlsx',sheet_name = 7) #0.5m/s 3rd width
# dfset1_4W = pd.read_excel('C1_0.5.xlsx',sheet_name = 8) #0.5m/s 4th width
# dfset1_5W = pd.read_excel('C1_0.5.xlsx',sheet_name = 9) #0.5m/s 5th depth
# dfset1_1D = pd.read_excel('C1_0.5.xlsx',sheet_name = 10) #0.5m/s 1st depth
# dfset1_2D = pd.read_excel('C1_0.5.xlsx',sheet_name = 11) #0.5m/s 2nd depth
# dfset1_3D = pd.read_excel('C1_0.5.xlsx',sheet_name = 12) #0.5m/s 3rd depth
# dfset1_4D = pd.read_excel('C1_0.5.xlsx',sheet_name = 13) #0.5m/s 4th depth
# dfset1_5D = pd.read_excel('C1_0.5.xlsx',sheet_name = 14) #0.5m/s 5th depth

       # 0.5m/s 训练集：1234；测试集：5 长度
# df_train = dfset1_1L.append([dfset1_2L,dfset1_3L,dfset1_4L,dfset2_1L,dfset2_2L,dfset2_2L,dfset2_2L,dfset1_5L])
# df_test = dfset1_5L
# df_train = dfset2_1D.append([dfset1_1D,dfset1_2D,dfset1_3D,dfset1_4D,dfset1_5D,dfset2_2D,dfset2_3D,dfset2_4D,dfset3_1D,dfset3_2D,dfset3_3D,dfset3_4D])
# df_test = dfset3_5D
# df_train = dfset2_1D.append([dfset2_2D,dfset2_3D,dfset2_4D,dfset3_1D,dfset3_2D,dfset3_2D,dfset3_2D])
# df_test = dfset3_5D
# df_train = dfset2_1D.append([dfset2_2D,dfset2_3D,dfset2_4D,dfset3_1D,dfset3_2D,dfset3_2D,dfset3_2D])
# df_test = dfset3_5D
# dfALLset_new = pd.read_excel('C1_0.5.xlsx',sheet_name = 1)
# df1 = pd.read_excel('16in.xlsx',sheet_name = 7 )
# df2 = pd.read_excel('16in_C1.xlsx',sheet_name = 7 )
# df = dfALL.append([df1,df2])

# 随机分 #
#df_all = []
print('dim1:',dfset1_1D.shape)
df_all = dfset1_1D.append([dfset1_2D,dfset1_3D,dfset1_4D,dfset1_5D,dfset2_1D,dfset2_2D,dfset2_3D,dfset2_4D,dfset3_1D,dfset3_2D,dfset3_3D,dfset3_4D,dfset3_5D])
df_dim1 = np.array([dfset1_1D.values,dfset1_2D.values,dfset1_3D.values,dfset1_4D.values,dfset1_5D.values,dfset2_1D.values,dfset2_2D.values,dfset2_3D.values,dfset2_4D.values,dfset3_1D.values,dfset3_2D.values,dfset3_3D.values,dfset3_4D.values,dfset3_5D.values])
#df_cell = df_dim1.reshape(14,169,41)
#print(dfset1_2D,dfset1_1D)
print('dim2:',df_all.shape)
print('dim3:',df_dim1.shape)
x,y = np.split(df_all,(40,),axis = 1)
x_train, x_test, y_train, y_test = TTS(x, y, train_size=0.95, random_state=24)

clf = RFR()
clf.fit(x_train,y_train)
with open('clf.pickle','wb') as fw:
    pickle.dump(clf,fw)
with open('clf.pickle','rb') as fr:
    new_rfr = pickle.load(fr)

def LowerCount(a, b, y1, y2):
    num = 0
    for i in a:
        if i < b:  # 可依需要修改条件
            num += 1
    percent = num / len(a)
    mserror = mean_squared_error(y1, y2)
    return percent, mserror

for t in range(0,df_dim1.shape[0]+1):
    df_test = pd.DataFrame(df_dim1[t])
    x_test0,y_test0 = np.split(df_test,(40,),axis = 1)
    y_pred0 = new_rfr.predict(x_test0)
    error = y_pred0.reshape(y_pred0.shape[0], 1) - y_test0
    Abs = abs(error)
    ln_x_test = range(len(x_test0))
    ans = LowerCount(Abs.values, 0.79, y_test0, y_pred0)
    print(ans)
#df = pd.read_excel('10inchdataset.xlsx',sheet_name = 1)
#x_train,y_train = np.split(df_train,(40,),axis = 1)
#df16 = pd.read_excel('16in_C1.xlsx',sheet_name = 10)
#x_test,y_test = np.split(df_test,(40,),axis = 1)
# print ("样本数据量:%d, 特征个数：%d" % x_train.shape)
# print ("target样本数据量:%d" % y_train.shape[0])

# X_DF = pd.DataFrame(x_train)
# X_DF.info()
# X_DF.describe().T
# X_DF.head()
#
# X_DF16 = pd.DataFrame(x_test)
# X_DF16.info()
# X_DF16.describe().T
# X_DF16.head()

#X_DF.hist(bins = 50,figsize = (30,20))
#alltime = []
# other_params = {'max_depth': 5, 'n_estimators':1000,'min_child_weight': 3, 'subsample': 0.5, 'colsample_bytree': 0.8,
#                 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 2, 'learning_rate': 0.1, 'objective': 'reg:gamma',
#                 'seed': 0}
#for num in range(1,11):
#x_train, x_test, y_train, y_test = TTS(x,y , train_size=0.75, random_state=15)

##不同机器学习算法比较
# other_params = {'max_depth':25, 'n_estimators':400,'min_child_weight': 1, 'subsample': 0.5, 'colsample_bytree': 0.8,
#                  'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 2, 'learning_rate': 0.026, 'objective': 'reg:gamma',
#                  'seed': 0}

# print(new_rfr.predict(x_test0))


# fig1 = plt.figure()
# stsresult=[]


#fig1.tight_layout(h_pad=10.0)
# def try_different_method(Fig1,clf,n):
#     clf.fit(x_train,y_train)
    # with open('clf.pickle','wb') as fw:
    #     pickle.dump(clf,fw)
    # with open('clf.pickle','rb') as fr:
    #     new_clf1 = pickle.load(fr)
    # print(new_clf1.predict(dfset2_1D[0:39]))
#     y_pred = clf.predict(x_test)
#     error = y_pred.reshape(y_pred.shape[0], 1) - y_test
#     Abs = abs(error)
#     ln_x_test = range(len(x_test))
#
#     def LowerCount(a,b,y1,y2):
#         num = 0
#         for i in a:
#             if i < b:  # 可依需要修改条件
#                 num += 1
#         percent = num / len(a)
#         mserror = mean_squared_error(y1, y2)
#         return percent,mserror
#     ans = LowerCount(Abs.values,0.79,y_test,y_pred)
#     print(ans)
#     ax1 = fig1.add_subplot(5,1,n)
#     plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
#     plt.plot(ln_x_test, y_pred, 'g-', lw=2, label=u'predict value')
#     plt.plot(ln_x_test, error, 'b-', lw=2, label=u'error')
#     plt.xlabel(u'num')
#     plt.ylabel(u'depth')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#     plt.title(u'prediction of depth')
#     stsresult.append(ans)
#     return stsresult
# ststime = []
# starttime1= time.time()
# try_different_method(fig1,LinearR(),1)
# costtime1 = time.time()-starttime1
# ststime.append(costtime1)
# starttime2= time.time()
# try_different_method(fig1,RFR(),2)
# costtime2 = time.time()-starttime2
# ststime.append(costtime2)
# starttime3= time.time()
# try_different_method(fig1,DTR(),3)
# costtime3 = time.time()-starttime3
# ststime.append(costtime3)
# starttime4= time.time()
# try_different_method(fig1,XGBR(),4)
# costtime4 = time.time()-starttime4
# ststime.append(costtime4)
# starttime5= time.time()
# try_different_method(fig1,GB(),5)
# costtime5 = time.time()-starttime5
# ststime.append(costtime5)
# print(ststime)

# plt.tight_layout()
# plt.show()
# df1 = pd.DataFrame(stsresult,index = ['linear','RFR','DTR','XGBR','GB'],columns=['TargetRate','MSE'])
# df1.plot(kind = 'bar',alpha = 0.5)
# plt.grid(True)
# plt.show()
# df2 = pd.DataFrame(ststime,index= ['linear','RFR','DTR','XGBR'],columns=['TIME'])
# df2.plot(kind = 'bar',alpha=0.5)
# plt.grid(True)
# plt.title(u'TIME')
# plt.show()

