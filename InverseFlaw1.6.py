import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import GradientBoostingRegressor as GB
import math
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
import time
import datetime
from sklearn.metrics import mean_squared_error
import openpyxl

def add_sheet(data,excel_writer,sheet_name):
    book = openpyxl.load_workbook(excel_writer.path)
    excel_writer.book = book
    data.to_excel(excel_writer = excel_writer,sheet_name = str(sheet_name))
    excel_writer.close()

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
df16 = pd.read_excel('16inwuqing.xlsx',sheet_name = 3)
df_test0 = df16.sample(frac=0.2, axis=0)



#df_train = dfset2_1L.append([dfset1_1L,dfset1_2L,dfset1_3L,dfset1_4L,dfset1_5L,dfset2_2L,dfset2_3L,dfset2_4L,dfset3_1L,dfset3_2L,dfset3_3L,dfset3_4L,dfset3_5L])
#df_train = dfset2_1W.append([dfset1_1W,dfset1_2W,dfset1_3W,dfset1_4W,dfset1_5W,dfset2_2W,dfset2_3W,dfset2_4W,dfset3_1W,dfset3_2W,dfset3_3W,dfset3_4W,dfset3_5W])
df_train = dfset2_1D.append([dfset1_1D,dfset1_2D,dfset1_3D,dfset1_4D,dfset1_5D,dfset2_2D,dfset2_3D,dfset2_4D,dfset3_1D,dfset3_2D,dfset3_3D,dfset3_4D,dfset3_5D,df_test0])
# 0.5m/s 训练集：1234；测试集：5
# dfset1_1L = pd.read_excel('C1_0.5.xlsx',sheet_name = 1) #0.5m/s 1st length
# dfset1_2L = pd.read_excel('C1_0.5.xlsx',sheet_name = 2) #0.5m/s 2nd length
# dfset1_3L = pd.read_excel('C1_0.5.xlsx',sheet_name = 3) #0.5m/s 3rd length
# dfset1_4L = pd.read_excel('C1_0.5.xlsx',sheet_name = 4) #0.5m/s 4th length
# dfset1_5L = pd.read_excel('C1_0.5.xlsx',sheet_name = 5) #0.5m/s 5th length
# dfset1_1W = pd.read_excel('C1_0.5.xlsx',sheet_name = 6) #0.5m/s 1st width
# dfset1_2W = pd.read_excel('C1_0.5.xlsx',sheet_name = 7) #0.5m/s 2nd width
# dfset1_3W = pd.read_excel('C1_0.5.xlsx',sheet_name = 8) #0.5m/s 3rd width
# dfset1_4W = pd.read_excel('C1_0.5.xlsx',sheet_name = 9) #0.5m/s 4th width
# dfset1_5W = pd.read_excel('C1_0.5.xlsx',sheet_name = 10) #0.5m/s 5th depth
# dfset1_1D = pd.read_excel('C1_0.5.xlsx',sheet_name = 11) #0.5m/s 1st depth
# dfset1_2D = pd.read_excel('C1_0.5.xlsx',sheet_name = 12) #0.5m/s 2nd depth
# dfset1_3D = pd.read_excel('C1_0.5.xlsx',sheet_name = 13) #0.5m/s 3rd depth
# dfset1_4D = pd.read_excel('C1_0.5.xlsx',sheet_name = 14) #0.5m/s 4th depth
# dfset1_5D = pd.read_excel('C1_0.5.xlsx',sheet_name = 15) #0.5m/s 5th depth
# dfALLset_new = pd.read_excel('C1_0.5.xlsx',sheet_name = 1)
# df1 = pd.read_excel('16in.xlsx',sheet_name = 7 )
# df2 = pd.read_excel('16in_C1.xlsx',sheet_name = 7 )
#df = dfALL.append([df1,df2])

#df = pd.read_excel('10inchdataset.xlsx',sheet_name = 1)
x_train,y_train = np.split(df_train,(40,),axis = 1)

x_test,y_test = np.split(df16,(40,),axis = 1)
print ("样本数据量:%d, 特征个数：%d" % x_train.shape)
print ("target样本数据量:%d" % y_train.shape[0])

X_DF = pd.DataFrame(x_train)
X_DF.info()
X_DF.describe().T
X_DF.head()

X_DF16 = pd.DataFrame(x_test)
X_DF16.info()
X_DF16.describe().T
X_DF16.head()

#X_DF.hist(bins = 50,figsize = (30,20))
alltime = []
# other_params = {'max_depth': 5, 'n_estimators':1000,'min_child_weight': 3, 'subsample': 0.5, 'colsample_bytree': 0.8,
#                 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 2, 'learning_rate': 0.1, 'objective': 'reg:gamma',
#                 'seed': 0}
#for num in range(1,11):
#x_train, x_test, y_train, y_test = TTS(x,y , train_size=0.75, random_state=15)

##不同机器学习算法比较
other_params = {'max_depth':25, 'n_estimators':400,'min_child_weight': 1, 'subsample': 0.5, 'colsample_bytree': 0.8,
                 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 2, 'learning_rate': 0.026, 'objective': 'reg:gamma',
                 'seed': 0}
fig1 = plt.figure()
stsresult=[]
#fig1.tight_layout(h_pad=10.0)
def try_different_method(clf):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    error = y_pred.reshape(y_pred.shape[0], 1) - y_test
    Abs = abs(error)
    # ErrorSort = Abs.values.sort()
    # temp = math.ceil(Abs.values.shape[0]*0.9)
    # Con = ErrorSort(temp)
    ln_x_test = range(len(x_test))

    def LowerCount(a,b,y1,y2):
        num = 0
        for i in a:
            if i <=b:  # 可依需要修改条件
                num += 1
        percent = num / len(a)
        mserror = mean_squared_error(y1, y2)
        return percent,mserror
    ans = LowerCount(Abs.values,0.79,y_test,y_pred)
    df_1 = pd.DataFrame(y_test)
    df_2 = pd.DataFrame(y_pred)
    df_result = pd.concat([df_1,df_2],axis = 1)
    excel_writer =  pd.ExcelWriter('16wuqingresult.xlsx')
    add_sheet(df_result,excel_writer,1)
    print(ans)
    # ax1 = fig1.add_subplot(5,1,n)
    # plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
    # plt.plot(ln_x_test, y_pred, 'g-', lw=2, label=u'predict value')
    # plt.plot(ln_x_test, error, 'b-', lw=2, label=u'error')
    # plt.xlabel(u'num')
    # plt.ylabel(u'depth')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.title(u'prediction of depth')
    stsresult.append(ans)
    return stsresult
try_different_method(XGBR(**other_params))
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
# try_different_method(fig1,XGBR(**other_params),4)
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

