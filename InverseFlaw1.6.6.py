import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import GradientBoostingRegressor as GB

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
import math
#定义
def add_sheet(data,excel_writer,sheet_name):
    book = openpyxl.load_workbook(excel_writer.path)
    excel_writer.book = book
    data.to_excel(excel_writer = excel_writer,sheet_name = str(sheet_name))
    excel_writer.close()

def try_different_method(clf,n,x_test,y_test):
    clf.fit(dfx_train,dfy_train)
    y_pred = clf.predict(x_test)
    error = y_pred.reshape(y_pred.shape[0], 1) - y_test
    Abs = abs(error)
    AbsSort = sorted(Abs.values)
    ln_x_testTemp = []
    ln_x_testTemp.append(math.ceil(len(AbsSort)*0.9))
    ln_x_testTemp.append(math.ceil(len(AbsSort) * 0.8))
    ConIn = []
    ConIn.append(AbsSort[ln_x_testTemp[0] - 1])
    ConIn.append(AbsSort[ln_x_testTemp[1] - 1])

    # print(clf.feature_importances_)
    # data = pd.DataFrame(clf.feature_importances_)
    # data.columns = ['featuresimportances']
    # feature_important = pd.Series(clf.feature_importances_, index=x_test.columns).sort_values(ascending=False)
    # plt.bar(feature_important.index, clf.feature_importances_,align = 'center')
    # plt.xticks(size = 'small',rotation = 90,fontsize = 6)
    # plt.show()

    def LowerCount(a,b,y1,y2):
        num = 0
        for i in a:
            if i <= b:  # 可依需要修改条件
                num += 1
        percent = num / len(a)
        mserror = mean_squared_error(y1, y2)
        return percent,mserror
    ans = LowerCount(Abs.values,0.79,y_test,y_pred)
    df_1 = pd.DataFrame(y_test)
    df_2 = pd.DataFrame(y_pred)
    df_result = pd.concat([df_1,df_2],axis = 1)
    excel_writer =  pd.ExcelWriter('12result1.xlsx')
    add_sheet(df_result,excel_writer,n)
    print(ans, n, ConIn)



#读取训练集数据
df_train12 = pd.read_excel('all.xlsx',sheet_name = 3) #原始数据库#
df_wq = pd.read_excel('16inwuqing.xlsx',sheet_name = 3) #武清数据#

df_test16 = pd.read_excel('16inch202101.xlsx', sheet_name=2)
dfx_test = df_test16.iloc[:,
           [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]]
dfy_testL = df_test16.iloc[:, [6]]
dfy_testW = df_test16.iloc[:, [7]]
dfy_testD = df_test16.iloc[:, [8]]


df_train = df_train12.append(df_wq)
dfx_train, dfy_train = np.split(df_train, (40,), axis=1)
dfx_train = dfx_train.append(dfx_test)
dfy_train = dfy_train.append(dfy_testD)
#df_test12 =  pd.read_excel('12in.xlsx',sheet_name = i )
df_test16 = pd.read_excel('16inchflawnew.xlsx', sheet_name=3)
    #dfx_train = df_train12.iloc[:,[]]
    #dfy_train = df_train12.iloc[:,[]]


dfx_test = df_test16.iloc[:,[0,1,2,3,4,5,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]]
dfy_testL = df_test16.iloc[:, [6]]
dfy_testW = df_test16.iloc[:, [7]]
dfy_testD = df_test16.iloc[:, [8]]
#try_different_method(RFR(), 3, dfx_test, dfy_testL)
#try_different_method(RFR(), 3, dfx_test, dfy_testW)
try_different_method(RFR(), 3, dfx_test, dfy_testD)

