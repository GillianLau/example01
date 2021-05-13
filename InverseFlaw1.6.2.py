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
def add_sheet(data,excel_writer,sheet_name):
    book = openpyxl.load_workbook(excel_writer.path)
    excel_writer.book = book
    data.to_excel(excel_writer = excel_writer,sheet_name = str(sheet_name))
    excel_writer.close()





def try_different_method(clf,n,x_test,y_test):
    clf.fit(x_train,y_train)
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

    def LowerCount(a,b,y1,y2):
        num = 0
        for i in a:
            if i <= b:  # 可依需要修改条件
                num += 1
        percent = num / len(a)
        mserror = mean_squared_error(y1, y2)
        return percent,mserror
    ans = LowerCount(Abs.values,10,y_test,y_pred)
    df_1 = pd.DataFrame(y_test)
    df_2 = pd.DataFrame(y_pred)
    df_result = pd.concat([df_1,df_2],axis = 1)
    excel_writer =  pd.ExcelWriter('16C2width.xlsx')
    add_sheet(df_result,excel_writer,n)
    print(ans, n, ConIn)
    # 特征重要度
    # features = list(x_test.columns)
    # importances = clf.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # num_features = len(importances)
    #
    # # 将特征重要度以柱状图展示
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(num_features), importances[indices], color="g", align="center")
    # plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
    # plt.xlim([-1, num_features])
    # plt.show()
    #writer.close()

# def read_xlsx(filename,n):

#     data = pd.read_excel(filename,sheet_name = n)
#     return data
#命名
for i in range(1,6):
    locals()['dfset1_{0}'.format(i)+'L'] = pd.read_excel('C1_0.5.xlsx',sheet_name=i-1)
    locals()['dfset1_{0}'.format(i) + 'W'] = pd.read_excel('C1_0.5.xlsx',sheet_name=i+4)
    locals()['dfset1_{0}'.format(i) + 'D'] = pd.read_excel('C1_0.5.xlsx',sheet_name=i+9)
for i in range(1,6):
    locals()['dfset2_{0}'.format(i)+'L'] = pd.read_excel('C1_1.0.xlsx',sheet_name=i-1)
    locals()['dfset2_{0}'.format(i) + 'W'] = pd.read_excel('C1_1.0.xlsx',sheet_name=i+4)
    locals()['dfset2_{0}'.format(i) + 'D'] = pd.read_excel('C1_1.0.xlsx',sheet_name=i+9)
for i in range(1,6):
    locals()['dfset3_{0}'.format(i)+'L'] = pd.read_excel('C1_1.5.xlsx',sheet_name=i-1)
    locals()['dfset3_{0}'.format(i) + 'W'] = pd.read_excel('C1_1.5.xlsx',sheet_name=i+4)
    locals()['dfset3_{0}'.format(i) + 'D'] = pd.read_excel('C1_1.5.xlsx',sheet_name=i+9)

#names = ['L','W','D']
# for i in range(1,6):
#     locals()['dfset1_{0}'.format(i)+'L'] = pd.read_excel('16in_C2_0.5.xlsx',sheet_name=i-1)
#     locals()['dfset1_{0}'.format(i) + 'W'] = pd.read_excel('16in_C2_0.5.xlsx',sheet_name=i+4)
#     locals()['dfset1_{0}'.format(i) + 'D'] = pd.read_excel('16in_C2_0.5.xlsx',sheet_name=i+9)
# for i in range(1,6):
#     locals()['dfset2_{0}'.format(i)+'L'] = pd.read_excel('16in_C2-1.xlsx',sheet_name=i-1)
#     locals()['dfset2_{0}'.format(i) + 'W'] = pd.read_excel('16in_C2-1.xlsx',sheet_name=i+4)
#     locals()['dfset2_{0}'.format(i) + 'D'] = pd.read_excel('16in_C2-1.xlsx',sheet_name=i+9)
# for i in range(1,6):
#     locals()['dfset3_{0}'.format(i)+'L'] = pd.read_excel('16in_C2-1.5.xlsx',sheet_name=i-1)
#     locals()['dfset3_{0}'.format(i) + 'W'] = pd.read_excel('16in_C2-1.5.xlsx',sheet_name=i+4)
#     locals()['dfset3_{0}'.format(i) + 'D'] = pd.read_excel('16in_C2-1.5.xlsx',sheet_name=i+9)
df_trainL = dfset1_1L.append([dfset1_2L,dfset1_3L,dfset1_4L,dfset1_5L,dfset2_1L,dfset2_2L,dfset2_3L,dfset2_4L,dfset2_5L,dfset3_1L,dfset3_2L,dfset3_3L,dfset3_4L,dfset3_5L])
df_trainW = dfset1_1W.append([dfset1_2W,dfset1_3W,dfset1_4W,dfset1_5W,dfset2_1W,dfset2_2W,dfset2_3W,dfset2_4W,dfset2_5W,dfset3_1W,dfset3_2W,dfset3_3W,dfset3_4W,dfset3_5W])
df_trainD = dfset1_1D.append([dfset1_2D,dfset1_3D,dfset1_4D,dfset1_5D,dfset1_1D,dfset2_2D,dfset2_3D,dfset2_4D,dfset2_5D,dfset3_1D,dfset3_2D,dfset3_3D,dfset3_4D,dfset3_5D])
df_list1L = [dfset1_1L,dfset1_2L,dfset1_3L,dfset1_4L,dfset1_5L];
df_list1W = [dfset1_1W,dfset1_2W,dfset1_3W,dfset1_4W,dfset1_5W];
df_list1D = [dfset1_1D,dfset1_2D,dfset1_3D,dfset1_4D,dfset1_5D];

df_list2L = [dfset2_1L,dfset2_2L,dfset2_3L,dfset2_4L,dfset2_5L];
df_list2W = [dfset2_1W,dfset2_2W,dfset2_3W,dfset2_4W,dfset2_5W];
df_list2D = [dfset2_1D,dfset2_2D,dfset2_3D,dfset2_4D,dfset2_5D];

df_list3L = [dfset3_1L,dfset3_2L,dfset3_3L,dfset3_4L,dfset3_5L];
df_list3W = [dfset3_1W,dfset3_2W,dfset3_3W,dfset3_4W,dfset3_5W];
df_list3D = [dfset3_1D,dfset3_2D,dfset3_3D,dfset3_4D,dfset3_5D];
df_train12= pd.read_excel('all.xlsx',sheet_name = 2) #原始数据库#
df_wq = pd.read_excel('16inwuqing.xlsx',sheet_name = 2)#武清数据#
for t in range(1,2):
    # for tt in range(1, 5):
    #     df_train = dfset3_1D.append(df_list3D[tt])
    #df_train = df_train12.append(df_wq)
    #df_test = df_list1L[t-1]
    df_train = df_trainW
    df_test = df_wq
    x_train, y_train = np.split(df_train, (40,), axis=1)
    x_test, y_test = np.split(df_test, (40,), axis=1)
    try_different_method(RFR(), t, x_test, y_test)