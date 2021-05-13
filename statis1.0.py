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
list = []
L=[]
W=[]
D=[]
listL = []
listW = []
listD = []
# for i in range(1,16):
#     locals()['dfset_{0}'.format(i)] = pd.read_excel('换管前0105.xlsx', sheet_name=i - 1)
#     value = pd.read_excel('换管前0105.xlsx', sheet_name=i - 1)
#     list.append(value.values)
    #print(list[0][3,4])

for i in range(1,16):
    locals()['dfset_{0}'.format(i)] = pd.read_excel('换管后0106.xlsx', sheet_name=i - 1)
    value = pd.read_excel('换管后0106.xlsx', sheet_name=i - 1)
    list.append(value.values)


for s in range(1,32):
    for t in range(1,16):
      L.append(list[t - 1][s - 1, 2])

      W.append(list[t - 1][s - 1, 4])

      D.append(list[t - 1][s - 1, 6])
    listL.append(L)
    listW.append(W)
    listD.append(D)
    L=[]
    W=[]
    D=[]
Laver = []
Waver = []
Daver = []
for s in range(1,32):
    Laver.append(np.mean(listL[s - 1]))
    Waver.append(np.mean(listW[s - 1]))
    Daver.append(np.mean(listD[s - 1]))
listLC = []
listWC = []
listDC = []
for t in range(1,32):
    listLC.append(listL[t - 1][:] - Laver[t - 1])
    listWC.append(listW[t - 1][:] - Waver[t - 1])
    listDC.append(listD[t - 1][:] - Daver[t - 1])

##绘图
plt.rcParams['font.sans-serif']=['Simhei']
plt.rcParams['axes.unicode_minus']=False

for ii in range(1,32):
    x = np.ones(15)
    x_values = x[:]*ii
    l_values = listLC[ii-1]
    plt.scatter(x_values,l_values,c = l_values,cmap=plt.cm.YlOrRd,edgecolors = (0,0,0),s = 20)
plt.xlim((0,32))
plt.ylim((-30,30))
plt.title('长度量化结果一致性统计图')
plt.grid()
plt.show()

for ii in range(1,32):
    x = np.ones(15)
    x_values = x[:]*ii
    w_values = listWC[ii-1]
    plt.scatter(x_values,w_values,c = w_values,cmap=plt.cm.Greens,edgecolors = (0,0,0),s = 20)
plt.xlim((0,32))
plt.ylim((-30,30))
plt.title('宽度量化结果一致性统计图')
plt.grid()
plt.show()

for ii in range(1,32):
    x = np.ones(15)
    x_values = x[:]*ii
    d_values = listDC[ii-1]
    plt.scatter(x_values,d_values,c = d_values,cmap=plt.cm.Reds,edgecolors = (0,0,0),s = 20)
plt.xlim((0,32))
plt.ylim((-1.5,1.5))
plt.title('深度量化结果一致性统计图')
plt.grid()
plt.show()

print()