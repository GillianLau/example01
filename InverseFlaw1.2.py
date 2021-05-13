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

df = pd.read_excel('allFlawData0915.xlsx',sheet_name = 4)
#df = pd.read_excel('10inchdataset.xlsx',sheet_name = 1)
x,y = np.split(df,(35,),axis = 1)



print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)


X_DF.info()
X_DF.describe().T
X_DF.head()

#X_DF.hist(bins = 50,figsize = (30,20))
alltime = []
other_params = {'max_depth':6, 'n_estimators':500,'min_child_weight': 8, 'subsample': 0.5, 'colsample_bytree': 0.8,
                 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 2, 'learning_rate': 0.026, 'objective': 'reg:gamma',
                 'seed': 0}
for num in range(1,11):
    x_train, x_test, y_train, y_test = TTS(x,y , train_size=0.75, random_state=14+num)

    ##不同机器学习算法比较
    fig1 = plt.figure()
    stsresult=[]

    #fig1.tight_layout(h_pad=10.0)
    def try_different_method(Fig1,clf,n):
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        error = y_pred.reshape(y_pred.shape[0], 1) - y_test
        Abs = abs(error)
        ln_x_test = range(len(x_test))

        def LowerCount(a, b,y1,y2):
            num = 0
            for i in a:
                if i < b:  # 可依需要修改条件
                    num += 1
            percent = num / len(a)
            mserror = mean_squared_error(y1, y2)
            return percent,mserror
        ans = LowerCount(Abs.values, 1.27,y_test,y_pred)
        print(ans)
        ax1 = fig1.add_subplot(5,1,n)
        plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'actual value')
        plt.plot(ln_x_test, y_pred, 'g-', lw=2, label=u'predict value')
        plt.plot(ln_x_test, error, 'b-', lw=2, label=u'error')
        plt.xlabel(u'num')
        plt.ylabel(u'depth')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.title(u'prediction of depth')
        stsresult.append(ans)
        return stsresult
    ststime = []
    starttime1= time.time()
    try_different_method(fig1,LinearR(),1)
    costtime1 = time.time()-starttime1
    ststime.append(costtime1)
    starttime2= time.time()
    try_different_method(fig1,RFR(),2)
    costtime2 = time.time()-starttime2
    ststime.append(costtime2)
    starttime3= time.time()
    try_different_method(fig1,DTR(),3)
    costtime3 = time.time()-starttime3
    ststime.append(costtime3)
    starttime4= time.time()
    try_different_method(fig1,XGBR(**other_params),4)
    costtime4 = time.time()-starttime4
    ststime.append(costtime4)
    starttime5= time.time()
    try_different_method(fig1,GB(),5)
    costtime5 = time.time()-starttime5
    ststime.append(costtime5)
    print(ststime)

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

