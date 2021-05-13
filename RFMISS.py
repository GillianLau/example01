#导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
# 以波士顿数据集为例，导入完整数据集并探索
dataset = load_boston()
dataset.data.shape

x_full,y_full = dataset.data,dataset.target #赋值
n_samples = x_full.shape[0]
n_features= x_full.shape[1]

#为完整数据集放入缺失
#首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%，那总共就要有3289个数据缺失
rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
#np.floor向下取整，返回.0格式的浮点数
n_missing_samples

missing_features = rng.randint(0,n_features,n_missing_samples)
missing_samples = rng.randint(0,n_samples,n_missing_samples)

X_missing = x_full.copy()
y_missing = y_full.copy()

X_missing[missing_samples,missing_features] = np.nan
X_missing = pd.DataFrame(X_missing)
imp_mean = SimpleImputer(missing_values = np.nan,strategy = 'mean')
X_missing_mean = imp_mean.fit_transform(X_missing)

#使用0进行填补

imp_0 = SimpleImputer(missing_values = np.nan,strategy = "constant",fill_value = 0)
X_missing_0 = imp_0.fit_transform(X_missing)

X_missing_reg = X_missing.copy()
sortindex = np.argsort(X_missing_reg.isnull().sum(axis = 0)).values
sortindex

for i in sortindex:
    df = X_missing_reg
    fillc = df.iloc[:,i]
    df = pd.concat([df.iloc[:,df.columns!=i],pd.DataFrame(y_full)],axis = 1)

    df_0 = SimpleImputer(
        missing_values=np.nan,
        strategy = 'constant',
        fill_value = 0
                        ).fit_transform(df)
    Ytrain = fillc[fillc.notnull()]
    Ytest = fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index,:]
    Xtest = df_0[Ytest.index,:]

    rfc = RandomForestRegressor(n_estimators = 100)
    rfc = rfc.fit(Xtrain,Ytrain)
    Ypredict = rfc.predict(Xtest)

    X_missing_reg.loc[X_missing_reg.iloc[:,i].isnull(),i] = Ypredict

X = [x_full,X_missing_mean,X_missing_0,X_missing_reg]
mse = []
std = []
for x in X:
    estimator = RandomForestRegressor(random_state = 0,n_estimators = 100)
    scores = cross_val_score(estimator,x,y_full,scoring = 'neg_mean_squared_error',cv = 5).mean()
    mse.append(scores * -1)
    x_labels = ['Full data',
                'Zero Imputation',
                'Mean Imputation',
                'Regressor Imputation'
                ]
    colors = ['r','g','b','orange']
    plt.figure(figsize = (12,6))
    ax = plt.subplot(111)
    for i in np.arange(len(mse)):
        ax.barh(i,mse[i],color = colors[i],alpha = 0.6,align = 'center')

        ax.set_title('Imputation Techniques with Boston Data')
        ax.set_xlim(left=np.min(mse) * 0.9, right=np.max(mse) * 1.1)
        ax.set_yticks(np.arange(len(mse)))
        ax.set_xlabel('MSE')
        ax.set_yticklabels(x_labels)
        plt.show()
