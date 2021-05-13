from sklearn import datasets
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston = datasets.load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['price'] = boston.target

y = data.pop('price')
#data_matrix = xgb.DMatrix(data,y)
train_X,test_X,train_y,test_y = train_test_split(data,y,test_size= 0.25)
dtrain = xgb.DMatrix(train_X,train_y)
dtest = xgb.DMatrix(test_X,test_y)
d = xgb.DMatrix(data,y)
xg_reg = xgb.XGBRegressor(
    objective = 'reg:linear',
    colsample_bytree = 0.3,
    learning_rate = 0.1,
    max_depth = 5,
    n_estimators = 10,
    alpha = 10
)
xg_reg.fit(train_X,train_y)
pred = xg_reg.predict(test_X)
mean_squared_error(pred,test_y)

params = {"objective":"reg:linear",'colsample_bytree':0.3,'learning_rate':0.1,
          'max_depth':5,'alpha':10}
cv_results = xgb.cv(dtrain = d, params = params,nfold = 3,
                    num_boost_round = 50, early_stopping_rounds =10, metrics = 'rmse', as_pandas = True, seed = 123)
cv_results.head()

#打印数的分裂情况#
xg_reg = xgb.train(params = params,dtrain = d,num_boost_round = 10)
import matplotlib.pyplot as plt
from xgboost import plot_tree
from graphviz import Digraph
import pydot
xgb.plot_tree(xg_reg,num_trees = 0)
plt.rcParams['figure.figsize'] = [80,60]
plt.show()

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [3,3]
plt.show()
