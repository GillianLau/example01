import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import gradient_boosting as GB
import missingno as msno
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
import time
import datetime
from sklearn.metrics import mean_squared_error
import seaborn as sns
## 数据导入
df = pd.read_excel('10inchdataset.xlsx',sheet_name = 7)
x,y = np.split(df,(29,),axis = 1)

print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)
X_DF.info()
X_DF.describe().T
X_DF.head()
# X_DF.hist(bins = 50,figsize = (30,20))
# plt.tight_layout()
# plt.show()
#
# msno.bar(X_DF.sample(1000))
#
#
# msno.heatmap(X_DF)
#
# msno.dendrogram(X_DF)
# plt.tight_layout()
# plt.show()

sns.heatmap(X_DF.corr())
plt.tight_layout()
plt.show()