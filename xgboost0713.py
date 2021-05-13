import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

data = load_boston()

X = data.data
y = data.target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)
reg.predict(Xtest)
reg.score(Xtest, Ytest)
MSE(Ytest, reg.predict(Xtest))
reg.feature_importances_

xgb.plot_importance(reg)
plt.rcParams['figure.figsize'] = [3,3]
plt.show()


# 另外一种用法
reg = XGBR(n_estimators=100)
CVS(reg, Xtrain, Ytrain, cv=5).mean()
# 严谨的交叉验证：只使用训练集进行交叉验证，用测试机进行最终测试。
# 不严谨的交叉验证：全数据集？
CVS(reg, Xtrain, Ytrain, scoring='neg_mean_squared_error', cv=5).mean()

# 查看sklearn 中所有的模型评估指标
import sklearn

sorted(sklearn.metrics.SCORERS.keys())

rfr = RFR(n_estimators=100)
CVS(rfr, Xtrain, Ytrain, cv=5).mean()
CVS(rfr, Xtrain, Ytrain, scoring='neg_mean_squared_error', cv=5).mean()
lr = LinearR()
CVS(lr, Xtrain, Ytrain, cv=5).mean()
CVS(lr, Xtrain, Ytrain, scoring='neg_mean_squared_error', cv=5).mean()

# 如果开启参数slient:在数据量大，预料到算法运行会非常缓慢是，可以使用这个参数监控模型的训练进度
reg = XGBR(n_estimators=10, slient=False)

CVS(reg, Xtrain, Ytrain, scoring='neg_mean_squared_error', cv=5).mean()


###############################
def plot_learning_curve(estimator, title, X, y, ax=None, ylim=None, cv=None, n_jobs=None):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , shuffle=True
                                                            , cv=cv
                                                            , random_state=420
                                                            , n_jobs=n_jobs)

    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label="Test Score")
    ax.legend(loc="best")
    return ax


cv = KFold(n_splits=5, shuffle=True, random_state=42)  # 交叉验证模式
plot_learning_curve(XGBR(n_estimators=100, random_state=420), "XGBR", Xtrain, Ytrain, ax=None, cv=cv)
plt.show()

# 看起来，模型是过拟合了，证明，我们由调参的空间
# 使用参数学习曲线观察n_estimators对模型的影响
axisx = range(10, 1010, 50)
rs = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    rs.append(CVS(reg, Xtrain, Ytrain, cv=cv).mean())

print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='r', label="XBG")
plt.legend()
plt.show()

# 进化的学习曲线，方差与泛化误差, bias^2 + var + e^2
# 方差-偏差困境
axisx = range(50, 1050, 50)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())

# 打印R2最高所对应的取值，并打印该参数下的方差
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], min(var), rs[var.index(min(var))])
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))

plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='r', label="XRGB")
plt.legend()
plt.show()

# 细化学习曲线，找出最佳的n_estimators_
axisx = range(100, 300, 10)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())

# 打印R2最高所对应的取值，并打印该参数下的方差
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], min(var), rs[var.index(min(var))])
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))

rs = np.array(rs)
var = np.array(var)

plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='r', label="XRGB")
plt.plot(axisx, rs + var, c='g', linestyle='-.')
plt.plot(axisx, rs - var, c='b', linestyle='-.')
plt.legend()
plt.show()

time0 = time()
print(XGBR(n_estimators=100, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

time0 = time()
print(XGBR(n_estimators=660, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

time0 = time()
print(XGBR(n_estimators=150, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

time0 = time()
print(XGBR(n_estimators=180, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

# 数据集很大时，上述思路就不时那么合适了，时间比较长

# 有放回的随机抽样，重要参数subsample,能够有效的减少过拟合
axisx = np.linspace(0, 1, 20)
rs = []

for i in axisx:
    reg = XGBR(n_estimators=150, subsample=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())

# 打印R2最高所对应的取值，并打印该参数下的方差
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])

plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='r', label="XRGB")
plt.legend()
plt.show()

axisx = np.linspace(0.05, 1, 20)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=150, subsample=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())

# 打印R2最高所对应的取值，并打印该参数下的方差
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], min(var), rs[var.index(min(var))])
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var)

plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c='r', label="XRGB")
plt.plot(axisx, rs + var, c='g', linestyle='-.')
plt.plot(axisx, rs - var, c='b', linestyle='-.')
plt.legend()
plt.show()

reg = XGBR(n_estimators=150, subsample=0.65, random_state=420)
cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
cvresult.mean()


# 由于数据量太小，调整subsample反而降低了学习能力，因此，在这种情况下，就不用调整该参数了
# 迭代决策树，重要参数eta 也就时sklearn中的learning_rate, 影响学习的总时间


def regassess(reg, Xtrain, Ytrain, cv, scoring=["r2"], show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i]
                                     , CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean()))
        score.append(CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean())

    return score


regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"])

reg = XGBR(n_estimators=150, random_state=420)
regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"])

from time import time
import datetime

for i in [0, 0.2, 0.5, 1]:
    time0 = time()
    reg = XGBR(n_estimators=150, random_state=420, learning_rate=i)
    print("learning_rate = {}".format(i))
    regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"])
    print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
    print("\t")