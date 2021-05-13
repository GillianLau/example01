from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense,Dropout
from keras.utils import multi_gpu_model
from keras import regularizers #正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
# (x_train,y_train),(x_valid,y_valid) = boston_housing.load_data()
# #转成DataFrame格式方便数据处理
# x_train_pd = pd.DataFrame(x_train)
# y_train_pd = pd.DataFrame(y_train)
# x_valid_pd = pd.DataFrame(x_valid)
# y_valid_pd = pd.DataFrame(y_valid)
# print(x_train_pd.head(5))
# print('________________')
# print(y_train_pd.head(5))
df = pd.read_excel('10inchdataset.xlsx', sheet_name=7)
x, y = np.split(df, (29,), axis=1)

print("样本数据量:%d, 特征个数：%d" % x.shape)
print("target样本数据量:%d" % y.shape[0])

X_DF = pd.DataFrame(x)
X_DF.info()
X_DF.describe().T
X_DF.head()
# X_DF.hist(bins = 50,figsize = (30,20))
x_train_pd, x_valid_pd, y_train_pd, y_valid_pd = TTS(x, y, train_size=0.75, random_state=75)




#数据归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

#验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

#训练模型
model = Sequential()
model.add(Dense(units = 10,
                activation = 'relu',
                input_shape = (x_train_pd.shape[1],)

               )
          )
model.add(Dropout(0.2))
model.add(Dense(units = 15,
                activation = 'relu'
                )
          )
model.add(Dense(units = 1,
                activation = 'linear'

               )
)
print(model.summary())
model.compile(loss = 'mse',
              optimizer = 'adam',
              )
history = model.fit(x_train,y_train,
                    epochs = 200,
                    batch_size = 200,
                    verbose = 2,
                    validation_data = (x_valid,y_valid)
                    )
#训练过程可视化
import matplotlib.pyplot as plt
#绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc = 'upper left')
plt.show()
#保存模型&模型可视化&加载模型
from keras.utils import plot_model
from keras.models import load_model
#保存模型
model.save('model_Mlp.h5') #生成模型文件'my_model.h5'
# 模型可视化 需要安装pydot pip install pydot
plot_model(model,to_file = 'model_MLP.png',show_shapes = True)
#加载模型
model= load_model('model_MLP.h5')

#模型的预测功能
# 预测
y_new = model.predict(x_valid)
# 反归一化还原原始量纲
min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)
error2 = abs(y_new-y_valid_pd)
Abs = abs(error2)
def LowerCount(a, b):
    num = 0
    for i in a:
        if i < b:  # 可依需要修改条件
            num += 1
    percent = num / len(a)
    return num, percent


ans2 = LowerCount(Abs.values, 1.27)
print(ans2)
plt.figure()
ln_x_test = range(len(y_new))
# ax1 = plt.subplot(211)
plt.plot(ln_x_test, y_valid_pd, 'r-', lw=2, label=u'actual value')
plt.plot(ln_x_test, y_new, 'g-', lw=2, label=u'predict value')
plt.plot(ln_x_test, y_new-y_valid_pd, 'b-', lw=2, label=u'error')
plt.xlabel(u'num')
plt.ylabel(u'depth')
plt.legend(loc='lower right')
plt.grid(True)
plt.title(u'xgb prediction of depth')
plt.show()