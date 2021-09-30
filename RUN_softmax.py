import matplotlib.pyplot as plt
from math import sqrt
# from matplotlib import pyplot
import pandas as pd
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,explained_variance_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
'''
keras实现神经网络回归模型
'''

n_features=2
# 读取数据
path = r'C:\Users\33\Desktop\name.csv'
train_df = pd.read_csv(path)
# 删掉不用字符串字段
#dataset = train_df.drop('jh', axis=1)
# df转array
values = train_df.to_numpy()
# 原始数据标准化，为了加速收敛
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(values[:, 22:24])
y = values[:, 24]
# X = scaled[:, 13]

# 随机拆分训练集与测试集
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

# 全连接神经网络
model = Sequential()
input=X.shape
print(input)
# 隐藏层128
model.add(Dense(128, input_dim=n_features))
model.add(Activation('softmax'))
# Dropout层用于防止过拟合
# model.add(Dropout(0.2))
# 隐藏层128
model.add(Dense(128))
model.add(Activation('softmax'))
# model.add(Dropout(0.2))
# 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
model.add(Dense(64,activation='softmax'))
model.add(Dense(1))
# 使用高效的 ADAM 优化算法以及优化的最小均方误差损失函数
model.compile(loss='mean_squared_error', optimizer=Adam(),
              )

# model.compile(loss='mean_absolute_error', optimizer=Adam(),
#               )

# 训练
model.summary()


history = model.fit(train_X, train_y, epochs=1500, batch_size=48, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
model.save('./1MIN_goes12flux_lm12_yubaogoes11_softmax.h5')

def calPerformance(y_true,y_pred):
    #  '''
    # 模型效果指标评估
    # y_true：真实的数据值
    # y_pred：回归模型预测的数据值
    # explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
    # 的方差变化，值越小则说明效果越差。
    # mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真X
    # 实数据集的接近程度的程度
    # ，其其值越小说明拟合效果越好。
    # mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
    # 平方和的均值，其值越小说明拟合效果越好。
    # r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    # 变量的方差变化，值越小则说明效果越差。
    # '''
    model_metrics_name=[explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    tmp_list=[]
    for one in model_metrics_name:
        tmp_score=one(y_true,y_pred)
        tmp_list.append(tmp_score)
    x=[]
    x.append(tmp_list)
    f1 = pd.DataFrame(x,columns=['explained_variance_score', 'mean_absolute_error', 'mean_squared_error', 'r2_score'])
    print(f1)
    return tmp_list

# 模型评分
calPerformance(test_y,model.predict(test_X))

# loss曲线
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# # 预测
# yhat = model.predict(9.69099,383.2,8.08,2.37,-4)
# # 预测y逆标准化
# inv_yhat0 = concatenate((test_X, yhat), axis=1)
# inv_yhat1 = scaler.inverse_transform(inv_yhat0)
# inv_yhat = inv_yhat1[:, -1]
# # 原始y逆标准化
# test_y = test_y.reshape((len(test_y), 1))
# inv_y0 = concatenate((test_X, test_y), axis=1)
# inv_y1 = scaler.inverse_transform(inv_y0)
# inv_y = inv_y1[:, -1]
# # 计算 RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
# plt.plot(inv_y)
# plt.plot(inv_yhat)
# plt.show()

print('test end')
