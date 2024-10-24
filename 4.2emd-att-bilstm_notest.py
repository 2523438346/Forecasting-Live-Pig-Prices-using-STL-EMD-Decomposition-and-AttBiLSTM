import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.api.layers import LSTM, Dense, Bidirectional, Lambda, Input, concatenate, Dropout
from keras.api.models import Sequential, load_model, Model
from keras.api.callbacks import ModelCheckpoint
from sklearn.metrics import r2_score  # 拟合优度
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
from sklearn import metrics
import tensorflow as tf
from keras.api.saving import register_keras_serializable
from keras.api.regularizers import l2

imf = pd.read_csv('IMFs_and_Residue.csv').set_index('date')
# emd_index预测第几列
emd_index = '6'
train_df = imf[emd_index][:-20] # 2016.01.03-2023-12-31  形状：(418,)
def normalize_dataframe(train_df, test_df):
    mean = train_df.mean()
    std = train_df.std()
    print(f"mean:{mean}")
    print(f"std:{std}")
    train_df = (train_df -mean)/std
    test_df = (test_df -mean)/std
    return train_df, test_df, mean, std


def prepare_data(data, win_size):
    X = []
    y = []
    for i in range(len(data) - win_size):
        temp_x = data[i:i + win_size]
        temp_y = data[i + win_size]
        X.append(temp_x)
        y.append(temp_y)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

# 定义注意力层
@register_keras_serializable(package='Custom')
class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, units, name=None, trainable=True, dtype='float32'):
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    def call(self, query, values):
        # query的形状是(batch_size, query_seq_len, query_dim)
        # values的形状是(batch_size, values_seq_len, values_dim)
        # 通常query_dim和values_dim应该是相同的，或者至少W1和W2的输出维度应该是相同的
        # 使用W1和W2转换query和values
        W1_output = self.W1(values)  # (batch_size, values_seq_len, units)
        W2_output = self.W2(query)  # (batch_size, query_seq_len, units)
        # 计算所有query时间步与所有values时间步之间的注意力分数
        # 这将产生一个(batch_size, query_seq_len, values_seq_len)形状的张量
        scores = tf.matmul(W2_output, W1_output, transpose_b=True)  # 使用矩阵乘法来计算分数
        # 对分数进行softmax标准化处理
        attention_weights = tf.nn.softmax(scores, axis=-1)  # 在values的时间维上进行softmax

        # 使用注意力权重*值来得到上下文向量
        # 这将产生一个(batch_size, query_seq_len, values_dim)形状的张量
        context_vectors = tf.matmul(attention_weights, values)
        # 如果你只想要一个上下文向量而不是每个query时间步都有一个，你可以进一步减少这个维度
        # 例如，你可以对query时间维进行求和或平均
        context_vector = tf.reduce_mean(context_vectors, axis=1)  # (batch_size, values_dim)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units
        })
        return config

def model_building(train_x, train_y):
    # 定义输入层
    input_layer = Input(shape=(train_x.shape[1], 1))
    # 添加双向LSTM层
    bidirectional_lstm = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(input_layer)
    # 添加注意力层
    attention = AttentionLayer(128)
    context_vector, attention_weight = attention(bidirectional_lstm, bidirectional_lstm)
    # 使用Lambda层来获取最后一个时间步的输出
    lambda_layer = Lambda(lambda x: x[:, -1, :])(bidirectional_lstm)
    print(f"vector_shape:{context_vector.shape}")
    print(f"lambda_layer:{lambda_layer.shape}")
    combined_vector = concatenate([context_vector, lambda_layer])
    # 添加全连接层，并在每个全连接层后添加Dropout层
    l2_reg = l2(0.01)  # L2正则化系数，可以根据需要调整
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2_reg)(combined_vector)
    dropout1 = Dropout(0.5)(dense1)  # 添加Dropout层，丢弃率为0.5
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2_reg)(dropout1)
    dropout2 = Dropout(0.5)(dense2)  # 再次添加Dropout层，丢弃率也为0.5
    output_layer = Dense(1)(dropout2)
    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer)
    # 定义ModelCheckpoint回调
    checkpoint = ModelCheckpoint(f'emd_model/best_model_imf{emd_index}.keras', monitor='loss', verbose=1, save_best_only=True, mode='min')
    # 编译模型
    model.compile(loss='mse', optimizer='adam')
    # 模型拟合
    history = model.fit(train_x, train_y, epochs=200, batch_size=32, callbacks=[checkpoint])
    model.summary()
    plt.figure()
    plt.plot(history.history['loss'], c='b', label='loss')
    plt.legend()
    plt.show()
    # return model

def future_evaluation(real, series):
    # 计算均方误差（MSE）
    mse = metrics.mean_squared_error(real, series)
    # 计算均方根误差（RMSE）
    rmse = np.sqrt(mse)
    # 计算平均绝对误差（MAE）
    mae = metrics.mean_absolute_error(real, series)
    r2 = r2_score(real, series)
    print("预测未来的评价指标如下：")
    print("均方误差 (MSE):", mse)
    print("均方根误差 (RMSE):", rmse)
    print("平均绝对误差 (MAE):", mae)
    print("拟合优度:", r2)

def predict_future(model, initial_sequence, steps):
    predicted_values = []  # 存储预测结果
    current_sequence = initial_sequence.copy()  # 初始序列
    for i in range(steps):
        # 使用模型进行单步预测
        print(initial_sequence.shape)
        predicted_value = model.predict(current_sequence.reshape(1, initial_sequence.shape[0]))
        # 将预测结果添加到列表中
        predicted_values.append(predicted_value[0][0])
        # 更新当前序列，删除第一个时间步并将预测值添加到最后一个时间步
        current_sequence[:-1] = current_sequence[1:]
        current_sequence[-1] = predicted_value
    return predicted_values

def finish_show(model, train_x, series):
    plt.figure(figsize=(15, 4))
    plt.subplot(2, 1, 1)
    train_pred = model.predict(train_x)
    train_csv = pd.DataFrame(train_pred)
    series_csv = pd.DataFrame(series)
    train_csv.to_csv(f"emd_pred/pred_resid_imf{emd_index}_trainpred.csv", index=False)
    series_csv.to_csv(f"emd_pred/pred_resid_imf{emd_index}_futurepred.csv", index=False)
    plt.plot(pd.date_range(start='2016-01-03', end='2024-05-19', freq='W')
             , imf[emd_index].iloc[:,], color='m', label='real')
    plt.plot(pd.date_range(start='2016-03-27', end='2023-12-31', freq='W')
             , train_pred, color='c', label='训练集pred')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), series, color='b', label='向后预测20周')
    plt.title(f'imf{emd_index}预测')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('元/公斤')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W')
             , imf[emd_index].iloc[-20:, ], color='m', label='real')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), series, color='b', label='向后预测20周')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('元/公斤')
    plt.legend()
    plt.show()
def exponential_smoothing(series, alpha):
    """
    简单的指数平滑法实现

    参数:
    series (pd.Series): 时间序列数据
    alpha (float): 平滑系数，取值范围 [0, 1]

    返回:
    pd.Series: 平滑后的时间序列
    """
    result = [series[0]]  # 第一个值作为初始预测值
    for n in range(1, len(series)):
        result.append((alpha * series[n] + (1 - alpha) * result[n - 1]))
    return pd.Series(result, index=series.index)
# 2.数据标准化
# train_df, test_df, mean, std = normalize_dataframe(train_df, test_df)
# 3.定义滑动窗口
win_size = 12 # 时间窗口
train_x, train_y = prepare_data(train_df.values, win_size)
print("训练集形状:", train_x.shape, train_y.shape)   # 训练集形状: (406, 12) (406,)
# 4.模型建立
model_building(train_x, train_y)
model = load_model(f'emd_model/best_model_imf{emd_index}.keras', safe_mode=False, custom_objects={'AttentionLayer': AttentionLayer})
# 6.预测未来
series = predict_future(model, train_x[-1], 20)   # 模型、初始序列、步长(预测天数)
# 7.预测未来的评价指标
real = imf[emd_index].iloc[-20:,]
s_series = pd.Series(series, index=pd.date_range('2024-01-07', periods=20, freq='W'))
# 8.平滑预测曲线
s_series = exponential_smoothing(s_series, 1)
# 9.评价
future_evaluation(real, s_series)
# 10.画最终结果
finish_show(model, train_x, s_series)
