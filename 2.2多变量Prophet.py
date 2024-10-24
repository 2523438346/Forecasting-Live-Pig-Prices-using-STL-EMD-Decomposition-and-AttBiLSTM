import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import r2_score  # 拟合优度
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
# 预测多久的时间步(改完需要给一下画图的日期范围)
pred_timestep = 20
df = pd.read_csv("trend.csv")
df = df[['date', '白萝卜','白条鸡','白条猪','大黄花鱼','豆粕','牛肉','农产品批发价格200指数','生猪','羊肉','玉米','进口量(吨)','出口量(吨)','进口金额(万美元)','出口金额(万美元)']]
df = df.rename(columns={'date':'ds', '生猪':'y'})
train_data = df[:-pred_timestep]
test_data = df[-pred_timestep:]
# 归一化
mean = train_data['y'].mean()
std = train_data['y'].std()
train_data['y'] = ((train_data['y'] - mean) / std)
def calculate_errors(y_true, y_pred):
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    # 确保预测值和真实值的长度相同
    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值的长度必须相同")
    # 计算RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # 计算MAE
    mae = np.mean(np.abs(y_true - y_pred))
    # 计算MAPE
    # 注意：如果y_true中有0，这将导致除以0的错误。可以添加一个小常数避免这种情况，但通常应检查y_true
    y_true_nonzero = y_true[y_true != 0]
    y_pred_nonzero = y_pred[y_true != 0]
    if len(y_true_nonzero) == 0:
        raise ValueError("真实值中全为0，无法计算MAPE")
    mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
    print(f"RMSE:{rmse}")
    print(f"MAE:{mae}")
    print(f"MAPE:{mape}")
    return rmse, mae, mape

# 训练模型
model = Prophet(changepoint_prior_scale = 0.15)
# 添加影响因素
model.add_regressor('白萝卜')
model.add_regressor('白条鸡')
model.add_regressor('白条猪')
model.add_regressor('大黄花鱼')
model.add_regressor('豆粕')
model.add_regressor('农产品批发价格200指数')
model.add_regressor('羊肉')
model.add_regressor('玉米')
model.add_regressor('牛肉')
model.add_regressor('进口量(吨)')
model.add_regressor('出口量(吨)')
model.add_regressor('进口金额(万美元)')
model.add_regressor('出口金额(万美元)')
model.fit(train_data)
# 设置预测的未来的时间
forecast = model.make_future_dataframe(periods=pred_timestep, freq='W')
forecast['牛肉'] = df['牛肉']
forecast['白条鸡'] = df['白条鸡']
forecast['白萝卜'] = df['白萝卜']
forecast['白条猪'] = df['白条猪']
forecast['大黄花鱼'] = df['大黄花鱼']
forecast['豆粕'] = df['豆粕']
forecast['农产品批发价格200指数'] = df['农产品批发价格200指数']
forecast['羊肉'] = df['羊肉']
forecast['玉米'] = df['玉米']
forecast['进口量(吨)'] = df['进口量(吨)']
forecast['出口量(吨)'] = df['出口量(吨)']
forecast['进口金额(万美元)'] = df['进口金额(万美元)']
forecast['出口金额(万美元)'] = df['出口金额(万美元)']
print(forecast)
# 导入数据进行预测
forecast = model.predict(forecast)
forecast.to_csv("result/test.csv")
# 反归一化
train_data['y'] = (train_data['y'] * (std)) + mean
forecast['yhat'] = (forecast['yhat'] * (std)) + mean
forecast['yhat_upper'] = (forecast['yhat_upper'] * (std)) + mean
forecast['yhat_lower'] = (forecast['yhat_lower'] * (std)) + mean
calculate_errors(test_data['y'], forecast['yhat'][-pred_timestep:])
plt.figure(figsize=(15, 4))
plt.subplot(2, 1, 1)
plt.plot(pd.date_range(start='2016-01-03', end='2023-12-31', freq='W'), train_data['y'], color='c', label='real')
plt.plot(pd.date_range(start='2016-01-03', end='2024-05-19', freq='W'), forecast['yhat'], color='y', label='pred')
plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), test_data['y'], color='c')
plt.fill_between(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), forecast['yhat_upper'][-pred_timestep:], forecast['yhat_lower'][-pred_timestep:])
plt.title('train forecast results')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('Yuan/kg')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), test_data['y'], color='c', label='test')
plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), forecast['yhat'][-pred_timestep:], color='y', label='pred')
plt.fill_between(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), forecast['yhat_upper'][-pred_timestep:], forecast['yhat_lower'][-pred_timestep:], color=(255/255,250/255,205/255))
plt.title('test forecast results')
plt.grid(True)
plt.xlabel('time')
plt.ylabel('Yuan/kg')
plt.legend()
plt.subplots_adjust(hspace=0.3)
plt.show()
forecast['yhat'][-pred_timestep:].to_csv("pred_trend.csv")
print(forecast['yhat'])

