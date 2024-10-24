import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.metrics import r2_score  # 拟合优度
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
trend = pd.read_csv("pred_trend.csv", index_col='date')
season = pd.read_csv("seasonal_pred.csv", index_col='date')
resid = pd.read_csv("pred_resid_noemd.csv", index_col='date')
true = pd.read_excel("true.xlsx", index_col="date")
def show(pred):
    df = pd.read_excel("basic_data.xlsx", index_col="date")
    pig_price = df.loc[:, '生猪']
    plt.figure(figsize=(15, 4))
    plt.subplot(2, 1, 1)
    plt.plot(pd.date_range(start='2016-01-03', end='2024-05-19', freq='W'), pig_price, color='c', label='real')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), pred, color='y', label='pred')
    plt.title(f'train forecast results')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Yuan/kg')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), pig_price[-20:], color='c', label='real')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), pred,
             color='y', label='pred')
    plt.title(f'test forecast results')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Yuan/kg')
    plt.legend()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
def calculate_errors(y_true, y_pred):
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.squeeze(y_true, axis=1)
    print(y_true.shape)
    print(y_pred.shape)
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
pred = trend['yhat']+season['y']+resid['0']
pred.to_csv("finally_pred.csv")
calculate_errors(true, pred)
show(pred)
