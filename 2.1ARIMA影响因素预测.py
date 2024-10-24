import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf   # acf pacf
from statsmodels.tsa.stattools import adfuller   # adf检验
from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.api import qqplot
from sklearn import metrics
from sklearn.metrics import r2_score  # 拟合优度
plt.rcParams['font.sans-serif'] = 'SimHei' # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

# 准备数据
df = pd.read_csv("trend.csv")
# 转为df格式、将日期列设为索引列、选取特定列
df = pd.DataFrame(df).set_index('date')
train_df = df[:-20]
test_df = df[-20:]

def draw_ts(train_sale):
    # 绘制原始序列
    plt.figure(figsize=(10, 5))
    train_sale.plot()
    ##自相关性检验
    plot_acf(train_sale, lags=80)
    plt.show()
def adfullerJY(train_sale):
    print('原始序列的ADF检验结果为：', adfuller(train_sale))
def diffYS(train_sale):
    """一阶差分序列检验"""
    d1_sale = train_sale.diff(periods=1).dropna()  # dropna删除NaN    periods属性为间隔几个数据做差分，默认为1
    # 时序图
    #plt.figure(figsize=(10, 5))
    #d1_sale.plot()
    # 自相关图
    #plot_acf(d1_sale, lags=20)
    # plt.show()
    # 平稳性检验
    print('原始序列一阶差分的ADF检验结果为：', adfuller(d1_sale))
    # 解读：P值小于显著性水平α（0.05），拒绝原假设（非平稳序列），说明一阶差分序列是平稳序列。
    return d1_sale
def AcfPcaf(d1_sale):
    """定阶"""
    ##1 人工识图
    # d1_sale = train_sale.diff(periods=1, axis=0).dropna()
    # 自相关图
    plot_acf(d1_sale, lags=34)
    # 偏自相关图
    plot_pacf(d1_sale, lags=20)
    # 偏自相关图
    plot_pacf(d1_sale, lags=34)
    plt.show()
def Bic(d1_sale):
    ##2参数调优：BIC
    pmax = int(len(d1_sale) / 50)  # 一般阶数不超过length/10
    qmax = int(len(d1_sale) / 50)  # 一般阶数不超过length/10
    bic_matrix = []
    print(d1_sale)
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(sm.tsa.ARIMA(d1_sale.astype(float), order=(p, 1, q)).fit().bic)
                print("times", "p", p, "q", q)
            except Exception as e:
                print(e)
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    print(bic_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(bic_matrix,
                     mask=bic_matrix.isnull(),
                     ax=ax,
                     annot=True,
                     fmt='.2f')
    ax.set_title('Bic')
    bic_matrix.stack()
    p, q = bic_matrix.stack().idxmin()  # 最小值的索引
    print('用BIC方法得到最优的p值是%d,q值是%d' % (p, q))


def Aic(d1_sale, c):
    import warnings
    # 忽视一些烦人的报错
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    ##3 参数调优：AIC
    pmax = int(len(d1_sale) / 100)
    qmax = int(len(d1_sale) / 100)

    aic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(sm.tsa.ARIMA(d1_sale.astype(float), order=(p, 1, q)).fit().aic)
                # print("times", "p", p, "q", q)
            except:
                tmp.append(None)

        aic_matrix.append(tmp)
    aic_matrix = pd.DataFrame(aic_matrix)
    p, q = aic_matrix.stack().idxmin()  # 最小值的索引
    print(f'{c}用AIC方法得到最优的p值是%d,q值是%d' % (p, q))
    return p, q
def modelJY(sale):
    """建模及预测"""
    model = sm.tsa.ARIMA(sale.astype(float), order=(1, 1, 1)).fit()
    ##残差检验
    resid = model.resid
    ##1
    # 自相关图
    plot_acf(resid, lags=35)
    # 解读：有短期相关性，但趋向于零。
    # 偏自相关图
    plot_pacf(resid, lags=20)
    # 偏自相关图
    plot_pacf(resid, lags=35)
    plt.show()
    # 2 qq图
    qqplot(resid, line='q', fit=True).show()
    plt.show()
    # 3 DW检验
    print('D-W检验的结果为：', sm.stats.durbin_watson(resid.values))
    # 解读：不存在一阶自相关

    # 4 LB检验
    print('残差序列的白噪声检验结果为：', acorr_ljungbox(resid, lags=1))  # 返回统计量、P值
    # 解读：残差是白噪声 p>0.05
    # confint,qstat,pvalues = sm.tsa.acf(resid.values, qstat=True)

def ForeCast(model, c):
    #预测
    forecastdata = model.forecast(20)  # 预测、标准差、置信区间
    calculate_errors(test_df[c], forecastdata)
    print(f"forecastdata:{forecastdata}")
    print(f"test_df[c]:{test_df[c]}")
    plt.figure(figsize=(15, 4))
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'),test_df[c],label='real')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W'), forecastdata, label='forecast')
    plt.show()
    return forecastdata

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


if __name__ == '__main__':
    result = pd.DataFrame()
    for c in train_df.columns:
        if c == 'date':
            continue
        d1_sale = diffYS(train_df[c])
        p, q = Aic(d1_sale, c)
        arima = sm.tsa.ARIMA(train_df[c].astype(float), order=(p, 1, q))
        model = arima.fit()
        result[c] = ForeCast(model, c)
    result.to_csv("influence_pred.csv")