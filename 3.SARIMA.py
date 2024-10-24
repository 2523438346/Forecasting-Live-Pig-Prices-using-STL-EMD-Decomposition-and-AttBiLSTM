import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller  # adf检验
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
from sklearn.metrics import r2_score as rs
import warnings

warnings.filterwarnings("ignore")#忽略输出警告
plt.rcParams["font.sans-serif"]=["SimHei"]#用来正常显示中文标签
plt.rcParams["axes.unicode_minus"]=False#用来显示负号

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
    step = 19
    d1_sale = train_sale.diff(step).dropna()  # dropna删除NaN    periods属性为间隔几个数据做差分，默认为1
    # 平稳性检验
    print(f'原始序列{step}步差分的ADF检验结果为：', adfuller(d1_sale))
    # 解读：P值小于显著性水平α（0.05），拒绝原假设（非平稳序列），说明一阶差分序列是平稳序列。
    return d1_sale
# 这个方法可以判断最佳差分阶数 df为数据 max_diff是最大差分阶数 significance_level自定义与adf检验的显著性水平
def calculate_diff(df, max_diff, significance_level=0.05):
    # 初始化最佳差分步数和最小p值
    best_diff = None
    min_pvalue = 1.0
    min_variance = float('inf')  # 初始化最小方差
    min_adf_stat = float('inf')  # 初始化最小ADF统计量

    # 循环，差分阶数从1到max_diff
    for i in range(1, max_diff+1):
        # 对数据进行差分，并去除NA值
        df_diff = df.diff(i).dropna()
        # 对差分后的数据进行ADF单位根检验
        result = adfuller(df_diff)
        # 打印出差分阶数，ADF统计量，p值，标准差和方差
        print(f'{i}步差分')
        print('ADF统计量: %f' % result[0])
        print('p值: %.10e' % result[1])
        print('标准差: %f' % df_diff.std())
        print('方差: %f' % df_diff.var())

        # 判断p值是否小于显著性水平，如果小于则认为差分后的数据可能是平稳的
        if result[1] < significance_level:
            print('=> 根据这个差分阶数，序列可能是平稳的')
            # 判断当前的p值是否小于最小p值，如果小于则更新最小p值和最佳差分阶数
            if result[1] < min_pvalue:
                min_pvalue = result[1]
                best_diff = i
                min_variance = df_diff.var()  # 更新最小方差
                min_adf_stat = result[0]  # 更新最小ADF统计量
        else:
            print('=> 根据这个差分阶数，序列可能是非平稳的')
        print('--------------------------------')

    # 如果找到了使数据平稳的差分阶数，打印出最佳差分阶数和其对应的p值
    if best_diff is not None:
        print(f'最佳差分步数是: {best_diff}，p值为: {min_pvalue}，方差为: {min_variance}，ADF统计量为: {min_adf_stat}')
# 搜索法定阶
def SARIMA_search(data):
    p = q = range(0, 3)
    s = [52]
    d = [1]
    D = [1]
    PDQs = list(itertools.product(p, D, q, s))  # itertools.product()得到的是可迭代对象的笛卡儿积
    pdq = list(itertools.product(p, d, q))  # list是python中是序列数据结构，序列中的每个元素都分配一个数字定位位置
    params = []
    seasonal_params = []
    results = []
    grid = pd.DataFrame()
    for param in pdq:
        for seasonal_param in PDQs:
            # 建立模型
            mod = sm.tsa.SARIMAX(data, order=param, seasonal_order=seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
            # 实现数据在模型中训练
            result = mod.fit()
            print("ARIMA{}x{}-AIC:{}".format(param, seasonal_param, result.aic))
            # format表示python格式化输出，使用{}代替%
            params.append(param)
            seasonal_params.append(seasonal_param)
            results.append(result.aic)
    grid["pdq"] = params
    grid["PDQs"] = seasonal_params
    grid["aic"] = results
    print(grid[grid["aic"] == grid["aic"].min()])
def modelJY(data_train, p, d, q, P, D, Q, Z):
    model = sm.tsa.SARIMAX(data_train, order=(p, d, q), seasonal_order=(P, D, Q, Z)).fit()
    ##残差检验
    resid = model.resid
    ##1
    # 自相关图
    # plot_acf(resid, lags=35)
    # # 解读：有短期相关性，但趋向于零。
    # # 偏自相关图
    # plot_pacf(resid, lags=20)
    # # 偏自相关图
    # plot_pacf(resid, lags=35)
    # plt.show()
    # # 2 qq图
    # qqplot(resid, line='q', fit=True).show()
    # plt.show()
    # 3 DW检验
    print('D-W检验的结果为：', sm.stats.durbin_watson(resid.values))
    # 解读：不存在一阶自相关
    # 4 LB检验
    print('残差序列的白噪声检验结果为：', acorr_ljungbox(resid, lags=1))  # 返回统计量、P值
    # 解读：残差是白噪声 p>0.05
    # confint,qstat,pvalues = sm.tsa.acf(resid.values, qstat=True)
    print(model.summary())
    fig = model.plot_diagnostics(figsize=(15, 12))  # plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为
    plt.show()
    return model


#获取预测结果，自定义预测误差
def PredictionAnalysis(data, model, dynamic=False):
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae
    pred=model.get_prediction(dynamic=dynamic,full_results=True)

    pci=pred.conf_int()#置信区间
    pm=pred.predicted_mean#预测值
    truth=data#真实值
    # pc=pd.concat([truth, pm, pci],axis=1)#按列拼接
    # pc.columns=['true','pred','up','low']#定义列索引
    print("1、MSE:{}".format(mse(truth,pm)))
    print("2、RMSE:{}".format(np.sqrt(mse(truth,pm))))
    print("3、MAE:{}".format(mae(truth,pm)))

    return pci, pm, truth
# 绘制预测结果
def PredictonPlot(pci, pm, truth):
    plt.figure(figsize=(10,8))
    print(pci)
    print(pm)
    print(truth)
    # plt.fill_between(pci.index,pci['upper value'],pci['lower value'],color='grey', alpha=0.15,label='confidence interval')#画出置信区间
    plt.plot(pm, label='base data')
    plt.plot(truth, label='prediction curve')
    plt.legend()
    plt.show()
    return True

# #预测未来
def Pred_Futrue(data, model, truth):
    forecast = model.get_forecast(steps=20)
    print(forecast)

    #预测整体可视化
    plt.figure(figsize=(20, 16))
    plt.subplot(2, 1, 1)
    plt.plot(pd.date_range(start='2016-01-03', end='2023-12-31', freq='W')
             , data, color='c', label='real')
    plt.title('train forecast results')
    pm = forecast.predicted_mean
    pm.to_csv('seasonal_pred.csv')    # 保存预测结果
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W')
             , truth, color='c')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W')
             , pm, color='y', label='forecast')
    plt.legend()
    plt.xlabel("time", fontsize=14)
    # plt.ylabel("pred_seasonal", fontsize=18)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    pm = forecast.predicted_mean
    calculate_errors(truth, pm)
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W')
             , truth, color='c', label='test')
    plt.plot(pd.date_range(start='2024-01-07', end='2024-05-19', freq='W')
             , pm, color='y', label='forecast')
    plt.legend()
    plt.title('test forecast results')
    plt.xlabel("time", fontsize=14)
    # plt.ylabel("pred_seasonal", fontsize=18)
    plt.grid(True)
    plt.subplots_adjust(hspace=0.3)
    plt.show()

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

# 0.数据准备
df = pd.read_csv('season.csv')
df.set_index('date', inplace=True)
# df = df.reset_index()
data_train, data_test = df[:-20], df[-20:]
# 1.绘制原始序列与自相关图
draw_ts(data_train['sz'])
# 2.单位根检验是否平稳
adfullerJY(data_train['sz'])
# 3.差分
# 3.1 判断最优差分步数
calculate_diff(data_train, max_diff=24)# 19
# 3.2 差分运算
d1_sale = diffYS(data_train)
# 4.白噪声检验
print('序列的白噪声检验结果为：', acorr_ljungbox(d1_sale, lags=[6, 12, 18, 24]))
# 5.模型定阶
SARIMA_search(data_train['sz'].dropna())
# 6.建立模型
# model1 = modelJY(data_train['sz'], 2, 1, 1, 1, 1, 1, 52)
model2 = modelJY(data_train['sz'], 2, 1, 2, 0, 19, 1, 52)# 这个误差更小一些
# 7.预测
# (1)静态预测：进行一系列的一步预测，即它必须用真实值来进行预测
# pci, pm, truth=PredictionAnalysis(data_train['value'], model)
# PredictonPlot(pci, pm, truth)
# (2)动态预测：进行多步预测，除了第一个预测值是用实际值预测外，其后各预测值都是采用递推预测
pci, pm, truth=PredictionAnalysis(data_train['value'], model2, dynamic=True)
PredictonPlot(pci, pm, truth)
# (3)预测未来(假设让他预测测试集的时间段，再根测试集做对比)
Pred_Futrue(data_train['sz'], model2, data_test['sz'])















