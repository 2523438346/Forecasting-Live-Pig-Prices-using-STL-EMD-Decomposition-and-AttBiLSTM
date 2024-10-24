import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import pylab
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#此步骤是为了解决字体问题
plt.rcParams['axes.unicode_minus']=False#此步骤是为了解决字体问题
# 1.数据预处理
# df = data_pretreatment('LivePig_WaiSanYuan_Heilongjiang.xlsx', '日期')
df = pd.read_excel('basic_data.xlsx', parse_dates=['date'])
col = '生猪'
df = df[col]

# 2.模型构建
def STL_Create():
    plt.rc("figure", figsize=(10, 6))  # 设置绘图区尺寸
    # df=pd.read_csv('C:/Users/Administrator/Desktop/STL.csv',encoding='ANSI',usecols=['time', 'value'],index_col='time') #读取原始数据，这里的time,value是原始数据中的列标签名，此外encoding参数需要根据csv的编码确认，usecols选择使用的列，index_col参数选择索引列
    # print(df)#查看读取好的文件
    stl = STL(df, period=52, robust=False)  # robust为True时会用一种更严格的方法来约束trend和season，同时也会导致更大的resid
    res = stl.fit()
    res.plot()
    plt.savefig(f"figures/STL_{col}.png")  # 保存STL分解结果图
    plt.clf()  # 清空绘图区防止重复绘图
    df['trend'] = res.trend  # 保存分解后数据
    df['seasonal'] = res.seasonal
    df['resid'] = res.resid
    dataframe = pd.DataFrame(
        {'trend': df['trend'], 'season': df['seasonal'], 'resid': df['resid']})  # 将分解结果按照列导入csv文件
    print(f"STL结果:\n{dataframe}")
    dataframe.to_csv(f"figures/table_{col}.csv", sep=',')
    # 查看残差正态性检验的均值,如果残差呈现出以0为均值的近似正太分布(这不是必须的)那么说明我们使用了正确的分解方法。
    mean = df.resid.mean()
    print('residual mean:', mean)
    sns.distplot(df.resid)  # 绘制带正态曲线的概率密度直方图
    plt.savefig("figures/{}mean={}.png".format(col, mean))
    plt.clf()
    sm.qqplot(df.resid, line='s')  # 绘制正态检验QQ图
    plt.savefig(f"figures/{col}QQ.png")
    return dataframe
if __name__ == '__main__':
    STL_Create()


