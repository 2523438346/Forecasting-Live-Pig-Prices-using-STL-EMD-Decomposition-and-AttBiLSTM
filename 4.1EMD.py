import numpy as np
import pandas as pd
from PyEMD import EMD
from PyEMD import Visualisation
t = np.linspace(0, 437, 438)
# s = np.cos(2 * np.pi * 11 * t * t) + t ** 2
df = pd.read_csv('resid.csv').set_index('date')
series = df['生猪']
# 创建EMD对象并进行分解
emd = EMD()
emd.emd(series.values)
IMFs, residue = emd.get_imfs_and_residue()
print(IMFs)
# 保存
# 将IMFs和残差转换为DataFrame
imfs_df = pd.DataFrame(IMFs).T  # 假设IMFs是二维数组，这里转置以匹配Pandas的列格式
residue_df = pd.DataFrame(residue, columns=['Residue'])
combined_df = pd.concat([imfs_df, residue_df], axis=1)
combined_df.to_csv('IMFs_and_Residue.csv', index=False)
# # 使用Visualisation模块来绘图
vis = Visualisation()
vis.plot_imfs(imfs=IMFs, residue=residue, t=t, include_residue=True)
vis.show()
