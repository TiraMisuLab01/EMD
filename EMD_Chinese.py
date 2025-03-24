# 单   位： 苏州大学
# 作   者： 许圣
# 修改时间： 2023/8/3 17:17
# 代码基于PyEMD进行修改：https://github.com/tianyagk/PyEMD
# EMD算法讲解：https://cloud.tencent.com/developer/article/1661978
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
# 1.求极大值点和极小值点
"""
通过Scipy的argrelextrema函数获取信号序列的极值点
"""
# 构建100个随机数
data = np.random.random(100)
# 获取极大值
max_peaks = argrelextrema(data, np.greater)
# 获取极小值
min_peaks = argrelextrema(data, np.less)
# 绘制极值点图像
plt.figure(figsize=(10, 5))
plt.plot(data)
plt.scatter(max_peaks, data[max_peaks], c='r', label='局部极大值')
plt.scatter(min_peaks, data[min_peaks], c='b', label='局部极小值')
plt.legend()
plt.xlabel('时间(s)')
plt.ylabel('振幅')
plt.title("寻找峰值")
# 2. 拟合包络函数
# 进行样条差值
import scipy.interpolate as spi
data = np.random.random(100) - 0.5
index = list(range(len(data)))
# 获取极值点
max_peaks = list(argrelextrema(data, np.greater)[0])
min_peaks = list(argrelextrema(data, np.less)[0])
# 将极值点拟合为曲线
ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值
ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值
# 计算平均包络线
iy3_mean = (iy3_max + iy3_min) / 2
# 绘制图像
plt.figure(figsize=(10, 5))
plt.plot(data, label='原始信号')
plt.plot(iy3_max, label='最大波峰振幅')
plt.plot(iy3_min, label='最小波峰振幅')
plt.plot(iy3_mean, label='平均值')
plt.legend()
plt.xlabel('时间 (s)')
plt.ylabel('微伏 (uV)')
plt.title("三次样条插值")
# 3.获取本征模函数（IMF）
def sifting(data):
    index = list(range(len(data)))
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值
    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值
    iy3_mean = (iy3_max + iy3_min) / 2
    return data - iy3_mean
def hasPeaks(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
    if len(max_peaks) > 3 and len(min_peaks) > 3:
        return True
    else:
        return False
# 判断IMFs
def isIMFs(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
    if min(data[max_peaks]) < 0 or max(data[min_peaks]) > 0:
        return False
    else:
        return True
def getIMFs(data):
    while (not isIMFs(data)):
        data = sifting(data)
    return data
# EMD函数
def EMD(data):
    IMFs = []
    while hasPeaks(data):
        data_imf = getIMFs(data)
        data = data - data_imf
        IMFs.append(data_imf)
    return IMFs
# 绘制对比图
data = np.random.random(1000) - 0.5
IMFs = EMD(data)
n = len(IMFs) + 1
# 原始信号
plt.figure(figsize=(12, 8))
plt.subplot(n, 1, 1)
plt.plot(data)
plt.ylabel('振幅')
plt.title("原始信号 ")
# 若干条IMFs曲线
for i in range(0, len(IMFs)):
    plt.subplot(n, 1, i + 2)
    plt.plot(IMFs[i])
    plt.ylabel('振幅')
    plt.title("IMFs " + str(i + 1))
plt.xlabel('时间 (s)')
plt.ylabel('振幅')
plt.subplots_adjust(hspace=2.0)  # 增加子图间的高度间距
plt.show()
