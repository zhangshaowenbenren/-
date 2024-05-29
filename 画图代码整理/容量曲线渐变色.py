import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))

norm = plt.Normalize(1100, 2700)
cmap = plt.cm.coolwarm              # 色系
# 其他颜色选项
# ['Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'Dark2', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1', 'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'tab10', 'tab20', 'tab20b', 'tab20c', 'terrain', 'twilight', 'twilight_shifted', 'viridis', 'winter']

with open('数据/电池容量数据.pkl', 'rb') as fp:
    res_dict = pickle.load(fp)

for capacity in res_dict:
    x = np.array(list(capacity.keys()))
    y = np.array(list(capacity.values())) / 1000
    ax.plot(x, y, color=cmap((max(x) - 1100) / 1600), linewidth=2)         # 根据最大循环数目进行显色
    # color(c)，c的取值范围为0~1，表示从A颜色渐变到B颜色

ax.set_xlabel('循环数')
ax.set_ylabel('放电容量/Ah')

# 不加norm，则颜色条的取值范围为[0,1]
# 指定norm后，则取值范围会线性放大到与Norm取值范围一样,即从[0, 1]映射到[1100, 2700]
clb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
clb.ax.set_title('循环', fontsize=10)
# ax.set_title('各电池的容量-循环曲线图', ha='center')
plt.show()