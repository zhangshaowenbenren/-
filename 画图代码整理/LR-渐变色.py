from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import pickle

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 10               # 12对应宋体小四大小
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

with open('数据/散点图数据.pkl', 'rb') as fp:
    data = pickle.load(fp)

fig, axes = plt.subplots(3, 2, figsize=(7, 6))                                               # 子图大小
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, wspace=0.35, hspace=0.35)   # 子图间距
axes = axes.flat

y_labels = ['最大值', '平均值', '最小值', '方差', '偏度', '峰度']
ylims = [[4.15, 4.20],
         [4.14, 4.20],
         [4.12, 4.20],
         [0, 0.0005],
         [-1, 2],
         [-2, 2]]

X, Y = [[], [], [], [], [], []], [[], [], [], [], [], []]

a = [4.16, 4.15, 4.18, 0.00045, -0.7, 1.5]

for key, cell in data.items():
    x = cell['V']
    y = cell['Q']
    for k in range(6):
        X[k].append(y)
        Y[k].append(x[:, k])


for k in range(6):
    # 拟合线性模型
    x, y = np.concatenate(X[k]), np.concatenate(Y[k])
    x, y = np.array(x), np.array(y)
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    # 计算每个点到拟合曲线的距离
    pred_y = model.predict(x.reshape(-1, 1))
    distances = np.abs(y - pred_y)
    r_square = r2_score(y, pred_y)
    print('样本相关系数 R:', r_square)

    # 创建颜色映射、根据点到直线的距离进行渐变色
    norm = plt.Normalize(distances.min(), distances.max())
    cmap = plt.cm.coolwarm            # 色系
    colors = cmap(norm(distances))    # 归一化到0~1之间

    # 绘制散点图
    axes[k].scatter(x, y, c=colors, s=0.4)
    axes[k].set_ylabel(y_labels[k])
    axes[k].set_xlabel('SOH')
    axes[k].set_ylim(ylims[k])                       # y轴的取值范围
    axes[k].text(0.85, a[k], f'相关系数： {r_square:.2f}')   # 文本的添加位置

# 可以保存画图的图片
# plt.savefig('paper_fig/特征相关系数图.jpeg', dpi=800, bbox_inches='tight')
plt.show()