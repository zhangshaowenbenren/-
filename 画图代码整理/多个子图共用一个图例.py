import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 10
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 3, figsize=(6, 2.5))
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.75, wspace=0.25, hspace=0.25)     # 调整子图间的距离

ax = ax.flat

x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = y1 * y2
y4 = y1 + y2

labels = ['A', 'B', 'C', 'D']

for i in range(3):
    for j, y in enumerate([y1, y2, y3, y4]):
        ax[i].plot(x, y, label=labels[j])

lines = []
labels = []
axLine, axLabel = ax[0].get_legend_handles_labels()
lines.extend(axLine)
labels.extend(axLabel)
fig.legend(lines, labels, ncol=2, loc='upper center')          # 图例的位置，bbox_to_anchor=(0.5, 0.92),
plt.show()