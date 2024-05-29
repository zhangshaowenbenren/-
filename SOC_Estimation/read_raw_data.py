import scipy
import numpy as np
import pickle
import os
import collections
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号

root_path = 'D:\Program Files\zsw\data\Panasonic 18650PF Data'
cell_deg_dict = collections.defaultdict(dict)
degs = [-20, -10, 0, 10, 25]
cycles = ['Cycle_1', 'Cycle_2', 'Cycle_3', 'Cycle_4', 'HWFET', 'LA92', 'NN', 'UDDS', 'US06']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
axes = axes.flat
for deg in degs:
    dir_path = root_path + '\%sdegC\Drive cycles' % deg
    for cycle in cycles:
        for p in os.listdir(dir_path):
            for ax in axes:                              # 清楚子图绘制的内容
                ax.cla()
            if cycle in p:
                f = scipy.io.loadmat(dir_path + '\\' + p)
                f = f['meas'][0][0]
                Time = f[0]
                for i in range(len(Time)):
                    cur = Time[i][0][0].split(' ')
                    if len(cur) == 1:
                        Time[i] = Time[i - 1]
                    else:
                        cur = cur[1].split(':')
                        Time[i] = int(cur[0]) * 3600 + int(cur[1]) * 60 + int(cur[2])
                Time = np.array(Time, dtype=np.float32)
                Voltage = f[1]
                Current = f[2]
                Ah = f[3]
                Temp = f[6]
                # time、V、I、SOC
                if Ah[-1] == 0:
                    print('error')

                # tmp = np.concatenate([Time, Voltage, Current, Temp, (Ah + 2.9) / 2.9], axis=1, dtype=np.float32)
                print(Ah[-1])
                tmp = np.concatenate([Time, Voltage, Current, Temp, (Ah - Ah[-1]) / -Ah[-1]], axis=1, dtype=np.float32)
                cell_deg_dict[deg][cycle] = tmp
                axes[0].plot(np.arange(len(Time)) / 10, Voltage, linewidth=1)
                axes[0].set_ylabel('电压/V', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离
                axes[0].set_xlabel('时间/s', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离

                axes[1].plot(np.arange(len(Time)) / 10, Current, linewidth=1)
                axes[1].set_ylabel('电流/A', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离
                axes[1].set_xlabel('时间/s', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离

                axes[2].plot(np.arange(len(Time)) / 10, Ah, linewidth=1)
                axes[2].set_ylabel('放电量/Ah', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离
                axes[2].set_xlabel('时间/s', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离

                axes[3].plot(np.arange(len(Time)) / 10, Temp, linewidth=1)
                axes[3].set_ylabel('温度/℃', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离
                axes[3].set_xlabel('时间/s', fontsize=None, labelpad=0.4)  # labelpad设置y轴标签与子图的距离
                plt.savefig('raw_data_fig/%s+%s.png'%(deg, cycle), dpi=300, bbox_inches='tight')  # 指定分辨率

                break

with open('preprocessed_data2.pkl', 'wb') as fp:
    pickle.dump(cell_deg_dict, fp)