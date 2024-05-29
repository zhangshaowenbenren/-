# 预测输入数据为电压、电流和温度；   模型为CNN+Transformer；  结合多任务学习框架
# 采用与HUST数据集相同的样本构造方式
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import lib
import time
import visdom
import numpy as np
import gc
from torch.utils.data import DataLoader, TensorDataset
import argparse
import pickle
import random
from torch.nn import functional as F
import os

need_permute = False
start = time.time()
def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch RUL Estimation')
    parser.add_argument('--load_raw_data', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=233, metavar='S', help='random seed (default: 233)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_size', type=int, default=2)                # 预测的关键节点个数
    parser.add_argument('--num_hiddens', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--L', type=int, default=30)                         # 输入循环的个数
    parser.add_argument('--S', type=int, default=2)                          # 滑动窗口移动的步长
    parser.add_argument('--num_samples', type=int, default=200)              # 重采样的个数
    parser.add_argument('--need_train', type=bool, default=True)            # 是否需要重新训练模型
    parser.add_argument('--cycle_steps', type=int, default=3)
    return parser


parser = get_parser()
args = parser.parse_args()
# 设置随机数种子
random.seed(args.seed)
np.random.seed(args.seed)
if args.load_raw_data:
    if os.path.exists('delete_lib_cycle.txt'):
        os.remove('delete_lib_cycle.txt')
    train_data, test_data, _ = lib.load_Data(cross_val=True, is_random=False)
    print(list(test_data.keys()))
    train_X, train_Y = lib.getRULTrainData(train_data, num_samples=args.num_samples, H=args.H, L=args.L, S=args.S, train=True, is_diff=True, cycle_steps=args.cycle_steps)
    del train_data
    gc.collect()

    test_X0, test_Y0 = lib.getRULTrainData(test_data, num_samples=args.num_samples, H=args.H, L=args.L, S=args.S, train=True, is_diff=True, cycle_steps=args.cycle_steps)
    test_X, test_Y = lib.getRULTrainData(test_data, num_samples=args.num_samples, H=args.H, L=args.L, S=args.S, train=False, is_diff=True, cycle_steps=args.cycle_steps)
    del test_data
    gc.collect()

    with open('processed_data_MIT.pkl', 'wb') as fp:
        pickle.dump([train_X, train_Y, test_X, test_Y, test_X0, test_Y0], fp)
else:
    with open('processed_data_MIT.pkl', 'rb') as fp:
        train_X, train_Y, test_X, test_Y, test_X0, test_Y0 = pickle.load(fp)

# 归一化的代码
train_X, test_X0, max0, min0 = lib.normalize(train_X, test_X0)
with open('normalize_max_min_MIT.pkl', 'wb') as fp:
    pickle.dump([max0, min0], fp)

if need_permute:
    train_X = train_X.permute(0, 2, 1, 3)
    test_X0 = test_X0.permute(0, 2, 1, 3)


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)


    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class base_CNN(nn.Module):
    def __init__(self, feature_size=64, inputChannels=1, dropout=0.2, num_AvgPool=(1, 2)):
        # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(base_CNN, self).__init__()
        self.model_type = 'CNN3'
        self.CNN3_01 = nn.Sequential(OrderedDict(
            [('conv3', nn.Conv3d(inputChannels, 16, kernel_size=(3, 4, 7), stride=(1, 1, 1), padding=(1, 0, 3))),
             ('bn', nn.BatchNorm3d(16)),
             ('relu', nn.ReLU())
        ]))

        self.CNN3_02 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv3d(16, 32, kernel_size=(3, 3, 7), stride=(2, 1, 3), padding=(1, 0, 3))),
            ('bn', nn.BatchNorm3d(32)),
            ('dropout', nn.Dropout(dropout)),
            ('relu', nn.ReLU())
        ]))

        self.conv3_res = nn.Conv3d(16, 32, kernel_size=1, stride=(2, 3, 3))

        self.CNN2_01 = Residual(32, feature_size, use_1x1conv=True, strides=(3, 3))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=2)


    def forward(self, src):
        src = src.unsqueeze(1)
        src = self.CNN3_01(src)
        y1 = self.CNN3_02(src)
        y2 = self.conv3_res(src)
        src = self.relu(y1 + y2)
        src = src.squeeze()
        y3 = self.CNN2_01(src)
        src = self.flatten(y3)
        return src


class CNN3_diff(nn.Module):
    def __init__(self, feature_size=128, inputChannels=1, dropout=0.2, num_AvgPool=(1, 2)):
        # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(CNN3_diff, self).__init__()
        self.model_type = 'CNN3'
        self.model = base_CNN(feature_size=feature_size, inputChannels=inputChannels, dropout=dropout, num_AvgPool=num_AvgPool)
        self.attention = lib.MultiHeadAttention(key_size=46, query_size=46, value_size=46, num_hiddens=64, num_heads=4, dropout=dropout)
        self.linear = nn.Linear(46, 64)                       # 残差块的分支
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(feature_size, 16, 1)
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv1d(16, 1, 1)), ('relu', nn.ReLU())]))
        self.conv2 = nn.Sequential(OrderedDict([('conv', nn.Conv1d(16, 1, 1)), ('relu', nn.ReLU())]))

        self.lr1 = nn.Linear(64, 10)
        self.lr2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.n_tasks = 2


    def forward(self, src):
        src = self.model(src)
        y1 = self.attention(src, src, src)
        y2 = y1 + self.linear(src)                  # 残差连接
        output = self.relu(y2)
        y = self.dropout(output)
        y = self.conv(y)
        y = self.dropout(self.relu(y))

        z1 = self.conv1(y)
        z1 = z1.squeeze()
        z1 = self.lr1(z1)

        z2 = self.conv2(y)
        z2 = z2.squeeze()
        z2 = self.lr2(z2)
        return torch.concat([z1, z2], dim=1)


    def get_last_shared_layer(self):
        return self.conv


class MTLTrain(nn.Module):
    def __init__(self, model, L=10):
        super(MTLTrain, self).__init__()
        self.model = model
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        self.loss = nn.HuberLoss(delta=20)
        self.model_type = 'MTL_MIT'
        self.L = L


    def forward(self, x, ts):
        y_hat = self.model(x)
        task_loss = []
        task_loss.append(self.loss(y_hat[:, :self.L], ts[:, :self.L]))
        task_loss.append(self.loss(y_hat[:, self.L], ts[:, self.L]))
        task_loss = torch.stack(task_loss)
        return task_loss, y_hat


    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


class CNN_LSTM(nn.Module):
    def __init__(self, L, feature_size=64, inputChannels=6, dropout=0.2):
        # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(CNN_LSTM, self).__init__()
        self.model_type = 'CNN_LSTM'
        self.CNN1 = nn.Sequential(nn.Conv2d(inputChannels, 16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
                                      nn.BatchNorm2d(16), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))

        self.CNN2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                                      nn.BatchNorm2d(32), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                      nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                                      nn.BatchNorm2d(64)
                                     )
        self.CNN3 = nn.Sequential( nn.Conv2d(64, feature_size, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1)),
                                   nn.BatchNorm2d(feature_size), nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))

        self.dropout = nn.Dropout(dropout)
        self.res_connect_01 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=1), nn.ReLU(), nn.Linear(50, 6))
        self.res_connect_02 = nn.Linear(64, 128)
        self.LSTM = nn.LSTM(feature_size, feature_size // 2, bidirectional=True, batch_first=True)
        self.attention = lib.MultiHeadAttention(key_size=64, query_size=64, value_size=64, num_hiddens=64, num_heads=4, dropout=dropout)
        self.SOH = nn.Linear(feature_size, 1)
        self.RUL = nn.Linear(L, 1)
        self.relu = nn.ReLU()
        self.LR = nn.Linear(feature_size, feature_size)


    def forward(self, x):
        x = self.CNN1(x)
        y1 = self.CNN2(x)
        y2 = self.res_connect_01(x)
        y = self.relu(y1 + y2)              # 参加连接
        y = self.CNN3(y)
        y = y.squeeze(3)
        y = y.permute(0, 2, 1)
        y2, _ = self.LSTM(y)
        # y2 = self.attention(y, y, y)
        y = self.relu(self.LR(y2) + y)            # 残差连接
        y = self.dropout(y)
        soh = self.SOH(y).squeeze(2)
        rul = self.RUL(soh)
        return torch.concat([soh, rul], dim=1)


wind = visdom.Visdom(env='main')
wind.line([[0., 0.]], [0.], win='SOH', opts=dict(title='SOH', legend=['train_MAE', 'val_MAE']))
wind.line([[0., 0.]], [0.], win='RUL', opts=dict(title='RUL', legend=['train_MAE', 'val_MAE']))
wind.line([[0., 0.]], [0.], win='loss', opts=dict(title='loss', legend=['SOH_loss', 'RUL_loss']))
wind.line([[1., 1.]], [0.], win='weights', opts=dict(title='weights', legend=['SOH_weight', 'RUL_weight']))
wind.line([0.], [0.], win='loss_ratios', opts=dict(title='loss_ratios', legend=['loss_ratios']))
wind.line([0.], [0.], win='grad_norm_losses', opts=dict(title='grad_norm_losses', legend=['grad_norm_losses']))

parser = get_parser()
args = parser.parse_args()

base_net = CNN3_diff(feature_size=64, inputChannels=1, dropout=args.dropout)
net = MTLTrain(base_net)

device = lib.try_gpu()
dataSet1 = TensorDataset(train_X, train_Y)
train_iter = DataLoader(dataSet1, batch_size=args.batch_size, shuffle=True)

dataSet3 = TensorDataset(test_X0, test_Y0)
test_iter = DataLoader(dataSet3, batch_size=args.batch_size, shuffle=True)

if args.need_train:
    lib.train_MTL(net, train_iter, test_iter, args.num_epochs, lr=args.lr, device=device, alpha=0.12, wind=wind, L=10, need_plot=True)
    torch.save(net.state_dict(), 'model_save/%s' % net.model_type)
else:
    net.load_state_dict(torch.load('model_save/%s' % net.model_type + '_best'))

# 测试集上的评估
scatter_X1, scatter_Y1, scatter_X2, scatter_Y2 = [], [], [], []
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.4)
axes = axes.flat
net.eval()
error_pd, index = [], []

for j, (key, value) in enumerate(test_X.items()):
    X, Y = value, test_Y[key]
    scatter_X1.extend(list(Y[:, :args.L // args.cycle_steps].reshape(-1)))
    scatter_X2.extend(list(Y[:, args.L // args.cycle_steps]))
    X = (X - min0) / (max0 - min0)
    if need_permute:
        X = X.permute(0, 2, 1, 3)
    X = X.to(device)
    net.to(device)
    with torch.no_grad():
        y_pred = net.model(X)
        pred_y1 = y_pred[:, :args.L // args.cycle_steps].detach().cpu()
        pred_y2 = y_pred[:, args.L // args.cycle_steps].detach().cpu()
    error_pd.append([len(pred_y2) * args.S,
                     float(torch.sum(torch.abs(pred_y1 - Y[:, :args.L // args.cycle_steps])) / (len(pred_y1) * args.L // args.cycle_steps)),
                     float(torch.sqrt(torch.sum(torch.square(pred_y1 - Y[:, :args.L // args.cycle_steps])) / (len(pred_y1) * args.L // args.cycle_steps))),
                     float(torch.sum(torch.abs(pred_y2 - Y[:, args.L // args.cycle_steps])) / len(pred_y2)),
                     float(torch.sqrt(torch.sum(torch.square(pred_y2 - Y[:, args.L // args.cycle_steps])) / len(pred_y2)))])
    # 只画12个电池的预测结果
    if j < 12:
        Ttime = np.arange(0, len(Y)) * args.S
        axes[2 * j].plot(Ttime, Y[:, args.L // args.cycle_steps- 1], label='true', color='r', linewidth=0.5)
        axes[2 * j].plot(Ttime, pred_y1[:, -1], label='predict', color='b', linewidth=0.5)
        axes[2 * j].set_title('%s_SOH' % key)
        axes[2 * j + 1].plot(Ttime, Y[:, args.L // args.cycle_steps], label='true', color='r', linewidth=0.5)
        axes[2 * j + 1].plot(Ttime, pred_y2, label='predict', color='b', linewidth=0.5)
        axes[2 * j + 1].set_title('%s_RUL' % key)
    index.append(key)
    scatter_Y1.extend(list(pred_y1.reshape(-1)))
    scatter_Y2.extend(list(pred_y2))
plt.savefig('res_fig/%s/SOH和RUL多任务学习预测结果.jpeg' % net.model_type, dpi=800, bbox_inches='tight')              # 指定分辨率

error_pd = pd.DataFrame(error_pd, index=index, columns=['EOL', 'SOH_MAE', 'SOH_RMSE', 'RUL_MAE', 'RUL_RMSE'])
error_pd.to_csv('error/%s/test_error.csv' % net.model_type)

def R_square(hat, value):
    hat, value = np.array(hat), np.array(value)
    m = np.mean(value)
    a1 = np.sum((hat - value) * (hat - value))
    a2 = np.sum((value - m) * (value - m))
    return 1 - a1 / a2

# 画散点图
scatter_X1, scatter_Y1 = np.array(scatter_X1), np.array(scatter_Y1)                                                # 转化为Numpy格式
scatter_X2, scatter_Y2 = np.array(scatter_X2), np.array(scatter_Y2)
plt.figure(figsize=(5, 5))
plt.scatter(scatter_X1, scatter_Y1, c=abs(scatter_X1 - scatter_Y1), cmap="seismic", zorder=2, s=3)
print('SOH整体的评估指标:MAE:%0.3f, RMSE:%0.3f, R_2:%0.3f' % (np.sum(np.abs(scatter_Y1 - scatter_X1)) / scatter_X1.size,
                                                              np.sqrt(np.sum(np.square(scatter_Y1 - scatter_X1)) / scatter_X1.size),
                                                              R_square(scatter_Y1, scatter_X1)))
plt.xlim([0.8, 1])
plt.ylim([0.8, 1])
plt.plot([0.8, 1], [0.8, 1], '--', zorder=1)
plt.colorbar()
plt.savefig('res_fig/%s/SOH在测试集下的整体预测偏差.jpeg' % net.model_type, dpi=800, bbox_inches='tight')              # 指定分辨率

plt.figure(figsize=(5, 5))
plt.scatter(scatter_X2, scatter_Y2, c=abs(scatter_X2 - scatter_Y2), cmap="seismic", zorder=2, s=3)
print('RUL整体的评估指标:MAE:%0.3f, RMSE:%0.3f, WMAPE:%0.3f, R_2:%0.3f'%(np.sum(np.abs(scatter_Y2 - scatter_X2)) / scatter_X2.size,
                                                        np.sqrt(np.sum(np.square(scatter_Y2 - scatter_X2)) / scatter_X2.size),
                                                        np.sum(np.abs(scatter_Y2 - scatter_X2)) / np.sum(scatter_X2),
                                                        R_square(scatter_Y2, scatter_X2)))
plt.xlim([0, 2000])
plt.ylim([0, 2000])
plt.plot([0, 2000], [0, 2000], '--', zorder=1)
plt.colorbar()
plt.savefig('res_fig/%s/RUL在测试集下的整体预测偏差.jpeg' % net.model_type, dpi=800, bbox_inches='tight')              # 指定分辨率
plt.show()