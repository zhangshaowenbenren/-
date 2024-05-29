from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import visdom
import numpy as np
import pickle
import lib
from torch.utils.data import DataLoader, TensorDataset
import argparse

with open('preprocessed_data.pkl', 'rb') as rb:
    data = pickle.load(rb)

degs = [25, 10, 0]
train_cycles = ['Cycle_1', 'Cycle_2', 'Cycle_3', 'Cycle_4', 'HWFET', 'LA92', 'NN', 'UDDS']
test_cycles = ['US06']

sd_steps, label_steps = 1000, 100
train_x, train_y = lib.get_processed_data(data, degs, train_cycles, sd_steps)
test_x, test_y = lib.get_processed_data(data, degs, test_cycles, sd_steps)

train_x, test_x, max0, min0 = lib.normalize(train_x, test_x)

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch SOC Estimation')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=233, metavar='S', help='random seed (default: 233)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_channel', type=int, default=1)                       # 用于预测的循环数
    parser.add_argument('--output_size', type=int, default=100)                       # 预测的关键节点个数
    parser.add_argument('--num_hiddens', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.2)
    return parser


# 序列预测的模型
class CNN_GRU(nn.Module):
    def __init__(self, inputChannels=1, output_size=100, dropout=0.):
        super(CNN_GRU, self).__init__()
        self.model_type = 'CNN_GRU'
        self.block0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inputChannels, 16, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1))),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=(2, 1))),
            ('conv2', nn.Conv2d(16, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
            ('bn2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=(2, 1))),
            ('conv3', nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2, 1)))
        ]))

        self.GRU = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.linear1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 64))
        self.mid_layer = nn.Sequential(nn.Linear(64, 64))                      # 残差块的分支
        self.linear = nn.Sequential(nn.Dropout(dropout), nn.Linear(960, output_size))
        self.reLU = nn.ReLU()


    def forward(self, x):
        # x的shape：批量 * 通道维数 * 长 * 宽
        x = x.unsqueeze(1)
        y = self.block0(x)
        y = y.squeeze(3)
        y = y.permute(0, 2, 1)
        y1, _ = self.GRU(y)
        y1 = self.linear1(y1)
        y2 = y1 + self.mid_layer(y)                        # 残差连接
        y2 = self.reLU(y2)
        y2 = self.linear(y2.reshape(y2.shape[0], -1))
        return y2


class CNN_GRU2(nn.Module):
    def __init__(self, inputChannels=3, output_size=100, dropout=0.):
        super(CNN_GRU2, self).__init__()
        self.model_type = 'CNN_GRU2'
        self.block0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(inputChannels, 16, kernel_size=7, stride=2, padding=3)),
            ('bn1', nn.BatchNorm1d(16)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(kernel_size=2)),
            ('conv2', nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)),
            ('bn2', nn.BatchNorm1d(32)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(kernel_size=2)),
            ('conv3', nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)),
            ('bn3', nn.BatchNorm1d(64)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(kernel_size=2))
        ]))

        self.GRU = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.linear1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, 64))
        self.mid_layer = nn.Sequential(nn.Linear(64, 64))                      # 残差块的分支
        self.conv1d = nn.Sequential(nn.Conv1d(15, 1, 1), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(64, output_size))
        self.reLU = nn.ReLU()


    def forward(self, x):
        # x的shape：批量 * 通道维数 * 长 * 宽
        # x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        y = self.block0(x)
        # y = y.squeeze(3)
        y = y.permute(0, 2, 1)
        y1, _ = self.GRU(y)
        y1 = self.linear1(y1)
        y2 = y1 + y                            # 残差连接
        y2 = self.reLU(y2)
        y2 = self.conv1d(y2).squeeze(dim=1)
        y2 = self.linear(y2)
        return y2


wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))                 # 图像的标例
wind.line([[0., 0.]], [0.], win='test0', opts=dict(title='MAE', legend=['train_MAE', 'test_MAE']))
wind.line([[0., 0.]], [0.], win='test1', opts=dict(title='RMSE', legend=['train_RMSE', 'test_RMSE']))

parser = get_parser()
args = parser.parse_args()

net = CNN_GRU(inputChannels=args.input_channel, output_size=args.output_size, dropout=args.dropout)
# net = CNN_GRU2(inputChannels=3, output_size=args.output_size, dropout=args.dropout)
device = lib.try_gpu()
dataSet1 = TensorDataset(train_x, train_y)
train_iter = DataLoader(dataSet1, batch_size=args.batch_size, shuffle=True)

dataSet2 = TensorDataset(test_x, test_y)
test_iter = DataLoader(dataSet2, batch_size=args.batch_size, shuffle=True)

need_train = True
if need_train:
    lib.train_ch6(net, train_iter, test_iter, args.num_epochs, lr=args.lr, device=device, wind=wind, need_variation_loss=False)
    torch.save(net.state_dict(), 'model_save/%s' % net.model_type)
else:
    net.load_state_dict(torch.load('model_save/%s' % net.model_type))

scatter_X, scatter_Y = [], []
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flat
net.eval()
for k, deg in enumerate(degs):
    X, Y, Time = [], [], []
    for cycle in test_cycles:
        cur = data[deg][cycle]
        for i in range(sd_steps, len(cur) - label_steps, 50):
            x, y = torch.tensor(cur[i - sd_steps:i, 1:4], dtype=torch.float32), torch.tensor(cur[i:i + label_steps:, 4], dtype=torch.float32)
            X.append(x)
            Y.append(np.array(y[:50]).reshape(-1))
    X, Y = torch.stack(X), np.concatenate(Y)
    scatter_X.extend(list(Y))
    X = (X - min0) / (max0 - min0)
    pred_y = []
    X = X.to(device)
    dataSet = TensorDataset(X)
    train_iter = DataLoader(dataSet, batch_size=args.batch_size, shuffle=False, drop_last=False)
    with torch.no_grad():
        for x in train_iter:
            y_pred = net(x[0])
            pred_y.append(np.array(y_pred[:, :50].detach().cpu()).reshape(-1))
    pred_y = np.concatenate(pred_y)
    scatter_Y.extend(list(pred_y))
    Ttime = np.arange(0, len(Y)) / 10
    axes[k].plot(Ttime, Y, label='true', color='r')
    axes[k].plot(Ttime, pred_y, label='predict', color='b')
    axes[k].set_title('%s℃'%deg)
    print('在%s温度下的测试集评估指标:' % deg, end='  ')
    print('MAE:%0.3f, RMSE:%0.3f' % (np.sum(np.abs(pred_y - Y)) / Y.size, np.sqrt(np.sum(np.square(pred_y - Y)) / Y.size)))
plt.savefig('res_fig/%s/在不同温度的测试集下的预测结果.png' % net.model_type, dpi=1000, bbox_inches='tight')              # 指定分辨率

plt.figure()                                # 画散点图
scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)     # 转化为Numpy格式
plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
print('整体的评估指标:MAE:%0.3f, RMSE:%0.3f'%(np.sum(np.abs(scatter_Y - scatter_X)) / scatter_X.size, np.sqrt(np.sum(np.square(scatter_Y - scatter_X)) / scatter_X.size)))
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.plot([0, 1], [0, 1], '--', zorder=1)
plt.colorbar()
plt.savefig('res_fig/%s/在不同温度测试集下的整体预测偏差.png' % net.model_type, dpi=1000, bbox_inches='tight')              # 指定分辨率
plt.show()