# SOC论文的标签方式

import matplotlib.pyplot as plt
import torch
from torch import nn
import visdom
import numpy as np
import pickle
import lib
from torch.utils.data import DataLoader, TensorDataset
import argparse
from collections import OrderedDict

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

with open('preprocessed_data.pkl', 'rb') as rb:
    data = pickle.load(rb)

degs = [25, 10, 0]
train_cycles = ['Cycle_1', 'Cycle_2', 'Cycle_3', 'Cycle_4', 'HWFET', 'LA92', 'NN', 'UDDS']
test_cycles = ['US06']

sd_steps, label_steps = 1000, 100
train_x, train_y = lib.get_processed_data(data, degs, train_cycles, sd_steps)
test_x, test_y = lib.get_processed_data(data, degs, test_cycles, sd_steps)

train_x, test_x, max0, min0 = lib.normalize(train_x, test_x)
with open('normalize_max_min.pkl', 'wb') as fp:
    pickle.dump([max0, min0], fp)

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch SOC Estimation')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=233, metavar='S', help='random seed (default: 233)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--output_size', type=int, default=100)      # 预测的关键节点个数
    parser.add_argument('--num_hiddens', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--need_train', type=bool, default=False)
    return parser


#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout=0., max_len=1500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class CNN2d_TransAm(nn.Module):
    def __init__(self, feature_size=128, num_layers=2, dropout=0.2):
        # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(CNN2d_TransAm, self).__init__()
        self.model_type = 'CNN2d_Transformer'
        self.embedding = nn.Sequential(nn.Linear(1, feature_size), nn.ReLU())
        self.block0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 16, kernel_size=(7, 3), stride=(5, 1), padding=(3, 1))),
            ('bn1', nn.BatchNorm2d(16)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(kernel_size=(2, 1))),
            ('conv2', nn.Conv2d(16, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(kernel_size=(2, 1))),
            ('conv3', nn.Conv2d(64, feature_size, kernel_size=(3, 3), stride=(2, 1), padding=(1, 0))),
            ('bn3', nn.BatchNorm2d(feature_size)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2, 1)))
        ]))

        self.pos_encoder = PositionalEncoding(feature_size, dropout=dropout)                                    # 位置编码前要做归一化，否则捕获不到位置信息
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout, dim_feedforward=256, batch_first=True)     # 多头注意力
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.conv1d = nn.Conv1d(6, 1, 1)
        self.decoder = nn.Sequential(nn.Linear(feature_size, 100))               # 这里用全连接层代替了decoder， 其实也可以加一下Transformer的decoder试一下效果
        self.GRU = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.linear1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 128))
        self.mid_layer = nn.Sequential(nn.Linear(128, 128))                      # 残差块的分支
        self.relu = nn.ReLU()


    def forward(self, src):
        src = src.unsqueeze(dim=1)
        src = self.block0(src)
        src = src.squeeze(dim=3)
        src = src.permute(0, 2, 1)                                       # 卷积层用于缩减数据尺寸

        y1 = self.pos_encoder(src)
        y1 = self.transformer_encoder(y1)
        y2 = y1 + self.mid_layer(src)             # 残差连接
        output = self.relu(y2)

        # y1, _ = self.GRU(src)
        # y1 = self.linear1(y1)
        # y2 = y1 + self.mid_layer(src)           # 残差连接
        # output = self.relu(y2)

        output = self.conv1d(output).squeeze(dim=1)
        output = self.decoder(output)
        return output


wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))                 # 图像的标例
wind.line([[0., 0.]], [0.], win='test0', opts=dict(title='MAE', legend=['train_MAE', 'test_MAE']))
wind.line([[0., 0.]], [0.], win='test1', opts=dict(title='RMSE', legend=['train_RMSE', 'test_RMSE']))

parser = get_parser()
args = parser.parse_args()

net = CNN2d_TransAm(feature_size=args.num_hiddens, num_layers=args.num_layers, dropout=args.dropout)
device = lib.try_gpu()
dataSet1 = TensorDataset(train_x, train_y)
train_iter = DataLoader(dataSet1, batch_size=args.batch_size, shuffle=True)

dataSet2 = TensorDataset(test_x, test_y)
test_iter = DataLoader(dataSet2, batch_size=args.batch_size, shuffle=True)

if args.need_train:
    lib.train_ch6(net, train_iter, test_iter, args.num_epochs, lr=args.lr, device=device, wind=wind, need_variation_loss=False, gamma=0.01)
    torch.save(net.state_dict(), 'model_save/%s' % net.model_type)
else:
    net.load_state_dict(torch.load('model_save/%s' % net.model_type))

scatter_X, scatter_Y = [], []
fig, axes = plt.subplots(1, len(degs), figsize=(8, 3.5))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.05)
axes = axes.flat
net.eval()
for k, deg in enumerate(degs):
    X, Y, Time = [], [], []
    for cycle in test_cycles:
        cur = data[deg][cycle]
        start, end = 0, len(cur) - 1
        while cur[start][4] == cur[start + 1][4]: start += 1
        while cur[end][4] == cur[end - 1][4]: end -= 1
        cur = cur[start:end + 1]
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
    net.to(device)
    with torch.no_grad():
        for x in train_iter:
            y_pred = net(x[0])
            pred_y.append(np.array(y_pred[:, :50].detach().cpu()).reshape(-1))
    pred_y = np.concatenate(pred_y)
    scatter_Y.extend(list(pred_y))
    Ttime = np.arange(0, len(Y)) / 10
    axes[k].plot(Ttime, Y, label='真实值', color='r', linewidth=1)
    axes[k].plot(Ttime, pred_y, label='预测值', color='b', linewidth=1)
    axes[k].set_title('%s℃' % deg)
    axes[k].grid(ls='-.')
    axes[k].set_xlabel('时间/s')
    axes[k].set_ylabel('SOC')
    axes[k].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[k].set_xticks(list([i * 1000 for i in range(int(Ttime[-1] // 1000 + 1))]))
    axes[k].legend()
    print('在%s温度下的测试集评估指标:' % deg, end='  ')
    print('MAE:%0.3f, RMSE:%0.3f' % (np.sum(np.abs(pred_y - Y)) / Y.size, np.sqrt(np.sum(np.square(pred_y - Y)) / Y.size)))
plt.savefig('res_fig/%s/在不同温度的测试集下的预测结果.png' % net.model_type, dpi=1000, bbox_inches='tight')              # 指定分辨率

plt.figure(figsize=(5, 4))                                # 画散点图
scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)     # 转化为Numpy格式
plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
print('整体的评估指标:MAE:%0.3f, RMSE:%0.3f'%(np.sum(np.abs(scatter_Y - scatter_X)) / scatter_X.size, np.sqrt(np.sum(np.square(scatter_Y - scatter_X)) / scatter_X.size)))
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], '--', zorder=1)
plt.xlabel('真实SOC')
plt.ylabel('预测SOC')
plt.colorbar()
plt.savefig('res_fig/%s/在不同温度测试集下的整体预测偏差.png' % net.model_type, dpi=1000, bbox_inches='tight')              # 指定分辨率
plt.show()