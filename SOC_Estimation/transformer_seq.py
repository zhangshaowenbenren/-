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

with open('preprocessed_data2.pkl', 'rb') as rb:
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
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=233, metavar='S', help='random seed (default: 233)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--output_size', type=int, default=100)      # 预测的关键节点个数
    parser.add_argument('--num_hiddens', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.2)
    return parser

#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout=0, max_len=1200):
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


class TransAm(nn.Module):
    def __init__(self, feature_size=128, inputChannels=3, num_layers=2, dropout=0.2):
        # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Sequential(nn.Linear(1, feature_size), nn.ReLU())
        self.block0 = nn.Sequential(nn.Conv1d(inputChannels, feature_size, kernel_size=5, stride=5, padding=2),
                                    nn.BatchNorm1d(feature_size), nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2))
        self.pos_encoder = PositionalEncoding(feature_size)            # 位置编码前要做归一化，否则捕获不到位置信息
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout, dim_feedforward=256, batch_first=True)     # 多头注意力
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.decoder = nn.Linear(feature_size * 13, 100)  # 这里用全连接层代替了decoder， 其实也可以加一下Transformer的decoder试一下效果
        self.transformer = nn.Transformer(d_model=128, nhead=2, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(feature_size, 1)


    def forward(self, src, tgt):
        tgt = self.embedding(tgt.unsqueeze(2))
        src = src.permute(0, 2, 1)
        src = self.block0(src)
        src = src.permute(0, 2, 1)                                       # 卷积层用于缩减数据尺寸
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        output = self.linear(output)
        return output


wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))                 # 图像的标例
wind.line([[0., 0.]], [0.], win='test0', opts=dict(title='MAE', legend=['train_MAE', 'test_MAE']))
wind.line([[0., 0.]], [0.], win='test1', opts=dict(title='RMSE', legend=['train_RMSE', 'test_RMSE']))

parser = get_parser()
args = parser.parse_args()

net = TransAm(feature_size=args.num_hiddens, inputChannels=args.input_channel, num_layers=args.num_layers, dropout=args.dropout)
device = lib.try_gpu()
dataSet1 = TensorDataset(train_x, train_y)
train_iter = DataLoader(dataSet1, batch_size=args.batch_size, shuffle=True)

dataSet2 = TensorDataset(test_x, test_y)
test_iter = DataLoader(dataSet2, batch_size=args.batch_size, shuffle=True)

need_train = True
if need_train:
    lib.train_transformer(net, train_iter, test_iter, args.num_epochs, lr=args.lr, device=device, wind=wind)
    torch.save(net.state_dict(), 'model_save/%s' % 'transformer_seq')
else:
    net.load_state_dict(torch.load('model_save/%s' % 'transformer_seq'))

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
            y_pred = lib.predict_transformer(net, x[0], device)
            y_pred = y_pred.squeeze()
            pred_y.append(np.array(y_pred[:, :50].detach().cpu()).reshape(-1))
    pred_y = np.concatenate(pred_y)
    scatter_Y.extend(list(pred_y))
    Ttime = np.arange(0, len(Y)) / 10
    axes[k].plot(Ttime, Y, label='true', color='r')
    axes[k].plot(Ttime, pred_y, label='predict', color='b')
    axes[k].set_title('%s℃'%deg)
    print('在%s温度下的测试集评估指标:' % deg, end='  ')
    print('MAE:%0.3f, RMSE:%0.3f' % (np.sum(np.abs(pred_y - Y)) / Y.size, np.sqrt(np.sum(np.square(pred_y - Y)) / Y.size)))
plt.savefig('res_fig/transformer_seq/在不同温度的测试集下的预测结果.png', dpi=800, bbox_inches='tight')              # 指定分辨率

plt.figure()                                # 画散点图
scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)     # 转化为Numpy格式
plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
print('整体的评估指标:MAE:%0.3f, RMSE:%0.3f'%(np.sum(np.abs(scatter_Y - scatter_X)) / scatter_X.size, np.sqrt(np.sum(np.square(scatter_Y - scatter_X)) / scatter_X.size)))
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.plot([0, 1], [0, 1], '--', zorder=1)
plt.colorbar()
plt.savefig('res_fig/transformer_seq/在不同温度测试集下的整体预测偏差.png', dpi=800, bbox_inches='tight')              # 指定分辨率
plt.show()