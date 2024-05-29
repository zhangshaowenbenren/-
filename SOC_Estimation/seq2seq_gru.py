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
from d2l import torch as d2l

with open('preprocessed_data.pkl', 'rb') as rb:
    data = pickle.load(rb)

degs = [25, 10, 0]
train_cycles = ['Cycle_1', 'Cycle_2', 'Cycle_3', 'Cycle_4', 'HWFET', 'LA92', 'NN', 'UDDS']
test_cycles = ['US06']

sd_steps, label_steps = 1000, 100
train_x, train_y = lib.get_processed_data(data, degs, train_cycles, sd_steps, label_steps=label_steps)
test_x, test_y = lib.get_processed_data(data, degs, test_cycles, sd_steps, label_steps=label_steps)

train_x, test_x, max0, min0 = lib.normalize(train_x, test_x)

def get_parser():
    parser = argparse.ArgumentParser(description='Pytorch SOC Estimation')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=233, metavar='S', help='random seed (default: 233)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_channel', type=int, default=1)      # 用于预测的循环数
    parser.add_argument('--output_size', type=int, default=100)      # 预测的关键节点个数
    parser.add_argument('--num_hiddens', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--need_train', type=bool, default=False)
    return parser

parser = get_parser()
args = parser.parse_args()

class Encoder(nn.Module):
    # 编码器-解码器架构的基本编码接口
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    # 编码器-解码器架构
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.model_type = 'CGRU_seq2seq'
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

#@save
class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, inputChannels=1, hidden_size=64, num_layers=1, dropout=0., **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
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
            ('maxplool3', nn.MaxPool2d(kernel_size=(2, 1)))
        ]))

        self.GRU = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)


    def forward(self, x, *args):
        # x的shape：批量 * 通道维数 * 长 * 宽
        x = x.unsqueeze(1)
        y = self.block0(x)
        y = torch.squeeze(y)
        y = y.permute(0, 2, 1)
        output, state = self.GRU(y)
        return output, state


class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self,  num_hiddens, num_layers=1, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.GRU = nn.GRU(1 + num_hiddens, num_hiddens, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dense = nn.Linear(num_hiddens, 1)

    def init_state(self, enc_outputs, *args):
        # return enc_outputs[1]
        return (enc_outputs[1], enc_outputs[1][-1])                # 最后一步的隐状态和解码器中的上下文变量

    def forward(self, X, state):
        # 'X'的形状：(batch_size,num_steps,label_size)
        # state是init_state()的返回结果
        # 广播context，使其具有与X相同的num_steps
        X = X.unsqueeze(2)
        context = state[-1].unsqueeze(1).repeat(1, X.shape[1], 1)               # state[-1]表示与解码器输入进行concat的上下文变量;在步长维度进行复制
        state, encode = state[0], state[1]
        X_and_context = torch.cat((X, context), 2)
        output, state = self.GRU(X_and_context, state)
        output = self.dense(output)
        # output的形状:(batch_size, num_steps, label_size)
        # state的形状:(num_layers, batch_size, num_hiddens)
        return output, (state, encode)                             # 每个时间布更新的隐状态以及恒定不变的上下文变量


#@save
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self,  num_hiddens, num_layers=1, dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.GRU = nn.GRU(1 + num_hiddens, num_hiddens, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads=2, dropout=dropout, bias=False)
        self.dense = nn.Linear(num_hiddens, 1)


    def init_state(self, enc_outputs, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs, hidden_state)


    def forward(self, X, state):
        # 'X'的形状：(batch_size,num_steps,label_size)
        # state是init_state()的返回结果
        X = X.unsqueeze(2).permute(1, 0, 2)
        enc_outputs, hidden_state = state
        outputs = []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)     # query的形状为(batch_size,1,num_hiddens)，上一时间步最后层的输出
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, valid_lens=None)
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.GRU(x, hidden_state)
            outputs.append(out)

        outputs = self.dense(torch.cat(outputs, dim=1))     # shape：(批量， 步长， 标签)
        return outputs, [enc_outputs, hidden_state]


wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))                 # 图像的标例
wind.line([[0., 0.]], [0.], win='test0', opts=dict(title='MAE', legend=['train_MAE', 'test_MAE']))
wind.line([[0., 0.]], [0.], win='test1', opts=dict(title='RMSE', legend=['train_RMSE', 'test_RMSE']))

encoder = Seq2SeqEncoder(inputChannels=args.input_channel, hidden_size=args.num_hiddens, num_layers=args.num_layers, dropout=args.dropout)
decoder = Seq2SeqAttentionDecoder(num_hiddens=args.num_hiddens, num_layers=args.num_layers, dropout=args.dropout)

net = EncoderDecoder(encoder, decoder)
device = lib.try_gpu()
dataSet1 = TensorDataset(train_x, train_y)
train_iter = DataLoader(dataSet1, batch_size=args.batch_size, shuffle=True)

dataSet2 = TensorDataset(test_x, test_y)
test_iter = DataLoader(dataSet2, batch_size=args.batch_size, shuffle=True)


if args.need_train:
    lib.train_seq2seq(net, train_iter, test_iter, lr=args.lr, num_epochs=args.num_epochs, device=device, wind=wind)
    torch.save(net.state_dict(), 'model_save/%s' % net.model_type)
else:
    net.load_state_dict(torch.load('model_save/%s' % net.model_type))

scatter_X, scatter_Y = [], []
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flat
net.eval()
net.to(device)
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
            y_pred, _ = lib.predict_seq2seq(net, x[0], device)
            pred_y.append(np.array(y_pred[:, :50].detach().cpu()).reshape(-1))
    pred_y = np.concatenate(pred_y)
    scatter_Y.extend(list(pred_y))
    Ttime = np.arange(0, len(Y)) / 10
    axes[k].plot(Ttime, Y, label='true', color='r', linewidth=0.5)
    axes[k].plot(Ttime, pred_y, label='predict', color='b', linewidth=0.5)
    axes[k].set_title('%s℃'%deg)
    print('在%s温度下的测试集评估指标:' % deg, end='  ')
    print('MAE:%0.3f, RMSE:%0.3f' % (
    np.sum(np.abs(pred_y - Y)) / Y.size, np.sqrt(np.sum(np.square(pred_y - Y)) / Y.size)))
plt.savefig('res_fig/%s/在不同温度的测试集下的预测结果.png' % net.model_type, dpi=1000, bbox_inches='tight')              # 指定分辨率

plt.figure()                                                                                                        # 画散点图
scatter_X, scatter_Y = np.array(scatter_X), np.array(scatter_Y)                                                     # 转化为Numpy格式
plt.scatter(scatter_X, scatter_Y, c=abs(scatter_X - scatter_Y), cmap="seismic", zorder=2, s=3)
print('整体的评估指标:MAE:%0.3f, RMSE:%0.3f'%(np.sum(np.abs(scatter_Y - scatter_X)) / scatter_X.size, np.sqrt(np.sum(np.square(scatter_Y - scatter_X)) / scatter_X.size)))
plt.colorbar()
plt.savefig('res_fig/%s/在不同温度测试集下的整体预测偏差.png' % net.model_type, dpi=1000, bbox_inches='tight')              # 指定分辨率
plt.show()