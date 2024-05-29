import math
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

def RMSE(hat, value):
    res = torch.square(hat - value)
    return torch.sqrt(torch.sum(res) / value.numel())


# 计算训练样本上的平均百分误差
def MAPE(hat, value):
    temp = torch.abs((hat - value) / value)
    return torch.sum(temp) / value.numel()


# 计算训练样本上的平均绝对值误差
def MAE(hat, value):
    temp = torch.abs(hat - value)
    return torch.sum(temp) / value.numel()


# 计算相关系数
def R_square(hat, value):
    hat, value = np.array(hat), np.array(value)
    m = np.mean(value)
    a1 = np.sum((hat - value) * (hat - value))
    a2 = np.sum((value - m) * (value - m))
    return 1 - a1 / a2


def get_processed_data(data, degs, cycles, sd_steps, label_steps=100):
    stpes = {25:50, 10:50, 0:50, -10:50, -20:50}
    X, Y = [], []
    for deg in degs:
        k = 0
        for cycle in cycles:
            cur = data[deg][cycle]
            start, end = 0, len(cur) - 1
            while cur[start][4] == cur[start + 1][4]:start += 1
            while cur[end][4] == cur[end - 1][4]:end -= 1
            cur = cur[start:end + 1]
            for i in range(sd_steps, len(cur) - label_steps, stpes[deg]):
                k += 1
                x, y = torch.tensor(cur[i - sd_steps:i, 1:4], dtype=torch.float32), torch.tensor(cur[i:i + label_steps:, 4], dtype=torch.float32)
                X.append(x)
                Y.append(y)
            x, y = torch.tensor(cur[-label_steps - sd_steps:-label_steps, 1:4], dtype=torch.float32), torch.tensor(cur[-label_steps:, 4], dtype=torch.float32)
            X.append(x)
            Y.append(y)
        # print('%s温度下下的样本数量为%s'%(deg, k))
    return torch.stack(X), torch.stack(Y)


# 一个累加器
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_metric(net, data_iter, device):
    net.eval()                       # Set the model to evaluation mode
    metric = Accumulator(3)              # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_pred = net(X)
            metric.add(MAE(y_pred, y), RMSE(y_pred, y), 1)
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device, wind=None, need_variation_loss=False, gamma=0.5):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # nn.init.kaiming_uniform_(m.weight)            # ReLU激活函数更适合
            nn.init.xavier_uniform_(m.weight)               # sign和tanh更适合

    net.apply(init_weights)
    print('training on=======================', device)
    net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 10, gamma=0.7)

    loss = nn.MSELoss()
    # loss = nn.L1Loss()
    for epoch in range(1, num_epochs + 1):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(4)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            if need_variation_loss:
                tmp = y_hat[:, 1:] - y_hat[:, :-1]
                # l += gamma * torch.mean(tmp[tmp > 0])               # 错位相减取绝对值的平均值,约束输出值的剧烈变化程度
                l += gamma * torch.mean(torch.abs(tmp))  # 错位相减取绝对值的平均值,约束输出值的剧烈变化程度
            l.backward()
            # grad_clipping(net, 10)                                  # 梯度裁剪,防止梯度爆炸
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=5, foreach=None)
            optimizer.step()
            with torch.no_grad():
                metric.add(l, MAE(y_hat, y), RMSE(y_hat, y), 1)
        scheduler.step()
        if wind and epoch % 5 == 0:
            test_metric = evaluate_metric(net, test_iter, device=device)
            wind.line([metric[0] / metric[3]], [epoch - 2], win='train', update='append')
            wind.line([[metric[1] / metric[3], test_metric[0]]], [epoch - 2], win='test0', update='append')
            wind.line([[metric[2] / metric[3], test_metric[1]]], [epoch - 2], win='test1', update='append')
            print('epoch {}'.format(epoch), f'loss {metric[0] / metric[3]:.3f}, train MAE {metric[1] / metric[3]:.3f}, 'f'test MAE {test_metric[0]:.3f}')


def train_snapshot(net, train_iter, test_iter, num_epochs, lr, device, path, cycles, wind=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # nn.init.kaiming_uniform_(m.weight)            # ReLU激活函数更适合
            nn.init.xavier_uniform_(m.weight)               # sign和tanh更适合

    net.apply(init_weights)
    print('training on=======================', device)
    net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycles)      # T_max表示学习率变化的半周期

    loss = nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(4)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            # grad_clipping(net, 10)                                  # 梯度裁剪,防止梯度爆炸
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=5, foreach=None)
            optimizer.step()
            with torch.no_grad():
                metric.add(l, MAE(y_hat, y), RMSE(y_hat, y), 1)
        scheduler.step()
        # if wind:wind.line([metric[0] / metric[3]], [epoch - 2], win='train', update='append')
        if wind and epoch % 1 == 0:
            test_metric = evaluate_metric(net, test_iter, device=device)
            wind.line([metric[0] / metric[3]], [epoch - 1], win='train', update='append')
            wind.line([[metric[1] / metric[3], test_metric[0]]], [epoch - 1], win='test0', update='append')
            wind.line([[metric[2] / metric[3], test_metric[1]]], [epoch - 1], win='test1', update='append')
            wind.line([scheduler.get_last_lr()[0]], [epoch - 1], win='lr', update='append')
            print('epoch {}'.format(epoch), f'loss {metric[0] / metric[3]:.3f}, train MAE {metric[1] / metric[3]:.3f}, 'f'test MAE {test_metric[0]:.3f}')
        if epoch % cycles == 0:
            torch.save(net.state_dict(), path + str(epoch // cycles - 1))

# 最大最小归一化
def normalize(train_data, test_data):
    max0, _ = torch.max(train_data, dim=0, keepdim=True)
    max0, _ = torch.max(max0, dim=1, keepdim=True)

    min0, _ = torch.min(train_data, dim=0, keepdim=True)
    min0, _ = torch.min(min0, dim=1, keepdim=True)
    return (train_data - min0) / (max0 - min0), (test_data - min0) / (max0 - min0), max0, min0


def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def grad_clipping(net, theta):
    # 梯度裁剪
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


#@save
def train_seq2seq(net, train_iter, test_iter, lr, num_epochs, device, wind=None):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 10, gamma=0.8)
    loss = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        net.train()
        # loss, MAE, RMSE, 次数
        metric = Accumulator(4)                         # 训练损失总和，词元数量
        for batch in train_iter:
            optimizer.zero_grad()
            X, y = [x.to(device) for x in batch]
            bos = torch.tensor([0] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y[:, :-1]], 1)  # 强制教学
            y_hat, _ = net(X, dec_input)
            y_hat = y_hat.squeeze()
            l = loss(y_hat, y)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            grad_clipping(net, 5)
            optimizer.step()
            with torch.no_grad():
                metric.add(l, MAE(y_hat, y), RMSE(y_hat, y), 1)
        scheduler.step()
        if wind and epoch % 2 == 0:
            test_metric = evaluate_metric2(net, test_iter, device=device)
            wind.line([metric[0] / metric[3]], [epoch - 2], win='train', update='append')
            wind.line([[metric[1] / metric[3], test_metric[0]]], [epoch - 2], win='test0', update='append')
            wind.line([[metric[2] / metric[3], test_metric[1]]], [epoch - 2], win='test1', update='append')
            print('epoch {}'.format(epoch),
                  f'loss {metric[0] / metric[3]:.3f}, train MAE {metric[1] / metric[3]:.3f}, 'f'test MAE {test_metric[0]:.3f}')


def evaluate_metric2(net, data_iter, device):
    net.eval()
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_pred, _ = predict_seq2seq(net, X, device)
            metric.add(MAE(y_pred, y), RMSE(y_pred, y), 1)
    return metric[0] / metric[2], metric[1] / metric[2]


def predict_seq2seq(net, X, device, num_steps=100, save_attention_weights=False):
    net.eval()                                                                   # Set the model to evaluation mode
    # 添加批量轴
    enc_outputs = net.encoder(X)
    dec_state = net.decoder.init_state(enc_outputs)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([0] * X.shape[0], dtype=torch.float32, device=device), dim=1)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)           # net.decoder的输入为二维：批次 * 步长，特征维度默认为1
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.reshape(-1, 1)
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        output_seq.append(dec_X.squeeze())
    return torch.stack(output_seq).T, attention_weight_seq


def train_transformer(net, train_iter, test_iter, num_epochs, lr, device, wind=None):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 10, gamma=0.8)
    loss = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        net.train()
        # loss, MAE, RMSE, 次数
        metric = Accumulator(4)                             # 训练损失总和，词元数量
        for batch in train_iter:
            optimizer.zero_grad()
            X, y = [x.to(device) for x in batch]
            bos = torch.tensor([0] * y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, y[:, :-1]], 1)      # 强制教学，同时输入编码序列和解码序列
            y_hat = net(X, dec_input)
            y_hat = y_hat.squeeze()
            l = loss(y_hat, y)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            grad_clipping(net, 5)
            optimizer.step()
            with torch.no_grad():
                metric.add(l, MAE(y_hat, y), RMSE(y_hat, y), 1)
        scheduler.step()
        if wind and epoch % 2 == 0:
            test_metric = evaluate_transformer(net, test_iter, device=device)
            wind.line([metric[0] / metric[3]], [epoch - 2], win='train', update='append')
            wind.line([[metric[1] / metric[3], test_metric[0]]], [epoch - 2], win='test0', update='append')
            wind.line([[metric[2] / metric[3], test_metric[1]]], [epoch - 2], win='test1', update='append')
            print('epoch {}'.format(epoch), f'loss {metric[0] / metric[3]:.3f}, train MAE {metric[1] / metric[3]:.3f}, 'f'test MAE {test_metric[0]:.3f}')


def evaluate_transformer(net, data_iter, device):
    net.eval()
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_pred = predict_transformer(net, X, device)
            metric.add(MAE(y_pred, y), RMSE(y_pred, y), 1)
    return metric[0] / metric[2], metric[1] / metric[2]


def predict_transformer(net, src, device, num_steps=100):
    net.eval()                                                                   # Set the model to evaluation mode
    # 添加批量轴
    tgt = torch.unsqueeze(torch.tensor([0] * src.shape[0], dtype=torch.float32, device=device), dim=1)
    output_seq = []
    for _ in range(num_steps):
        Y = net(src, tgt)
        tgt = torch.concat([tgt, Y[:, -1]], dim=1)
        output_seq.append(Y[:, -1])
    return torch.stack(output_seq).T