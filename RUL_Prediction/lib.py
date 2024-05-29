import math

import pandas as pd
from scipy import interpolate
import numpy as np
import torch
from torch import nn
import pickle
import gc
import os
import random
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,  TensorDataset
from scipy.optimize import curve_fit

# 加载数据
def load_Data(cross_val=False, is_random=False):
    '''
    :param cross_val:如果是交叉验证，则返回训练集和测试机；如果不是，则返回训练集、验证集和测试集
    :param is_random: 如果是随机的，则随机划分电池；否则使用手动划分的电池数据集
    :return:
    '''
    data_name = 'raw_data/batch1.pkl'
    batch1 = pickle.load(open(data_name, 'rb'))
    # remove batteries that do not reach 80% capacity
    temp_keys = ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']
    for key in temp_keys:
        if key in batch1.keys():
            del batch1[key]
            gc.collect()
    data_name = 'raw_data/batch2.pkl'
    batch2 = pickle.load(open(data_name, 'rb'))
    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2 and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]
    # 第一批次中有5个电池
    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

    temp_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    for key in temp_keys:
        if key in batch2.keys():
            del batch2[key]
            gc.collect()

    data_name = 'raw_data/batch3.pkl'
    with open(data_name, 'rb') as fp:
        batch3 = pickle.load(fp)
    # batch3 = pickle.load(open(data_name, 'rb'))

    # remove noisy channels from batch3
    temp_keys = ['b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']
    for key in temp_keys:
        if key in batch3.keys():
            del batch3[key]
            gc.collect()
    del batch1['b1c0']
    del batch1['b1c18']
    del batch2['b2c12']
    del batch2['b2c44']
    lib_dict = {}
    temp = ['Qd', 'dQdV', 'Tdlin']                                # 需要删除的其他特征
    i = 0
    for batch in [batch1, batch2, batch3]:
        for key, value in batch.items():
            # 缩减数据尺寸，只留需要的
            lib_dict[key] = i
            i += 1
            for k, v in batch[key]['cycles'].items():
                for t in temp:
                    del batch[key]['cycles'][k][t]

    gc.collect()
    # ** 是解析字典用的
    # bat_dict = {**batch1, **batch2, **batch3}
    # dict_key_ls = list(bat_dict.keys())
    # random.shuffle(dict_key_ls)                                    # 打乱电池的排列顺序
    # new_dic = {}
    # for key in dict_key_ls:
    #     new_dic[key] = bat_dict.get(key)
    # keys = list(new_dic.keys())

    train_set, val_set, test_set = {}, {}, {}
    A = pd.read_csv('电池寿命信息.csv')
    test_index = ['b1c16', 'b1c26', 'b1c27', 'b1c32', 'b1c33', 'b1c35', 'b1c36', 'b1c43',
                  'b2c5', 'b2c12', 'b2c11', 'b2c17', 'b2c27', 'b2c30', 'b2c32', 'b2c43', 'b2c45',
                  'b3c15', 'b3c13', 'b3c19', 'b3c21', 'b3c29', 'b3c30', 'b1c34', 'b3c35', 'b3c25']
    # test_index = list(A[A['is_train'] == 0]['lib'])


    if cross_val:
        for batch0 in [batch1, batch2, batch3]:
            dict_key_ls = list(batch0.keys())
            random.shuffle(dict_key_ls)                                    # 打乱电池的排列顺序
            batch = {}
            for key in dict_key_ls:
                batch[key] = batch0.get(key)
            keys = list(batch.keys())
            for i in range(0, len(keys)):
                if is_random:
                    if i < len(keys) * 0.8:
                        train_set[keys[i]] = batch[keys[i]]
                    else:
                        test_set[keys[i]] = batch[keys[i]]
                else:
                    if keys[i] in test_index:
                        test_set[keys[i]] = batch[keys[i]]
                    else:
                        train_set[keys[i]] = batch[keys[i]]
                del batch[keys[i]]
                gc.collect()
        return train_set, test_set, lib_dict
    else:
        # 训练集、验证集和测试集的比例为 7:1:2（电池维度）
        for batch0 in [batch1, batch2, batch3]:
            dict_key_ls = list(batch0.keys())
            random.shuffle(dict_key_ls)                                    # 打乱电池的排列顺序
            batch = {}
            for key in dict_key_ls:
                batch[key] = batch0.get(key)
            keys = list(batch.keys())
            count = 0
            for i in range(0, len(keys)):
                if is_random:
                    if i < len(keys) * 0.7:
                        train_set[keys[i]] = batch[keys[i]]
                    elif i > len(keys) * 0.8:
                        test_set[keys[i]] = batch[keys[i]]
                    else:
                        val_set[keys[i]] = batch[keys[i]]
                else:
                    if keys[i] in test_index:
                        test_set[keys[i]] = batch[keys[i]]
                    else:
                        count += 1
                        if count < len(keys) * 0.7:
                            train_set[keys[i]] = batch[keys[i]]
                        else:
                            val_set[keys[i]] = batch[keys[i]]
                del batch[keys[i]]
                gc.collect()
        return train_set, val_set, test_set, lib_dict



# 特征数据重采样，按step的间隔分别进行插值计算
def getNew(y1, y2, y3, x, num_samples, lib_num, lst=None):          # y1表示电压数据，y2表示电流数据, y3表示温度数据，x表示时间，
    if not lst:
        deleteList = []
        for i in range(1, len(x)):
            if x[i] - x[i - 1] < 0.00015:
                deleteList.append(i)
        y1, y2, y3, x = np.delete(y1, deleteList), np.delete(y2, deleteList), np.delete(y3, deleteList), np.delete(x, deleteList)
        i0, i1 = 0, 0                        # 分别为恒流（4C）放电开始下标和放电结束下标
        i = 0
        while not (y2[i] > 0 and y2[i] - y2[i-1] > 0.01 and y2[i+2] > 0.1 and y2[i+4] > 0.2):
            i += 1
        i0 = i
        for i in range(1, len(y2)):
            if i1 == 0 and y2[i] < 0 and y2[i] - y2[i-1] < -0.01 and y2[i+2] < -0.2 and y2[i+4] < -0.2:
                i1 = i
            # if i1 != 0 and y2[i] < 0 and y2[i] - y2[i-1] > 0.5 and y2[i+2] - y2[i-1] > 0.5 and y2[i+4] > -3:
            #     i2 = i
                break
        if i1 == 0:
            print(i0, i1)
    else:
        i0, i1 = lst
    y1, y2, y3, x = y1[i0:i1], y2[i0:i1], y3[i0:i1], x[i0:i1]
    delete_list1, delete_list2, delete_list3 = RemoveIndex(y1, sd_size=20), RemoveIndex(y2, sd_size=20), RemoveIndex(y3, sd_size=20)
    delete_list = [*delete_list1, *delete_list2, *delete_list3]
    y = np.vstack([y1, y2, y3])
    y = np.delete(y, delete_list, axis=1)
    x = np.delete(x, delete_list)
    # spline = CubicSpline(x, y, axis=1)                    # y1、x的样条曲线
    spline = interp1d(x, y, kind='linear', axis=-1)
    t1 = np.linspace(x[0], x[-1], num_samples)            # 插值后的时间点
    if lib_num == -1:
        return spline(t1)[0].tolist(), spline(t1)[1].tolist(), spline(t1)[2].tolist(), t1.tolist()
    else:
        return [lib_num] + spline(t1)[0].tolist(), [lib_num] + spline(t1)[1].tolist(), [lib_num] + spline(t1)[2].tolist(), t1.tolist()


def getNew2(y1, y2, y3, x, num_samples):                # y1表示电压数据，y2表示电流数据, y3表示温度数据，x表示时间，
    i0, i1 = 0, 0                        # 分别为恒流（4C）放电开始下标和放电结束下标
    i = 0
    while not (y2[i] > 0 and y2[i] - y2[i-1] > 0.01 and y2[i+2] > 0.1 and y2[i+4] > 0.2):
        i += 1
    i0 = i
    for i in range(1, len(y2)):
        if i1 == 0 and y2[i] < 0 and y2[i] - y2[i-1] < -0.01 and y2[i+2] < -0.2 and y2[i+4] < -0.2:
            i1 = i
        # if i1 != 0 and y2[i] < 0 and y2[i] - y2[i-1] > 0.5 and y2[i+2] - y2[i-1] > 0.5 and y2[i+4] > -3:
        #     i2 = i
            break
    if i1 == 0:
        print(i0, i1)
    y1, y2, y3, x = y1[i0:i1], y2[i0:i1], y3[i0:i1], x[i0:i1]
    deleteList = []
    for i in range(1, len(x)):
        if x[i] - x[i - 1] < 0.00015:
            deleteList.append(i)
    y1, y2, y3, x = np.delete(y1, deleteList), np.delete(y2, deleteList), np.delete(y3, deleteList), np.delete(x, deleteList)
    delete_list1, delete_list2, delete_list3 = RemoveIndex(y1, sd_size=20), RemoveIndex(y2, sd_size=20), RemoveIndex(y3, sd_size=20)
    delete_list = [*delete_list1, *delete_list2, *delete_list3]
    y = np.vstack([y1, y2, y3])
    y = np.delete(y, delete_list, axis=1)
    x = np.delete(x, delete_list)
    # spline = CubicSpline(x, y, axis=1)                    # y1、x的样条曲线
    spline = interp1d(x, y, kind='linear', axis=-1)
    t1 = np.linspace(x[0], x[-1], num_samples)              # 插值后的时间点
    return spline(t1)[0].tolist(), spline(t1)[1].tolist(), spline(t1)[2].tolist(), (i0, i1)


# 3sigma法则去除离群点， 删除容量数据异常对应的行
def RemoveIndex(data, sd_size=40):
    delete_index_list = []
    for i in range(0, len(data), sd_size):
        a = data[i:min(i + sd_size, len(data))]
        mean0, std0 = np.mean(a), np.std(a)
        delete_index = np.array([j for j, x in enumerate(a) if x > mean0 + 2 * std0 or x < mean0 - 2 * std0]) + i      # 3sigma法则
        delete_index_list.extend(delete_index)
    return delete_index_list


def get_start_end_feature(V, I, T, t):
    deleteList = []
    for i in range(1, len(t)):
        if t[i] - t[i - 1] < 0.00015:
            deleteList.append(i)
    V, I, T, t = np.delete(V, deleteList), np.delete(I, deleteList), np.delete(T, deleteList), np.delete(t, deleteList)
    i0, i1 = 0, 0  # 分别为充电开始下标和恒流（4C）放电开始下标
    i = 0
    while not (I[i] > 0 and I[i] - I[i - 1] > 0.01 and I[i + 2] > 0.1 and I[i + 4] > 0.2):
        i += 1
    i0 = i
    for i in range(1, len(I)):
        if i1 != 0:
            break
        if I[i] < 0 and I[i] - I[i - 1] < -0.01 and I[i + 2] < -0.2 and I[i + 4] < -0.4:
            i1 = i
    return V[i0:i1], I[i0:i1], T[i0:i1], t[i0:i1]


# Define bacon-watts formula
def bacon_watts_model(x, alpha0, alpha1, alpha2, x1):
    ''' Equation of bw_model'''
    return alpha0 + alpha1 * (x - x1) + alpha2 * (x - x1) * np.tanh((x - x1) / 1e-8)


def fit_bacon_watts(y, p0):
    ''' Function to fit Bacon-Watts model to identify knee-point in capacity fade data

    Args:
    - capacity fade data (list): cycle-to-cycle evolution of Qd capacity
    - p0 (list): initial parameter values for Bacon-Watts model

    Returns:
    - popt (int): fitted parameters
    - confint (list): 95% confidence interval for fitted knee-point
    '''

    # Define array of cycles
    x = np.arange(len(y)) + 1

    # Fit bacon-watts
    popt, pcov = curve_fit(bacon_watts_model, x, y, p0=p0)

    confint = [popt[3] - 1.96 * np.diag(pcov)[3],
               popt[3] + 1.96 * np.diag(pcov)[3]]
    return popt, confint


def get_attention_values(value, bias):
    '''
    :param value:电池的特征数据，数据类型为字典
    :return: 返回该电池键值对特征数据
    '''
    res = []
    res.append(value['cycle_life'])                       # 截止寿命
    res.extend(value['charge_policy'])                    # 充电电流数据

    QD = value['summary']['QD']  # 放电容量
    p0 = [1, -1e-4, -1e-4, len(QD) * .7]  # 初始参数alpha0, alpha1, alpha2, x1
    popt_kpoint, _ = fit_bacon_watts(QD, p0)
    knee_point = round(popt_kpoint[3])                    # 拐点对应的循环数

    I2, I100, Iknee = value['cycles']['2']['I'], value['cycles']['100']['I'], value['cycles'][str(knee_point)]['I']
    t2, t100, tknee = value['cycles']['2']['t'], value['cycles']['100']['t'], value['cycles'][str(knee_point)]['t']
    V2, V100, Vknee = value['cycles']['2']['V'], value['cycles']['100']['V'], value['cycles'][str(knee_point)]['V']
    T2, T100, Tknee = value['cycles']['2']['T'], value['cycles']['100']['T'], value['cycles'][str(knee_point)]['T']
    V2, I2, T2, t2 = get_start_end_feature(V2, I2, T2, t2)                                     # 第2循环的特征曲线
    V100, I100, T100, t100 = get_start_end_feature(V100, I100, T100, t100)                     # 第100循环的特征曲线
    Vknee, Iknee, Tknee, tknee = get_start_end_feature(Vknee, Iknee, Tknee, tknee)             # 第100循环的特征曲线

    avg_I2, avg_I100, avg_Iknee = np.mean(I2), np.mean(I100), np.mean(Iknee)
    avg_T2, avg_T100, avg_Tknee = np.mean(T2), np.mean(T100), np.mean(Tknee)

    Q_V2, Q_V100, Q_Vknee = value['cycles']['2']['Qdlin'], value['cycles']['100']['Qdlin'], value['cycles'][str(knee_point)]['Qdlin']         # 放电阶段Q(V)曲线
    T_V2, T_V100, T_Vknee = value['cycles']['2']['Tdlin'], value['cycles']['100']['Tdlin'], value['cycles'][str(knee_point)]['Tdlin']         # 放电阶段T(V)曲线
    delta_QV, delta_TV = Q_V100 - Q_V2, T_V100 - T_V2
    delta_QV_mean, delta_TV_mean = np.mean(delta_QV), np.mean(delta_TV)                        # Q(v)的第100循环与第2循环差的均值
    delta_QV_std, delta_TV_std = np.std(delta_QV), np.std(delta_TV)                            # Q(v)的第100循环与第2循环差的标准差
    delta_QV_var, delta_TV_var = np.var(delta_QV), np.var(delta_TV)                            # Q(v)的第100循环与第2循环差的标准差
    delta_QV_max, delta_TV_max = np.max(delta_QV), np.max(delta_TV)
    delta_QV_min, delta_TV_min = np.min(delta_QV), np.min(delta_TV)
    # 第2循环的特征统计数据
    Q_V2_min, T_V2_min = np.min(Q_V2), np.min(T_V2)
    Q_V2_max, T_V2_max = np.max(Q_V2), np.max(T_V2)
    Q_V2_mean, T_V2_mean = np.mean(Q_V2), np.mean(T_V2)
    Q_V2_std, T_V2_std = np.std(Q_V2), np.std(T_V2)
    Q_V2_var, T_V2_var = np.var(Q_V2), np.var(T_V2)

    # 拐点对应循环的特征数据
    Q_Vknee_min, T_Vknee_min = np.min(Q_Vknee), np.min(T_Vknee)
    Q_Vknee_max, T_Vknee_max = np.max(Q_Vknee), np.max(T_Vknee)
    Q_Vknee_mean, T_Vknee_mean = np.mean(Q_Vknee), np.mean(T_Vknee)
    Q_Vknee_std, T_Vknee_std = np.std(Q_Vknee), np.std(T_Vknee)
    Q_Vknee_var, T_Vknee_var = np.var(Q_Vknee), np.var(T_Vknee)

    Qd2, Qd100, Qdknee = value['summary']['QD'][1 + bias], value['summary']['QD'][99 + bias], value['summary']['QD'][knee_point]
    Qc2, Qc100, QCknee = value['summary']['QC'][1 + bias], value['summary']['QC'][99 + bias], value['summary']['QC'][knee_point]
    IR2, IR100, IRknee = value['summary']['IR'][1 + bias], value['summary']['IR'][99 + bias], value['summary']['IR'][knee_point]
    Tmax2, Tmax100, Tmaxknee = value['summary']['Tmax'][1 + bias], value['summary']['Tmax'][99 + bias], value['summary']['Tmax'][knee_point]
    Tavg2, Tavg100, Tavgknee = value['summary']['Tavg'][1 + bias], value['summary']['Tavg'][99 + bias], value['summary']['Tavg'][knee_point]
    cha_time2, cha_time100, charge_timeknee = value['summary']['chargetime'][1 + bias], value['summary']['chargetime'][99 + bias], value['summary']['chargetime'][knee_point]

    res.extend([avg_I2, avg_I100, avg_Iknee, avg_T2, avg_T100, avg_Tknee, knee_point,
                delta_QV_mean, delta_TV_mean, delta_QV_std, delta_TV_std, delta_QV_var, delta_TV_var, delta_QV_max, delta_TV_max, delta_QV_min, delta_TV_min,
                Q_V2_min, T_V2_min, Q_V2_max, T_V2_max, Q_V2_mean, T_V2_mean, Q_V2_std, T_V2_std, Q_V2_var, T_V2_var, Q_Vknee_min, T_Vknee_min, Q_Vknee_max, T_Vknee_max,
                Q_Vknee_mean, T_Vknee_mean, Q_Vknee_std, T_Vknee_std, Q_Vknee_var, T_Vknee_var, Qd2, Qd100, Qdknee, Qc2, Qc100, QCknee,
                IR2, IR100, IRknee, Tmax2, Tmax100, Tmaxknee, Tavg2, Tavg100, Tavgknee, cha_time2, cha_time100, charge_timeknee])
    return torch.tensor(res, dtype=torch.float32)


def get_query_values(value, bias):
    '''
    :param value:电池的特征数据，数据类型为字典
    :return: 返回该电池训练时多头注意力对应的查询query
    '''
    res = []
    res.extend(value['charge_policy'])                    # 充电电流数据

    I2 = value['cycles']['2']['I']
    t2 = value['cycles']['2']['t']
    V2 = value['cycles']['2']['V']
    T2 = value['cycles']['2']['T']
    V2, I2, T2, t2 = get_start_end_feature(V2, I2, T2, t2)                                     # 第2循环的特征曲线

    avg_I2, avg_T2 = np.mean(I2), np.mean(T2)

    Q_V2 = value['cycles']['2']['Qdlin']      # 放电阶段Q(V)曲线
    T_V2 = value['cycles']['2']['Tdlin']      # 放电阶段T(V)曲线
    # 第2循环的特征统计数据
    Q_V2_min, T_V2_min = np.min(Q_V2), np.min(T_V2)
    Q_V2_max, T_V2_max = np.max(Q_V2), np.max(T_V2)
    Q_V2_mean, T_V2_mean = np.mean(Q_V2), np.mean(T_V2)
    Q_V2_std, T_V2_std = np.std(Q_V2), np.std(T_V2)
    Q_V2_var, T_V2_var = np.var(Q_V2), np.var(T_V2)

    Qd2 = value['summary']['QD'][1 + bias]
    Qc2= value['summary']['QC'][1 + bias]
    IR2 = value['summary']['IR'][1 + bias]
    Tmax2= value['summary']['Tmax'][1 + bias]
    Tavg2 = value['summary']['Tavg'][1 + bias]
    cha_time2 = value['summary']['chargetime'][1 + bias]

    res.extend([avg_I2, avg_T2, Q_V2_min, T_V2_min, Q_V2_max, T_V2_max, Q_V2_mean, T_V2_mean, Q_V2_std,
                T_V2_std, Q_V2_var, T_V2_var, IR2, Qd2, Qc2, Tmax2, Tavg2, cha_time2])
    return torch.tensor(res, dtype=torch.float32)


# 滑动窗口构造样本数据____充电阶段的数据，每个样本包含前a个初始循环数据；L窗口大小；S滑动步长； data是一个字典;
# 特征是充电阶段的[V, I, T]，标签为SOH和RUL
def getRULTrainData(data, num_samples, H, L, S, lib_dict=None, train=True, plot=False, key_value=False, query=False, cycle_steps=1, is_diff=False):
    '''
    :param data: 电池字典数据
    :param num_samples:采样点个数
    :param H: 初始循环的个数
    :param L: 最近循环的个数
    :param S: 滑动窗口的滑动步长
    :param train:
    :param plot：是否需要画图
    :param key_value:需要构建键值对
    :param query:是否构建query
    :param cycle_steps:循环数采样间隔
    :return:
    '''
    if train:
        X, Y = [], []
    else:
        X, Y = {}, {}
    if plot:
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        axes = axes.flat
    if key_value:
        key_value_X = []
    if query:
        query_X = []
    # 从训练集电池中手动提取特征，作为了注意力机制中的key-value对；每个电池产生一维tensor。
    for key, value in data.items():
        if plot:fig.suptitle('%s' % key)
        # 将该电池的特征数据和标签数据整理好
        feature, label = [], []
        cycles = value['cycles']                            # 单个电池的多个循环数据
        cmap = plt.cm.get_cmap('RdBu', len(cycles))
        Eol = value['cycle_life']                           # 截止寿命
        count = 0
        bias = 1 if key[1] in ['2', '3'] else 0
        if key_value:
            attention_value = get_attention_values(value, bias)
            attention_value = torch.concat([torch.tensor(lib_dict[key], dtype=torch.float32).unsqueeze(dim=0), attention_value], dim=-1)       # 第一个值为电池的序号
            # attention_value = torch.tensor(attention_value, dtype=torch.float32)
            # assert not torch.isnan(attention_value).any(), [key, attention_value]
            key_value_X.append(attention_value)
        if query:
            query_value = get_query_values(value, bias)
            # assert not torch.isnan(query_value).any(), [key, query_value]
            query_X.append(torch.concat([torch.tensor(lib_dict[key], dtype=torch.float32).unsqueeze(dim=0), query_value], dim=0))             # 查询的第一个值是电池的序号
        for j, cycle in enumerate(cycles.values()):                     # 遍历每次循环
            if j == 0 and key[1] in ['2', '3']:continue                 # 跳过第2、3批次的第一个循环2
            Qdlin = np.array(cycle['Qdlin'])
            V = cycle['V']
            I = cycle['I']
            t = cycle['t']
            T = cycle['T']
            Q = value['summary']['QD'][j]
            num = value['summary']['cycle'][j]
            # 去除异常值
            if Q > 1.3 or Q < 0.88 or np.max(Qdlin) > 1.5 or np.min(Qdlin) < -0.25 or np.max(np.abs(np.diff(t))) > 10:
                count += 1
                continue
            # if np.max(np.abs(np.diff(V))) > 0.5:   # 判断当前循环是否有异常值
            #     continue
            deleteList = []
            for i in range(1, len(t)):
                if t[i] - t[i - 1] < 0.00015:
                    deleteList.append(i)
            V, I, T, t = np.delete(V, deleteList), np.delete(I, deleteList), np.delete(T, deleteList), np.delete(t, deleteList)
            i0, i1 = 0, 0                             # 分别为充电开始下标和恒流（4C）放电开始下标
            i = 0
            while not (I[i] > 0 and I[i] - I[i - 1] > 0.01 and I[i + 2] > 0.1 and I[i + 4] > 0.2):
                i += 1
            i0 = i
            for i in range(1, len(I)):
                if i1 != 0:
                    break
                if I[i] < 0 and I[i] - I[i - 1] < -0.01 and I[i + 2] < -0.2 and I[i + 4] < -0.4:
                    i1 = i
            if plot:
                axes[0].plot(t[i0:i1], V[i0:i1], color=cmap(j), linewidth=0.2)
                axes[1].plot(t[i0:i1], I[i0:i1], color=cmap(j), linewidth=0.2)
                axes[2].plot(t[i0:i1], T[i0:i1], color=cmap(j), linewidth=0.2)
                axes[3].scatter(j, Q, color=cmap(j), s=2)
                axes[0].set_title('V'), axes[1].set_title('I'), axes[2].set_title('T'), axes[3].set_title('Q')
            if lib_dict is not None:
                V, I, T, _ = getNew(V, I, T, t, num_samples, lib_dict[key], [i0, i1])          # 线性插值等间隔采样
            else:
                V, I, T, _ = getNew(V, I, T, t, num_samples, -1, [i0, i1])                         # 线性插值等间隔采样

            if plot:
                axes[4].plot(np.arange(len(V)), V, color=cmap(j), linewidth=0.2)
                axes[5].plot(np.arange(len(V)), I, color=cmap(j), linewidth=0.2)
                axes[6].plot(np.arange(len(V)), T, color=cmap(j), linewidth=0.2)
                axes[7].scatter(j, Q, color=cmap(j), s=2)
                axes[4].set_title('V'), axes[5].set_title('I'), axes[6].set_title('T'), axes[7].set_title('Q')

            V, I, T = torch.tensor(V, dtype=torch.float32), torch.tensor(I, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
            feature.append(torch.stack([V, I, T]))                # 单个循环对应的特征数据，包括电压、电流和温度
            label.append(torch.tensor([Q / 1.1, Eol - num], dtype=torch.float32))                # 单个循环对应的SOH和RUL
        with open('delete_lib_cycle.txt', 'a') as f:
            f.write(key + '  ' + str(count) + '\n')
        # print(key, count)
        if plot:
            fig.savefig('plot_fig/%s.jpeg' % key, dpi=200, bbox_inches='tight')                  # 指定分辨率
        # plt.show()
        if plot:
            axes[0].cla(), axes[1].cla(), axes[2].cla(), axes[3].cla()               # 清空axes的子图
            axes[4].cla(), axes[5].cla(), axes[6].cla(), axes[7].cla()               # 清空axes的子图
        # 针对标签异常的数据，进行删除，并采用滑动窗口的思想生成样本
        feature, label = torch.stack(feature), torch.stack(label)

        if is_diff:
            init_X = feature[H].unsqueeze(dim=0)  # 初始循环特征
            if not train:
                temp_X, temp_Y = [], []
                for i in range(L, len(feature), S):
                    x = feature[i - L + 1:i + 1:cycle_steps]
                    x_diff = x - init_X
                    temp_X.append(torch.concat([x_diff, x], dim=1))
                    temp_Y.append(torch.concat([label[i - L + cycle_steps - 1:i - 1:cycle_steps, 0], label[i]]))
                X[key] = torch.stack(temp_X)
                Y[key] = torch.stack(temp_Y)
            else:
                for i in range(L, len(feature), S):
                    x = feature[i - L + 1:i + 1:cycle_steps]
                    x_diff = x - init_X
                    X.append(torch.concat([x_diff, x], dim=1))
                    Y.append(torch.concat([label[i - L + cycle_steps - 1:i - 1:cycle_steps, 0], label[i]]))
        else:
            init_X = feature[:H]                                                     # 初始循环特征
            if not train:
                temp_X, temp_Y = [], []
                for i in range(L + H, len(feature), S):
                    temp_X.append(torch.concat([init_X, feature[i - L + 1:i + 1:cycle_steps]], dim=0))
                    temp_Y.append(label[i])
                X[key] = torch.stack(temp_X)
                Y[key] = torch.stack(temp_Y)
            else:
                for i in range(L + H, len(feature), S):
                    X.append(torch.concat([init_X, feature[i - L + 1:i + 1:cycle_steps]], dim=0))
                    Y.append(label[i])

    if train:
        X, Y = torch.stack(X), torch.stack(Y)
    if plot:plt.close()

    # 只有训练集都有注意力的键值对，而所有的数据集都有注意力的query。
    if key_value:
        if query:
            return X, Y, torch.stack(key_value_X), torch.stack(query_X)
        else:
            return X, Y, torch.stack(key_value_X)
    else:
        if query:
            return X, Y, torch.stack(query_X)
        else:
            return X, Y


# 滑动窗口构造样本数据____放电阶段的数据，每个样本包含前H个初始循环数据；L窗口大小；S滑动步长； data是一个字典;
# 特征是放电阶段的Q(V)，T(V)     标签为SOH和RUL
def getRULTrainDataDischarge(data, H, L, S, train=True, plot=True, start=0, end=1000, cycle_steps=1):
    '''
    :param data: 电池字典数据
    :param H: 初始循环的个数
    :param L: 最近循环的个数
    :param S: 滑动窗口的滑动步长
    :param train:True:返回所有电池的样本格式；否则返回电池的字典格式
    :param start、end：放电曲线的开始和结束索引
    :return:
    '''
    if train:
        X, Y = [], []
    else:
        X, Y = {}, {}
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(10, 6))

    for key, value in data.items():
        if plot:fig.suptitle('%s' % key)
        # 将该电池的特征数据和标签数据整理好
        feature, label = [], []
        cycles = value['cycles']                            # 单个电池的多个循环数据
        cmap = plt.cm.get_cmap('RdBu', len(cycles))
        Eol = value['cycle_life']                           # 截止寿命
        count = 0
        for j, cycle in enumerate(cycles.values()):         # 遍历每次循环
            Qdlin = np.array(cycle['Qdlin'])[start:end:2]
            Tdlin = np.array(cycle['Tdlin'])[start:end:2]
            Q = value['summary']['QD'][j]
            num = value['summary']['cycle'][j]
            I = cycle['I']
            t = cycle['t']
            i1, i2 = 0, 0
            for i in range(1, len(I)):
                if i1 != 0:
                    break
                if I[i] < 0 and I[i] - I[i - 1] < -0.01 and I[i + 2] < -0.2 and I[i + 4] < -0.4:
                    i1 = i
            for i in range(i1, len(I)):
                if i2 != 0:break
                if I[i] > I[i - 1] and abs(I[i] + 4) > 1 and abs(I[i + 2] + 4) > 1:
                    i2 = i
            Dc_time = t[i2] - t[i1]
            IR = value['summary']['IR'][j]
            # 去除异常值
            if Q > 1.3 or Q < 0.88 or np.max(Qdlin) > 1.5 or np.min(Qdlin) < -0.25:
                count += 1
                continue
            if plot:
                axes[0].plot(np.arange(len(Qdlin)), Qdlin, color=cmap(j), linewidth=0.2)
                axes[1].plot(np.arange(len(Tdlin)), Tdlin, color=cmap(j), linewidth=0.2)
                axes[2].scatter(j, Q, color=cmap(j), s=2)
                axes[0].set_title('Q(V)'), axes[1].set_title('Q'), axes[2].set_title('T(V)')

            Qdlin = torch.tensor(Qdlin, dtype=torch.float32)
            Tdlin = torch.tensor(Tdlin, dtype=torch.float32)
            scalas = torch.tensor([Q, Dc_time, IR], dtype=torch.float32)
            Qdlin = torch.concat([Qdlin, scalas], dim=0)
            Tdlin = torch.concat([Tdlin, scalas], dim=0)
            feature.append(torch.stack([Qdlin, Tdlin]))                     # 单个循环对应的特征数据，包括电压、电流和温度
            label.append(torch.tensor([num, Eol - num], dtype=torch.float32))                # 单个循环对应的SOH和RUL

        # with open('delete_lib_cycle_discharge.txt', 'a') as f:
        #     f.write(key + '  ' + str(count) + '\n')

        print(key, count)
        if plot:
            fig.savefig('plot_fig_dis/%s.jpeg' % key, dpi=800, bbox_inches='tight')                  # 指定分辨率
        # plt.show()
        if plot:
            axes[0].cla(), axes[1].cla(), axes[2].cla()                          # 清空axes的子图
        # 针对标签异常的数据，进行删除，并采用滑动窗口的思想生成样本
        feature, label = torch.stack(feature), torch.stack(label)

        init_X = feature[:H]                                                     # 初始循环特征
        if not train:
            temp_X, temp_Y = [], []
            for i in range(L + H, len(feature), S):
                temp_X.append(torch.concat([init_X, feature[i - L:i:cycle_steps]], dim=0))
                temp_Y.append(label[i])
            X[key] = torch.stack(temp_X)
            Y[key] = torch.stack(temp_Y)
        else:
            for i in range(L + H, len(feature), S):
                X.append(torch.concat([init_X, feature[i - L:i:cycle_steps]], dim=0))
                Y.append(label[i])

    if train:
        X, Y = torch.stack(X), torch.stack(Y)
    if plot:plt.close()
    return X, Y


# 滑动窗口构造样本数据____充电阶段的数据，每个样本包含前a个初始循环数据；L窗口大小；S滑动步长；data是一个字典;
# 特征是充电阶段的[V, I, T]，标签为SOH、CCl和RUL, 用Qc进行重采样和插值
def getRULTrainDataDc(data, num_samples, H, L, S, train=True, plot=True, cycle_steps=1):
    '''
    :param data: 电池字典数据
    :param num_samples:采样点个数
    :param H: 初始循环的个数
    :param L: 最近循环的个数
    :param S: 滑动窗口的滑动步长
    :param train:
    :param plot：是否需要画图
    :param key_value:需要构建键值对
    :param query:是否构建query
    :param cycle_steps:循环数采样间隔
    :return:
    '''
    if train:
        X, Y = [], []
    else:
        X, Y = {}, {}
    if plot:
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        axes = axes.flat
    # 从训练集电池中手动提取特征，作为了注意力机制中的key-value对；每个电池产生一维tensor。
    for key, value in data.items():
        if plot:fig.suptitle('%s' % key)
        # 将该电池的特征数据和标签数据整理好
        feature, label = [], []
        cycles = value['cycles']                            # 单个电池的多个循环数据
        cmap = plt.cm.get_cmap('RdBu', len(cycles))
        Eol = value['cycle_life']                           # 截止寿命
        count = 0

        for j, cycle in enumerate(cycles.values()):                     # 遍历每次循环
            if j == 0 and key[1] in ['2', '3']:continue                 # 跳过第2、3批次的第一个循环2
            Qdlin = np.array(cycle['Qdlin'])
            V = cycle['V']
            I = cycle['I']
            t = cycle['t']
            T = cycle['T']
            Qc = cycle['Qc']
            Q = value['summary']['QD'][j]
            num = value['summary']['cycle'][j]
            IR = value['summary']['IR'][j]
            # 去除异常值
            if Q > 1.3 or Q < 0.88 or np.max(Qdlin) > 1.5 or np.min(Qdlin) < -0.25 or np.max(np.abs(np.diff(t))) > 10:
                count += 1
                continue
            # if np.max(np.abs(np.diff(V))) > 0.5:   # 判断当前循环是否有异常值
            #     continue
            V1, I1, T1, (i0, i1) = getNew2(V, I, T, Qc, num_samples)                    # 线性插值等间隔采样
            if plot:
                axes[0].plot(t[i0:i1], V[i0:i1], color=cmap(j), linewidth=0.2)
                axes[1].plot(t[i0:i1], I[i0:i1], color=cmap(j), linewidth=0.2)
                axes[2].plot(t[i0:i1], T[i0:i1], color=cmap(j), linewidth=0.2)
                axes[3].scatter(j, Q, color=cmap(j), s=2)
                axes[0].set_title('V'), axes[1].set_title('I'), axes[2].set_title('T'), axes[3].set_title('Q')

            if plot:
                axes[4].plot(np.arange(len(V1)), V1, color=cmap(j), linewidth=0.2)
                axes[5].plot(np.arange(len(V1)), I1, color=cmap(j), linewidth=0.2)
                axes[6].plot(np.arange(len(V1)), T1, color=cmap(j), linewidth=0.2)
                axes[7].scatter(j, Q, color=cmap(j), s=2)
                axes[4].set_title('V'), axes[5].set_title('I'), axes[6].set_title('T'), axes[7].set_title('Q')

            V, I, T = torch.tensor(V1, dtype=torch.float32), torch.tensor(I1, dtype=torch.float32), torch.tensor(T1, dtype=torch.float32)
            # scalas = torch.tensor([Q, Dc_time, IR], dtype=torch.float32)
            feature.append(torch.stack([V, I, T]))                # 单个循环对应的特征数据，包括电压、电流和温度
            label.append(torch.tensor([Q / 1.1, num, Eol - num], dtype=torch.float32))                # 单个循环对应的SOH和RUL
        with open('delete_lib_cycle.txt', 'a') as f:
            f.write(key + '  ' + str(count) + '\n')
        # print(key, count)
        if plot:
            fig.savefig('plot_fig_QC/%s.jpeg' % key, dpi=300, bbox_inches='tight')                  # 指定分辨率
        # plt.show()
        if plot:
            axes[0].cla(), axes[1].cla(), axes[2].cla(), axes[3].cla()               # 清空axes的子图
            axes[4].cla(), axes[5].cla(), axes[6].cla(), axes[7].cla()               # 清空axes的子图
        # 针对标签异常的数据，进行删除，并采用滑动窗口的思想生成样本
        feature, label = torch.stack(feature), torch.stack(label)
        init_X = feature[:H]                                                     # 初始循环特征
        if not train:
            temp_X, temp_Y = [], []
            for i in range(L + H, len(feature), S):
                temp_X.append(torch.concat([init_X, feature[i - L:i:cycle_steps]], dim=0))
                temp_Y.append(label[i])
            X[key] = torch.stack(temp_X)
            Y[key] = torch.stack(temp_Y)
        else:
            for i in range(L + H, len(feature), S):
                X.append(torch.concat([init_X, feature[i - L:i:cycle_steps]], dim=0))
                Y.append(label[i])

    if train:X, Y = torch.stack(X), torch.stack(Y)
    if plot:plt.close()
    return X, Y


# 最大最小归一化
def normalize(train_data, test_data):
    temp = torch.concat([train_data, test_data], dim=0)
    max0, _ = torch.max(temp, dim=0, keepdim=True)
    max0, _ = torch.max(max0, dim=1, keepdim=True)
    max0, _ = torch.max(max0, dim=3, keepdim=True)

    min0, _ = torch.min(temp, dim=0, keepdim=True)
    min0, _ = torch.min(min0, dim=1, keepdim=True)
    min0, _ = torch.max(min0, dim=3, keepdim=True)
    return (train_data - min0) / (max0 - min0), (test_data - min0) / (max0 - min0), max0, min0


def normalize2(train_data, test_data):
    temp = torch.concat([train_data, test_data], dim=0)
    max0 = torch.max(temp)
    min0 = torch.min(temp)
    return (train_data - min0) / (max0 - min0), (test_data - min0) / (max0 - min0), max0, min0


def try_gpu(i = 0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


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


def evaluate_metric(net, data_iter, device, L=1):
    metric = Accumulator(5)              # No. of correct predictions, no. of predictions
    with torch.no_grad():
        net.eval()                       # Set the model to evaluation mode
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_pred = net(X)
            metric.add(torch.sum(torch.abs(y_pred[:, :L] - y[:, :L])),          # MAE
                       torch.sum(torch.square(y_pred[:, :L] - y[:, :L])),       # RMSE
                       torch.sum(torch.abs(y_pred[:, L] - y[:, L])),
                       torch.sum(torch.square(y_pred[:, L] - y[:, L])),
                       torch.tensor(len(X)))
    # SOH评估的MAE、RMSE；  RUL评估的MAE和RMSE
    return metric[0] / (metric[4] * L), np.sqrt(metric[1] / (metric[4] * L)), metric[2] / metric[4], np.sqrt(metric[3] / metric[4])


# def f(y):
#     return torch.log10(y + 1)
#
# def f2(y):
#     return torch.pow(10, y) - 1


# MAPE损失函数
# def loss(y_hat, y):
#     return torch.mean(torch.abs(y_hat - y) / y)


# 训练神经网络的代码
def train_ch6(net, train_iter, val_iter, num_epochs, lr, device, gamma=(1, 1e-1), wind=None, test_iter=None, delta=5):
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

    # loss1 = nn.L1Loss()
    loss = nn.HuberLoss(delta=delta)
    for epoch in range(1, num_epochs + 1):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(7)
        # true_y, pred_y = [], []
        for i, (X, y) in enumerate(train_iter):
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l1, l2 = gamma[0] * loss(y_hat[:, 0], y[:, 0]), gamma[1] * loss(y_hat[:, 1], y[:, 1])
            l = l1 + l2
            l.backward()
            # true_y.extend(y[:, 1].detach().cpu())
            # pred_y.extend(y_hat[:, 1].detach().cpu())
            # grad_clipping(net, 10)                                              # 梯度裁剪,防止梯度爆炸
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10, foreach=None)
            optimizer.step()
            with torch.no_grad():
                metric.add(l1, l2,
                           torch.sum(torch.abs(y_hat[:, 0] - y[:, 0])),           # MAE
                           torch.sum(torch.square(y_hat[:, 0] - y[:, 0])),        # RMSE
                           torch.sum(torch.abs(y_hat[:, 1] - y[:, 1])),
                           torch.sum(torch.square(y_hat[:, 1] - y[:, 1])),
                           torch.tensor(len(X)))
        scheduler.step()
        if wind and epoch % 5 == 0:
            net.eval()
            test_metric = evaluate_metric(net, val_iter, device=device)
            wind.line([[metric[0], metric[1]]], [epoch - 2], win='train', update='append')
            wind.line([[metric[2] / metric[6], test_metric[0]]], [epoch - 2], win='SOH', update='append')
            wind.line([[metric[4] / metric[6], test_metric[2]]], [epoch - 2], win='RUL', update='append')
            print('epoch {}'.format(epoch), f'loss {(metric[0] + metric[1]) / metric[6]:.3f}, ### SOH ### train MAE {metric[2] / metric[6]:.3f}, 'f'val MAE {test_metric[0]:.3f}, '
                                            f'     ### RUL ### train MAE {metric[4] / metric[6]:.3f}, 'f'val MAE {test_metric[2]:.3f}')
            # true_y, pred_y = torch.tensor(true_y), torch.tensor(pred_y)
            # print('RUL 训练集MAE：%.3f' % (MAE(pred_y, true_y)))
            if test_iter:
                test_metric = evaluate_metric(net, test_iter, device=device)
                print('RUL 测试集MAE：%.3f' % (test_metric[2]))


def train_ch7(net, train_iter, num_epochs, lr, device, gamma=(1, 1), wind=None, test_iter=None, delta=5, L=12):
    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         # nn.init.kaiming_uniform_(m.weight)            # ReLU激活函数更适合
    #         nn.init.xavier_uniform_(m.weight)               # sign和tanh更适合

    # net.apply(init_weights)
    # print('training on=======================', device)
    net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 10, gamma=0.7)
    # K = num_epochs // 10
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: np.cos(x % K / K) / ((x // K) * 0.4 + 1))

    # loss1 = nn.L1Loss()
    loss = nn.HuberLoss(delta=20)
    # loss = nn.MSELoss()
    for epoch in range(1, num_epochs + 1):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(7)
        # true_y, pred_y = [], []
        for i, (X, y) in enumerate(train_iter):
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l1, l2 = gamma[0] * loss(y_hat[:, :L], y[:, :L]), gamma[1] * loss(y_hat[:, L], y[:, L])
            l = l1 + l2
            l.backward()
            # true_y.extend(y[:, 1].detach().cpu())
            # pred_y.extend(y_hat[:, 1].detach().cpu())
            # grad_clipping(net, 10)                                              # 梯度裁剪,防止梯度爆炸
            # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=5, foreach=None)
            optimizer.step()
            with torch.no_grad():
                metric.add(l1, l2,
                           torch.sum(torch.abs(y_hat[:, :L] - y[:, :L])),           # MAE
                           torch.sum(torch.square(y_hat[:, L:] - y[:, :L])),          # RMSE
                           torch.sum(torch.abs(y_hat[:, L] - y[:, L])),
                           torch.sum(torch.square(y_hat[:, L] - y[:, L])),
                           torch.tensor(len(X)))
        scheduler.step()
        if wind and epoch % 5 == 0:
            net.eval()
            test_metric = evaluate_metric(net, test_iter, device=device, L=L)
            wind.line([[metric[0], metric[1]]], [epoch - 2], win='train', update='append')
            wind.line([[metric[2] / (metric[6] * L), test_metric[0]]], [epoch - 2], win='SOH', update='append')
            wind.line([[metric[4] / metric[6], test_metric[2]]], [epoch - 2], win='RUL', update='append')
            print('epoch {}'.format(epoch), f'loss {(metric[0] + metric[1]) / metric[6]:.3f}, ### SOH ### train MAE {metric[2] / (metric[6] * L):.3f}, 'f'test MAE {test_metric[0]:.3f}, '
                                            f'     ### RUL ### train MAE {metric[4] / metric[6]:.3f}, 'f'test MAE {test_metric[2]:.3f}')


def train_MTL(net, train_iter, test_iter, num_epochs, lr, device, alpha, wind=None, L=10, need_plot=False):
    # def init_weights(m):
    #     if type(m) == nn.Linear or type(m) == nn.Conv2d:
    #         # nn.init.kaiming_uniform_(m.weight)            # ReLU激活函数更适合
    #         nn.init.xavier_uniform_(m.weight)               # sign和tanh更适合
    # net.apply(init_weights)
    if need_plot:plot_datas = []
    print('training on=======================', device)
    net.to(device)
    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4] == 'bias' or 'bn' in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0}]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 10, gamma=0.7)

    for epoch in range(1, num_epochs + 1):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(7)
        for i, (X, y) in enumerate(train_iter):
            net.train()
            X, y = X.to(device), y.to(device)
            task_loss, y_hat = net(X, y)
            weighted_task_loss = torch.mul(net.weights, task_loss)
            if epoch == 1:
                initial_task_loss = task_loss.data.cpu().numpy()
            l = torch.sum(weighted_task_loss)
            optimizer.zero_grad()
            l.backward(retain_graph=True)
            net.weights.grad.data = net.weights.grad.data * 0.0

            W = net.get_last_shared_layer()
            norms = []
            for i in range(len(task_loss)):
                # 梯度会累加，所以norms最终是整个数据集整体的梯度范数和
                gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                norms.append(torch.norm(torch.mul(net.weights[i], gygw[0])))

            norms = torch.stack(norms)
            loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
            inverse_train_rate = loss_ratio / np.mean(loss_ratio)
            mean_norm = np.mean(norms.data.cpu().numpy())
            # compute the GradNorm loss, this term has to remain constant
            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
            constant_term = constant_term.to(device)
            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
            net.weights.grad = torch.autograd.grad(grad_norm_loss, net.weights)[0]
            optimizer.step()
            if net.weights.data[1] <= 1e-5:
                net.weights.data[1] = 1e-5

            with torch.no_grad():
                metric.add(torch.sum(torch.abs(y_hat[:, :L] - y[:, :L])),           # MAE
                           torch.sum(torch.square(y_hat[:, :L] - y[:, :L])),        # RMSE
                           torch.sum(torch.abs(y_hat[:, L] - y[:, L])),
                           torch.sum(torch.square(y_hat[:, L] - y[:, L])),
                           torch.tensor(len(X)))
        normalize_coeff = len(task_loss) / torch.sum(net.weights.data, dim=0)
        net.weights.data = net.weights.data * normalize_coeff

        scheduler.step()
        if need_plot:
            net.eval()
            test_metric = evaluate_metric(net.model, test_iter, device=device, L=L)
            loss_ratios_SOH_epoch = task_loss[0].detach().cpu().numpy() / initial_task_loss[0]
            loss_ratios_RUL_epoch = task_loss[1].detach().cpu().numpy() / initial_task_loss[1]
            weights_SOH_epoch = net.weights[0].detach().cpu().numpy()
            weights_RUL_epoch = net.weights[1].detach().cpu().numpy()
            SOH_train_epoch = metric[0] / (metric[4] * L)
            SOH_test_epoch = test_metric[0]
            RUL_train_epoch = metric[2] / metric[4]
            RUL_test_epoch = test_metric[2]
            plot_datas.append([loss_ratios_SOH_epoch, loss_ratios_RUL_epoch, weights_SOH_epoch, weights_RUL_epoch, SOH_train_epoch, SOH_test_epoch, RUL_train_epoch, RUL_test_epoch])


        if wind and epoch % 5 == 0:
            net.eval()
            test_metric = evaluate_metric(net.model, test_iter, device=device, L=L)

            wind.line([[metric[0] / (metric[4] * L), test_metric[0]]], [epoch - 2], win='SOH', update='append')
            wind.line([[metric[2] / metric[4], test_metric[2]]], [epoch - 2], win='RUL', update='append')
            weighted_task_loss = weighted_task_loss.detach().cpu().numpy()
            wind.line([[weighted_task_loss[0], weighted_task_loss[1]]], [epoch - 2], win='loss', update='append')
            wind.line([[net.weights[0].detach().cpu().numpy(), net.weights[1].detach().cpu().numpy()]], [epoch - 2], win='weights', update='append')
            wind.line([np.mean(task_loss.detach().cpu().numpy() / initial_task_loss)], [epoch - 2], win='loss_ratios', update='append')
            wind.line([grad_norm_loss.detach().cpu().numpy()], [epoch - 2], win='grad_norm_losses', update='append')

            print('epoch {}'.format(epoch), f' ### SOH ### train MAE {metric[0] / (metric[4] * L):.3f}, 'f'test MAE {test_metric[0]:.3f}, '
                                            f'     ### RUL ### train MAE {metric[2] / metric[4]:.3f}, 'f'test MAE {test_metric[2]:.3f}')
    if need_plot:
        plot_datas = np.array(plot_datas)
        with open('plot_train_raw_data.pkl', 'wb') as fp:
            pickle.dump(plot_datas, fp)



def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 双斜杠表示除完后再向下取整
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)        # slice(start,end,step)切片函数
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)      # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 传入的X， y都是电池维度的字典数据
def get_k_fold_data2(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = len(X.keys()) // k          # 双斜杠表示除完后再向下取整
    lib_keys = list(X.keys())               # 电池字典的电池索引
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)        # slice(start,end,step)切片函数
        x_keys = lib_keys[idx]
        X_part = torch.concat([X[key] for key in x_keys], dim=0)
        y_part = torch.concat([y[key] for key in x_keys], dim=0)
        # X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)      # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


# 不画图不调试，仅仅训练模型
def train(net, train_X, train_Y, test_X, test_Y, num_epochs, batch_size, updater, device, delta=5):
    loss = nn.HuberLoss(delta=delta)
    train_X, train_Y = train_X.to(device), train_Y.to(device)
    test_X, test_Y = test_X.to(device), test_Y.to(device)
    dataSet = TensorDataset(train_X, train_Y)
    dataLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=True)
    dataSet2 = TensorDataset(test_X, test_Y)
    test_iter = DataLoader(dataSet2, batch_size=batch_size, shuffle=True)
    scheduler = torch.optim.lr_scheduler.StepLR(updater, step_size=num_epochs // 10, gamma=0.7)
    for epoch in range(num_epochs):
        net.train()
        for X, y in dataLoader:
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat.reshape(y.shape), y)
            l.backward()
            updater.step()
        scheduler.step()
        # 在测试集上评估
    net.eval()
    test_metric = evaluate_metric(net, test_iter, device=device)
    # 返回最后的MAE和RMSE
    return test_metric[2], test_metric[3]



#@save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)


    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # self.attention = AdditiveAttention(key_size, query_size, num_hiddens, dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)


    def forward(self, queries, keys, values, valid_lens=None):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:(batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttention2(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention2, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)


    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, mask_idx=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)         # score的shape:批量 * 查询个数 * 键值对个数
        scores[mask_idx] = -1e6                                                  # 掩蔽注意力
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 适用于电池多头注意力机制的模块实现
class MultiHeadAttention2(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention2, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention2(dropout)
        # self.attention = AdditiveAttention(key_size, query_size, num_hiddens, dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)


    def forward(self, queries, keys, values, mask_num, columns):
        if mask_num is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            mask_num = torch.repeat_interleave(mask_num, repeats=self.num_heads, dim=0)
        mask_idx = pd.DataFrame(np.full((len(mask_num), len(columns)), False), columns=columns)
        for i, n in enumerate(mask_num):
            if n in columns:
                mask_idx.loc[i, n] = True                        # 添加掩码
        mask_idx = torch.tensor(mask_idx.values, dtype=torch.bool)
        mask_idx = mask_idx.unsqueeze(1)                         # shape:批量 * 查询个数 * 键值对个数
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:(batch_size，)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.attention(queries, keys, values, mask_idx)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def Load_HUST_Data():
    dir_name = 'our_data'
    res_dict = {}
    for i in os.listdir(dir_name):
        path = os.path.join(dir_name, i)
        data = pickle.load(open(path, 'rb'))
        res_dict.update(data)
    return res_dict


def getHUSTRULData(data, num_samples, H, L, S, train=True, plot=False, cycle_steps=1, is_diff=True):
    '''
    :param data: 电池字典数据
    :param num_samples:采样点个数
    :param H: 第H个初始循环
    :param L: 最近循环的个数
    :param S: 滑动窗口的滑动步长
    :param train:
    :param plot：是否需要画图
    :param key_value:需要构建键值对
    :param query:是否构建query
    :param cycle_steps:循环数采样间隔
    :return:
    '''
    if train:
        X, Y = [], []
    else:
        X, Y = {}, {}
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flat

    # 从训练集电池中手动提取特征，作为了注意力机制中的key-value对；每个电池产生一维tensor。
    for key, value in data.items():
        if plot:fig.suptitle('%s' % key)
        # 将该电池的特征数据和标签数据整理好
        feature, label = [], []
        cmap = plt.cm.get_cmap('RdBu', len(value['data']))
        for k in value['data']:                                 # cycle是每个循环的数据,data是字典，k是字典的索引
            cycle = value['data'][k]
            index1 = myFind(cycle['Status'].values, 'Constant current-constant voltage charge')       # 查找第二阶段开始充电时刻
            index2 = muFind2(cycle['Current (mA)'].values, cycle['Voltage (V)'].values, index1)       # 查找电压第一次达到3.6V的时刻
            if cycle.loc[index1, 'Capacity (mAh)'] < 500:
                continue  # 剔除电池7-5第2个循环，属于异常数据
            # 对电压异常值进行插值
            out_list = RemoveIndex0(cycle.loc[index1:index2, 'Voltage (V)'], sd_size=50)  # 电压采样点异常的index
            if len(out_list) != 0:
                spline = interp1d(cycle.loc[index1:index2, 'Time (s)'].values,
                                  cycle.loc[index1:index2, 'Voltage (V)'].values, kind='linear', axis=-1)  # 样条曲线
                # t1 = np.linspace(x[0], x[-1], num_samples)  # 插值后的时间点
                for i in out_list:
                    cycle.loc[i, 'Voltage (V)'] = spline(cycle.loc[i, 'Time (s)'])

            V = cycle['Voltage (V)'].values[index1:index2]             # 获取当前循环的电压数据，每6个取一个采样点
            Qc = cycle['Capacity (mAh)'].values[index1:index2]         # 获取当前循环的充电电荷数据，每6个取一个采样点
            Qd = value['dq'][k] / 1000  # 获取当前循环的容量
            RUL = value['rul'][k]
            Time = cycle['Time (s)'].values[index1:index2]
            if plot:
                axes[0].plot(Time, V, color=cmap(k), linewidth=0.2)
                axes[1].plot(Time, Qc, color=cmap(k), linewidth=0.2)
                axes[3].scatter(k, Qd, color=cmap(k), s=2)
                axes[0].set_title('V'), axes[1].set_title('Qc'), axes[2].set_title('Q')

            delete_list1, delete_list2 = RemoveIndex2(V, sd_size=20), RemoveIndex2(Qc, sd_size=20)
            delete_list = [*delete_list1, *delete_list2]
            y = np.vstack([V, Qc])
            y = np.delete(y, delete_list, axis=1)
            x = np.delete(Time, delete_list)
            spline = interp1d(x, y, kind='linear', axis=-1)
            t1 = np.linspace(x[0], x[-1], num_samples)           # 插值后的时间点
            V, Qc = spline(t1)[0].tolist(), spline(t1)[1].tolist()
            if plot:
                axes[3].plot(np.arange(len(V)), V, color=cmap(k), linewidth=0.2)
                axes[4].plot(np.arange(len(V)), Qc, color=cmap(k), linewidth=0.2)
                axes[5].scatter(k, Qd, color=cmap(k), s=2)
                axes[3].set_title('V'), axes[4].set_title('Qc'), axes[5].set_title('Q')

            V, Qc = torch.tensor(V, dtype=torch.float32), torch.tensor(Qc, dtype=torch.float32)
            feature.append(torch.stack([V, Qc]))                # 单个循环对应的特征数据，包括电压、电流和温度
            label.append(torch.tensor([Qd / 1.1, RUL], dtype=torch.float32))                # 单个循环对应的SOH和RUL
        # print(key, count)
        if plot:
            fig.savefig('plot_HUST_fig/%s.jpeg' % key, dpi=200, bbox_inches='tight')                  # 指定分辨率
        if plot:
            axes[0].cla(), axes[1].cla(), axes[2].cla()              # 清空axes的子图
            axes[3].cla(), axes[4].cla(), axes[5].cla()              # 清空axes的子图
        # 针对标签异常的数据，进行删除，并采用滑动窗口的思想生成样本
        feature, label = torch.stack(feature), torch.stack(label)

        if is_diff:
            init_X = feature[H].unsqueeze(dim=0)                                               # 初始循环特征
            if not train:
                temp_X, temp_Y = [], []
                for i in range(L, len(feature), S):
                    x = feature[i - L + 1:i + 1:cycle_steps]
                    x_diff = x - init_X
                    temp_X.append(torch.concat([x_diff, x], dim=1))
                    temp_Y.append(torch.concat([label[i - L + cycle_steps - 1:i - 1:cycle_steps, 0], label[i]]))
                X[key] = torch.stack(temp_X)
                Y[key] = torch.stack(temp_Y)
            else:
                for i in range(L, len(feature), S):
                    x = feature[i - L + 1:i + 1:cycle_steps]
                    x_diff = x - init_X
                    X.append(torch.concat([x_diff, x], dim=1))
                    Y.append(torch.concat([label[i - L + cycle_steps - 1:i - 1:cycle_steps, 0], label[i]]))
        else:
            init_X = feature[:H]                                                     # 初始循环特征
            if not train:
                temp_X, temp_Y = [], []
                for i in range(L + H, len(feature), S):
                    temp_X.append(torch.concat([init_X, feature[i - L + 1:i + 1:cycle_steps]], dim=0))
                    temp_Y.append(torch.concat([label[i - L + 1:i:cycle_steps, 0], label[i]]))
                X[key] = torch.stack(temp_X)
                Y[key] = torch.stack(temp_Y)
            else:
                for i in range(L + H, len(feature), S):
                    X.append(torch.concat([init_X, feature[i - L + 1:i + 1:cycle_steps]], dim=0))
                    Y.append(torch.concat([label[i - L + 1:i:cycle_steps, 0], label[i]]))

    if train:
        X, Y = torch.stack(X), torch.stack(Y)
    if plot:plt.close()
    return X, Y


# 返回value第一次在arr出现的索引
def myFind(arr, value):
    for i in range(len(arr)):
        if arr[i] == value:
            return i
    return -1


# 查找充电过程中80%SOC到电压值达到3.6V
def myFind2(I, V, start):
    for i in range(start, len(I)):
        if abs(I[i] - 1100) > 2 and abs(V[i] - 3.6) < 0.015 and 1100 - I[i+5] > 10:
            return i
    return -1


# 查找放电过程中80%SOC到电压值达到3.6V
def muFind2(I, V, start):
    for i in range(start, len(I)):
        if abs(I[i] - 1100) > 20 and abs(V[i] - 3.6) < 0.01 and 1100 - I[i+5] > 50:
            return i
    return -1


def RemoveIndex0(data, sd_size=40):
    # 3sigma法则去除离群点
    # 删除容量数据异常对应的行
    delete_index_list = []
    i = data.index[0]
    while i < data.index[-1]:
        a = data.loc[i:min(i + sd_size, data.index[-1])]
        mean0, std0 = np.mean(a), np.std(a)
        delete_index = np.array([j for j, x in enumerate(a) if x > mean0 + 3 * std0 or x < mean0 - 3 * std0]) + i      # 3sigma法则
        delete_index_list.extend(delete_index)
        i += sd_size
    return delete_index_list


def RemoveIndex2(data, sd_size=40):
    delete_index_list = []
    for i in range(0, len(data), sd_size):
        a = data[i:min(i + sd_size, len(data))]
        mean0, std0 = np.mean(a), np.std(a)
        delete_index = np.array([j for j, x in enumerate(a) if x > mean0 + 3 * std0 or x < mean0 - 3 * std0]) + i      # 3sigma法则
        delete_index_list.extend(delete_index)
    return delete_index_list