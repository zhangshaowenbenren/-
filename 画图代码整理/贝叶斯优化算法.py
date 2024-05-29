# BOA本质上是根据已有的采样点值来探索解空间，求得解空间上的最大值。
# 此处用来优化模型超参数
# 安装相关包
# pip install optuna
# pip install optuna-integration
# pip install scikit-optimize

import optuna
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_regression(n_samples=1000, n_features=40)          # 测试用例
def optuna_objective(trial):
    # 定义参数空间
    n_estimators = trial.suggest_int("n_estimators", 50, 200, 5)  # 整数型，(参数名称，下界，上界，步长)
    max_depth = trial.suggest_int("max_depth", 5, 20, 5)
    max_features = trial.suggest_int("max_features", 5, 20, 2)
    # max_features = trial.suggest_categorical("max_features",["log2","sqrt","auto"]) #字符型
    min_impurity_decrease = trial.suggest_int("min_impurity_decrease", 0, 5, 1)
    # min_impurity_decrease = trial.suggest_float("min_impurity_decrease",0,5,log=False) #浮点型

    # 定义评估器
    # 需要优化的参数由上述参数空间决定
    # 不需要优化的参数则直接填写具体值
    reg = RFR(n_estimators=n_estimators
              , max_depth=max_depth
              , max_features=max_features
              , min_impurity_decrease=min_impurity_decrease
              , random_state=1412
              , verbose=False
              , n_jobs=-1
              )

    # 交叉验证过程，输出-RMSE
    val = cross_val_score(reg, X, y
                                     , scoring="neg_root_mean_squared_error"
                                     , cv=5  # 交叉验证模式
                                     , verbose=False  # 是否打印进程
                                     , n_jobs=-1  # 线程数
                                     , error_score='raise'
                                     )
    # 最终输出RMSE
    return -val.mean()


def optimizer_optuna(n_trials, algo):
    # 定义使用TPE或者GP
    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    elif algo == "GP":
        from optuna.integration import SkoptSampler
        import skopt
        algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',  # 选择高斯过程
                                          'n_initial_points': 10,  # 初始观测点10个
                                          'acq_func': 'EI'}  # 选择的采集函数为EI，期望增量
                            )

    # 实际优化过程，首先实例化优化器
    study = optuna.create_study(sampler=algo  # 要使用的具体算法
                                ,direction="minimize"  # 优化的方向，可以填写minimize或maximize
                                )
    # 开始优化，n_trials为允许的最大迭代次数
    # 由于参数空间已经在目标函数中定义好，因此不需要输入参数空间
    study.optimize(optuna_objective  # 目标函数
                   , n_trials=n_trials  # 最大迭代次数（包括最初的观测值的）
                   , show_progress_bar=True  # 要不要展示进度条呀？
                   )

    # 可直接从优化好的对象study中调用优化的结果
    # 打印最佳参数与最佳损失值
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")

    return study.best_trial.params, study.best_trial.values


import warnings
warnings.filterwarnings('ignore')

model = RFR(n_estimators=100
              , max_depth=10
              , max_features=10
              , min_impurity_decrease=2
              , random_state=1412
              , verbose=False
              , n_jobs=-1
              )

val = np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error'))
print('BOA优化前的RMSE：%.3f' % -val)

best_params, best_score = optimizer_optuna(20, "GP")        # 默认打印迭代过程
print('best_params:%s' % best_params)
print('BOA优化后的RMSE：%.3f' % best_score[0])