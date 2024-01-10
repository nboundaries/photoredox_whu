# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 17:21
# @Author  : TXH
# @File    : utils.py
# @Software: PyCharm

import logging
import argparse
from functools import partial
import numpy as np
import torch
from torch import nn
from scipy import stats
import matplotlib.pyplot as plt
from math import floor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torchmetrics import R2Score, PearsonCorrCoef, Accuracy, Recall, Precision, AUROC


def simple_check_feats(x):
    fig = plt.figure()
    plt.plot(x.squeeze(), 'r-')
    plt.title('feat')
    plt.xlabel('dim')
    plt.ylabel('value')
    plt.show()

# 拟合一个权重函数
def get_loss_weights0(y: None, prior_bias=0.2) -> torch.tensor:
    # y can be float, list, np.array, or torch.tensor
    #
    yield_priors = lambda x:   0.0130 \
                             + 0.0035*x \
                             - 0.0950*x**2 \
                             + 0.8200*x**3 \
                             - 0.7400*x**4
    from math import floor
    y = floor(y*10.)//10.
    if isinstance(y, float) or isinstance(y, int):
        w = prior_bias-torch.tensor(yield_priors(y))
        return w

    ws = prior_bias-torch.tensor(list(map(yield_priors, y)))[...,None]
    # ws = (ws*10)**2 # too vigorous weights may make unfavored results
    return ws

# 采用dict方式预存。对不同yield区段采取不同的权重
def get_loss_weights(y: None, exponent: float = 1.0):
    probs = np.array([0.173, 0.207, 0.249, 0.351, 0.514, 0.750, 0.929, 1.000, 0.800, 0.383, 0.383])
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = 1. / (probs + 0.5) ** exponent
    weight_dict = dict(zip(ids, weights))
    if isinstance(y, float) or isinstance(y, int):
        w = prior_bias-torch.tensor(yield_priors(y))
        return weight_dict[floor(y*10)]
    if isinstance(y, torch.Tensor):
        return torch.tensor([weight_dict[floor(yy*10)] for yy in y])[...,None]
    if isinstance(y, np.ndarray):
        return np.array([weight_dict[floor(yy * 10)] for yy in y])


def plot_yield(x, y, title='', text='xxx'):
    fig = plt.figure()
    plt.plot(np.arange(0,1.01,0.01), np.arange(0,1.01,0.01), 'r-')
    plt.title(title+' '+'Yields')
    plt.text(x=0.02,
             y=0.93,
             s=text,
             fontdict=dict(fontsize=12,
                           family='sans-serif',
                           weight='bold',
                           ))
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.axis([0,1,0,1])
    plt.scatter(x, y, alpha=0.7)
    plt.show()


def yield_statistics(reaction_data):
    # raw yields
    # file = r'E:\projects\photocatalysis_whu\pred_yield\temp.npy'
    ys = [y for _, y in reaction_data]

    # histogram
    weights = np.ones_like(ys)/float(len(ys))
    fig = plt.figure()
    # hist
    nt, bins, _ = plt.hist(ys, bins=25, weights=weights)

    x = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(0, len(bins) - 1)]).reshape(-1, 1)
    y = np.array(nt).reshape(-1, 1)
    # prior probs
    plt.plot(x, y, 'm.')

    poly_feats = PolynomialFeatures(degree=4, include_bias=False)
    x_poly = poly_feats.fit_transform(x)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    # regression coefficients
    print('regression coefficients: ')
    print(lin_reg.intercept_, lin_reg.coef_)

    y_pred = np.dot(x_poly, lin_reg.coef_.T) + lin_reg.intercept_
    plt.plot(x, y_pred, 'm-')
    # plt.plot(x, 0.3-np.sqrt(y_pred), 'g.')
    plt.plot(x, 0.2 - y_pred, 'k*')
    plt.show()
    return x, y



def create_logger(logger_file_name):
    """
    :param logger_file_name:
    :return:
    """
    logger = logging.getLogger()         # 设定日志对象
    logger.setLevel(logging.INFO)        # 设定日志等级

    file_handler = logging.FileHandler(logger_file_name)   # 文件输出
    console_handler = logging.StreamHandler()              # 控制台输出

    # 输出格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       # 设置文件输出格式
    console_handler.setFormatter(formatter)    # 设施控制台输出格式
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class LogCoshLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(LogCoshLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inp, tgt):
        if self.reduction == 'mean':
            return torch.log(torch.cosh(inp - tgt)).mean()
        if self.reduction == 'sum':
            return torch.log(torch.cosh(inp - tgt)).sum()
        return torch.log(torch.cosh(inp - tgt))

def seed_torch(seed=42):
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # hash
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# activation functions
ACT_FUNCTIONS = {
                'relu': nn.ReLU,
                'prelu': nn.PReLU,
                'elu': nn.ELU,
                'leaky': nn.LeakyReLU,
                'swish': nn.Hardswish,
                'gelu': nn.GELU,
                'relu6': nn.ReLU6,
                'hardtanh': nn.Hardtanh,
                'relu2': partial(nn.Hardtanh, min_val=0, max_val=2),
                'softplus': nn.Softplus
                }

# loss function
LOSS_FUNCTIONS = {
                  'MAE': nn.L1Loss(reduce=False),
                  'MSE': nn.MSELoss(reduction='none'),
                  'Hubber': nn.HuberLoss(reduction='none'),
                  'LogCosh': LogCoshLoss(reduction='none'),
                  'CE': nn.CrossEntropyLoss()
                  }

# accuracy function
ACC_FUNCTIONS = {
                  'R2': R2Score(),
                  'Pearson': PearsonCorrCoef(),
                  'accuracy': Accuracy(),
                  'recall': Recall(),
                  'precision': Precision(),
                  'AUC': AUROC(average='macro', num_classes=3)
                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configTemplates')
    parser.add_argument('-log_path',
                        default='./results/test.log',
                        type=str,
                        help='log file path to save result')

    args = parser.parse_args()

    logger = create_logger(args.log_path)

    logger.info('Begin Training Model...')


    from gen_dataset import get_raw_dataset

    in_file = '../data/scifinder_clean.csv'
    # prepare all datasets
    reactions = get_raw_dataset(in_file)
    x, y = yield_statistics(reactions)

