# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 17:16
# @Author  : TXH
# @File    : train_regressor.py
# @Software: PyCharm

import os
from tqdm import tqdm
import time
from datetime import datetime
import logging
import json

import numpy as np
import torch
from torch import nn
from argparse import ArgumentParser
import random
from functools import partial

from gen_dataset import PhotoDatasets, loaders
from yield_model_dnn import MLP_YIELD as MLP_YIELD # be care of the dimensions of each model
from torchmetrics import R2Score, PearsonCorrCoef, Accuracy, Recall, Precision, AUROC
from sklearn.metrics import r2_score

from utils import create_logger, get_loss_weights, plot_yield, simple_check_feats, LogCoshLoss, seed_torch
from MolCLRInfer import gen_molclr_feats, molclr_pca
from gen_morgan import gen_morgan_feats, morgan_pca

seed_torch(seed=42)

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


def main():
    parser = ArgumentParser()
    parser.add_argument('--file_dir', type=str, default='../data/scifinder_clean.csv', help='Path to raw_data')
    parser.add_argument('--categorical', type=bool, default=False, help='False: regression; True: classification ')

    # use following 7 parameters to tune feats dimensions
    parser.add_argument('--regenerate_molclr', type=bool, default=False, help='regenerate molclr 512 feats')
    parser.add_argument('--molclr_do_pca', type=bool, default=True, help='dim reduction')
    parser.add_argument('--molclr_pca_out', type=int, default=96, help='molclr pca out dim')

    parser.add_argument('--regenerate_fp', type=bool, default=True, help='regenerate fp feats')
    parser.add_argument('--fpsize', type=int, default=512, help='morgan_fp_size')
    parser.add_argument('--morgan_do_pca', type=bool, default=True, help='dim reduction')
    parser.add_argument('--morgan_pca_out', type=int, default=96, help='morgan pca out dim')

    # use following 3 parameters to tune feats combination
    # feat_operations=no is only for MLP_YIELD2 now
    parser.add_argument('--feat_operations', type=str, default='-,+,||,0.5,0.5',
                        help='no, no operations; -,+,||,0.5,0.5 or -,+,+,+ for reactants, products, reagents, catalysts')
    parser.add_argument('--feat_mixing', type=str, default='add',
                        help='no, only the first one; add: w0*molclr+w1*morgan; cat: [w0*molclr||w1*morgan]')
    parser.add_argument('--feat_weights', type=float, default=[0.0, 1.0],
                        help='Feat weights. If mixing is add, then feat=w0*molclr+w1*morgan; if mixing is concat, feat=[w0*molclr||w1*morgan]. w0+w1 is not restricted to be 1')

    # use following 3 parameters to prepare datasets for train, validation, test
    parser.add_argument('--split_mode',
                        type=str,
                        default='rule',
                        help='how to split the dataset. rule guarantees molecules in test dataset have been seen during training')
    parser.add_argument('--split_ratios',
                        type=list,
                        default=[0.8, 0.1, -1],
                        help='ratios of training, validation, and test set')
    parser.add_argument('--batch_sizes',
                        type=list,
                        default=[1024, 64, 64],
                        help='batch_sizes for training, validation and testing')

    # model parameters
    # parser.add_argument('--in_dim', type=int, default=96, help='feat dim fed into the model')
    parser.add_argument('--fc_dims', type=list, default=[128, 16], help='fc dims of the model, 3 layers')
    parser.add_argument('--activation', type=str, default=['relu6', 'relu6'], help='score function, R2Score or Perason')

    # control training
    parser.add_argument('--training', type=bool, default=True, help='training or test')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs')
    parser.add_argument('--view_epochs', type=int, default=200, help='epoch number to show train results')
    parser.add_argument('--valid_epochs', type=int, default=200, help='epoch number to show validation results')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--use_lr_decay', type=bool, default=True, help='lr decay')
    parser.add_argument('--lr_schedule', type=str, default='exponential', help='lr_schedule')

    parser.add_argument('--weight_decay', type=float, default=0.0, help='adam weight decay') # clear default setting of regularization
    parser.add_argument('--regularization', type=str, default='L1', help='do regularization')
    parser.add_argument('--reg_lambda', type=float, default=5e-6,
                        help='coefficient for regularization. For L2, 5e-5 is OK; For L1, 1e-6 will be better')

    parser.add_argument('--grad_clip', type=bool, default=True, help='clip the gradient')
    parser.add_argument('--grad_thresh', type=float, default=3., help='gradient threshold to clip')

    parser.add_argument('--criterion', type=str, default='LogCosh', help='loss function, MAE, MSE, Hubber, LogCosh')
    parser.add_argument('--accuracy', type=str, default='R2', help='score function, R2Score or Pearson')

    parser.add_argument('--dropout_rate', type=float, default=0.28, help='dropout_rate')
    parser.add_argument('--prior_adjust', type=bool, default=True, help='penalize the loss according to data probs')
    parser.add_argument('--prior_bias', type=float, default=0.20, help='weight equals to bias minus predicted probability')
    parser.add_argument('--flooding', type=float, default=4e-3, help='flooding rate')

    parser.add_argument('--model_path', type=str, default='./model/yield_model_regressor.pt', help='save model path')
    parser.add_argument('--log_path', type=str, default='./results/training.log', help='log path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = main()

    # define logger
    logger = create_logger(args.log_path)
    logger.info('')
    logger.info('-' * 50)
    logger.info('')
    logging.info(datetime.now().strftime('date%d-%m_time%H-%M-%S.%f'))


    # molclr feats
    if args.regenerate_molclr:
        print('Regenerating MolCLR features...')
        molclr_feats = gen_molclr_feats(in_file='../data/scifinder_clean_all_mols.csv',
                                        out_file='../data/feats/all_mol_molclr_feats_512.npy',
                                        config_file='./config.yaml')
    else:
        print('Loading precalculated MolCLR features...')
        molclr_feats = np.load('../data/feats/all_mol_molclr_feats_512.npy', allow_pickle=True).item()

    if args.molclr_do_pca:
        if args.molclr_pca_out < 512:
            print('Do PCA for MolCLR feats')
            molclr_feats = molclr_pca(molclr_feats,
                                  out_dim=args.molclr_pca_out,
                                  save_model_path='./model/molclr_pca.pkl')  # 96
        else:
            raise ValueError('PCA in-dim must be larger than out-dim')

    # fp feats
    if args.regenerate_fp:
        print('Generating fp features...') # always. this is quick
        morgan_feats = gen_morgan_feats(in_file='../data/scifinder_clean_all_mols.csv',
                                   out_file='../data/feats/all_mol_morgan_feats_{}.npy'.format(args.fpsize),
                                   fpsize=args.fpsize)
    else:
        print('Loading precalculated fp features...')
        morgan_feats = np.load('../data/feats/all_mol_morgan_feats_{}.npy'.format(args.fpsize), allow_pickle=True).item()

    if args.morgan_do_pca:
        if args.morgan_pca_out < args.fpsize:
            print('Do PCA for FP feats')
            morgan_feats = morgan_pca(morgan_feats,
                                  out_dim=args.morgan_pca_out,
                                  save_model_path='./model/morgan_pca.pkl')  # 96
        else:
            raise ValueError('PCA in-dim must be larger than out-dim')

    ## Now meta data is OK. You can check the features to ensure that they are truly what you want.
    ## check feats
    # test_mol = 'CC(=O)c1ccc(Br)cc1'
    # simple_check_feats(molclr_feats[test_mol])
    # simple_check_feats(morgan_feats[test_mol])


    # Below is the model training
    # prepare all datasets
    dataset = PhotoDatasets(args.file_dir,
                            feat_ops=args.feat_operations,
                            feats_dict1=molclr_feats,
                            feats_dict2=morgan_feats,
                            mixing=args.feat_mixing,
                            weights=args.feat_weights,
                            categorical=args.categorical)
    args.in_dim = dataset.get_feat_dim() # in feat dimension

    train_loader, valid_loader, test_loader = loaders(dataset,
                                                      ratios=args.split_ratios,
                                                      batch_sizes=args.batch_sizes,
                                                      data_split_mode=args.split_mode,
                                                      shuffle=True)
    logging.info(
        'Dataset lenghts: training {}, validation {}, test {}'.format(len(train_loader.dataset),
                                                                      len(valid_loader.dataset),
                                                                      len(test_loader.dataset)))


    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    # if categorical is False, do regression
    model = MLP_YIELD(in_dim=args.in_dim,
                      fc_dims=args.fc_dims,
                      act_funs=[ACT_FUNCTIONS[act] for act in args.activation],
                      drop_rate=args.dropout_rate).to(device)
    model._init_parameters()


    # loss function
    loss_func = LOSS_FUNCTIONS[args.criterion]
    # accuracy function
    score_func = ACC_FUNCTIONS[args.accuracy]

    if args.training:
        logger.info('Start New Training Model...')
        logger.info('Model Parameters:')
        logging.info(json.dumps(vars(args)))
        logging.info('feature dim is {}'.format(args.in_dim))

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        # lr decay
        if args.use_lr_decay:
            SCHEDULERS = {
                'exponential': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),
                'cosine': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                'step': torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.97),
                'multistep': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.9)
            }
            scheduler = SCHEDULERS[args.lr_schedule]


        for epoch in range(args.epochs):
            start = time.time()
            model.train()

            train_losses = []
            ys_pred, ys_ = [], []
            for data, y_ in train_loader:
                y_preds = model(data.to(device))
                loss = loss_func(y_preds, y_.to(device))

                # L1 regularization
                reg_loss = 0.
                if args.regularization == 'L1':
                    for param in model.parameters():
                        reg_loss += torch.sum(torch.abs(param))
                if args.regularization == 'L2':
                    for param in model.parameters():
                        reg_loss += 0.5*torch.sum(param**2)

                if args.prior_adjust:

                    # loss_weights = get_loss_weights(y_, args.prior_bias).to(device)
                    loss_weights = get_loss_weights(y_,1.0).to(device)
                    loss *= loss_weights

                loss = loss.mean() + args.reg_lambda*reg_loss
                loss = (loss - args.flooding).abs() + args.flooding

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_thresh, norm_type=float('inf'))
                optimizer.step()
                train_losses.append(loss.cpu().detach().numpy())

                ys_.extend(y_)
                ys_pred.extend(y_preds.cpu().detach())

            score = score_func(torch.tensor(ys_pred), torch.tensor(ys_))
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print('Epoch{:>6d}, time={:>3.2f}s, lr={:>8.6f}, train_loss={:>6.5f}, R2={:>5.4f}'.format(epoch + 1, time.time() - start,
                                                                                         current_lr, np.mean(train_losses), score))

            # do lr schecule
            if args.use_lr_decay and (epoch+1)%10==0: scheduler.step()

            # plot train results
            if (epoch+1)%args.view_epochs==0:
                # plot training data
                plot_yield([y.numpy() for y in ys_], [y.numpy() for y in ys_pred],
                           title='Epoch {:<6d} Training '.format(epoch+1),
                           text='R2={:4.3f}'.format(score))
                logging.info(
                    'Epoch{:>6d}, time={:>3.2f}s, train_loss={:>6.5f}, R2={:>5.4f}'.format(epoch + 1, time.time() - start,
                                                                                           np.mean(train_losses), score))

            # validation
            if (epoch+1)%args.valid_epochs==0:
                model.eval()
                valid_losses = []
                ys_pred, ys_ = [], []
                with torch.no_grad():
                    for data, y_ in valid_loader:
                        y_preds = model(data.to(device))
                        loss = loss_func(y_preds, y_.to(device))
                        if args.prior_adjust:
                            loss_weights = get_loss_weights(y_, args.prior_bias).to(device)
                            loss *= loss_weights
                        loss = loss.mean()
                        valid_losses.append(loss.cpu().detach().numpy())

                        ys_.extend(y_)
                        ys_pred.extend(y_preds.cpu().detach())

                score = score_func(torch.tensor(ys_pred), torch.tensor(ys_))
                plot_yield([y.numpy() for y in ys_], [y.numpy() for y in ys_pred],
                           title='Epoch {:<6d} Validation '.format(epoch+1),
                           text='R2={:4.3f}'.format(score))

                print('Epoch{:>6d}, val_loss={:>6.5f}, R2={:>5.4f}'.format(epoch + 1, np.mean(valid_losses), score))
                logging.info('Epoch{:>6d}, val_loss={:>6.5f}, R2={:>5.4f}'.format(epoch + 1, np.mean(valid_losses), score))

        # save the final model
        state = {'net':model.state_dict(),
                 'optimizer':optimizer.state_dict(),
                 'epoch':epoch}
        torch.save(state, args.model_path)


    # always do test
    model = MLP_YIELD(in_dim=args.in_dim,
                      fc_dims=args.fc_dims,
                      act_funs=[ACT_FUNCTIONS[act] for act in args.activation])
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    with torch.no_grad():
        test_losses = []
        ys_pred, ys_ = [], []
        for data, y_ in test_loader:
            y_preds = model(data)
            loss = loss_func(y_preds, y_)
            if args.prior_adjust:
                loss_weights = get_loss_weights(y_, args.prior_bias)
                loss *= loss_weights
            loss = loss.mean()
            test_losses.append(loss.cpu().detach().numpy())

            ys_.extend(y_)
            ys_pred.extend(y_preds.cpu().detach())

        score = score_func(torch.tensor(ys_pred), torch.tensor(ys_))
        plot_yield([y.numpy() for y in ys_], [y.numpy() for y in ys_pred],
                   title='Test ',
                   text='R2={:4.3f}'.format(score))

        print('-'*50)
        logging.info('feature dim is {}'.format(args.in_dim))
        logger.info('test_loss={:>6.5f}, R2={:>5.4f}'.format(np.mean(test_losses), score))
        print('test_loss={:>6.5f}, R2={:>5.4f}'.format(np.mean(test_losses), score))
        print('-'*50)
        logger.info('')
        logger.info('-'*50)
        logger.info('')


