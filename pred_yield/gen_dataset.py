# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 09:14
# @Author  : TXH
# @File    : gen_dataset.py
# @Software: PyCharm

# from use_molclr.mlp.models import NoEmbedAllMultiFeat
import numpy as np
import torch
import os
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from MolCLRInfer import gen_molclr_feats, molclr_pca
from gen_morgan import gen_morgan_feats, morgan_pca


def get_raw_dataset(in_file):
    f = open(in_file, 'r')
    reader = csv.reader(f)
    header = next(reader)
    r_index = header.index('Reactants_Smiles')
    p_index = header.index('Products_Smiles')
    c_index = header.index('Catalysts_Smiles')
    rg_index = header.index('Reagents_Smiles')
    y_index = header.index('Yield')
    spl_index = header.index('split')

    all_mols, cats, reags = [], [], []
    reactions = []
    for row in tqdm(reader, desc='processing all datasets...'):
        reactants = eval(row[r_index])
        products = eval(row[p_index])
        catalysts = eval(row[c_index])
        reagents = eval(row[rg_index])
        y = eval(row[y_index])
        spl = row[spl_index]
        rxn = ([reactants, products, catalysts, reagents], y)
        reactions.append(rxn)

    f.close()
    return reactions


def gen_model_data(raw_data,
                   operations='-,+,+,+',
                   molclr_feats=None,
                   morgan_feats=None,
                   mixing='add',
                   weights=[0.5, 0.5],
                   categorical_targets=False):
    '''
    raw_data:  tuple:([reactants smiles, products, catalysts, reagents], yield)
    operations: how to combine feats among reactants, products, catalysts, reagents
    molclr_feats: dict, key is molecule smiles, value is corresponding feat, 512 or pca reduced
    morgan_feats: dict, key is molecule smiles, value is corresponding feat, 512 or pca reduced
    mixing: how to combine molclr and morgan feats
    weights: weights for molclr and morgan feats to combine
    categorical_targets: do classification or regression
    :return: (np.array shape [N, d] or [N, 4, d], yield float or yield int)
    '''
    if mixing is 'no':
        feats = molclr_feats
    if mixing=='add':
        feats = {k:(weights[0])*molclr_feats[k] + weights[1]*morgan_feats[k] for k in
                 list(molclr_feats.keys())}
    if mixing=='cat':
        feats = {k: np.concatenate([weights[0]*molclr_feats[k], weights[1]*morgan_feats[k]], axis=-1) for k in
                 list(molclr_feats.keys())}

    feat_dim = list(feats.values())[0].shape[-1]

    data = []
    for i, (rxn, y) in enumerate(raw_data):
        if len(rxn[0])>0:
            reactants_feats = np.concatenate([feats[s] for s in rxn[0]]).mean(axis=0, keepdims=True)
        else:
            reactants_feats = np.zeros([1, feat_dim], dtype=np.float32)
        if len(rxn[1]) > 0:
            products_feats = np.concatenate([feats[s] for s in rxn[1]]).mean(axis=0, keepdims=True)
        else:
            products_feats = np.zeros([1, feat_dim], dtype=np.float32)
        if len(rxn[2]) > 0:
            catalysts_feats = np.concatenate([feats[s] for s in rxn[2]]).mean(axis=0, keepdims=True)
        else:
            catalysts_feats = np.zeros([1, feat_dim], dtype=np.float32)
        if len(rxn[3]) > 0:
            reagents_feats = np.concatenate([feats[s] for s in rxn[3]]).mean(axis=0, keepdims=True)
        else:
            reagents_feats = np.zeros([1, feat_dim], dtype=np.float32)

        all_feats = np.concatenate([reactants_feats, products_feats, catalysts_feats, reagents_feats]) # [4, dim]
        if operations!='no':
            all_feats = gen_rxn_feats(all_feats, operations) # do some mixing.

        if categorical_targets:
            if y<0.40: y = 0 # 1884
            if y>=0.70: y = 2 # 4764
            if y>=0.4 and y<0.70 : y = 1 # 4421
            data.append((all_feats, y))
        else:
            data.append((all_feats, np.array(y, dtype=np.float32).reshape(1,)))
    return data


def gen_rxn_feats(feats, operations='-,+,+,+'):
    '''
    :param feats: feats, np.array, shape=[4, 512]
    :param operation: 4 signs for each column
    :return:
    '''
    # generate operations in floats
    if operations=='-,+,+,+':
        ops = [1. if o=='+' else -1. for o in operations.split(',')]
        assert len(ops)==feats.shape[0], 'Feats dimension 0 not equal to ops number'
        final_feats = np.zeros_like(feats[0:1]) # 1*dim
        for i, o in enumerate(ops):
            final_feats += o*feats[i:i+1]
    if operations=='-,+,||,0.5,0.5':
        x1 = feats[1:2]-feats[0:1] # product-reactant
        x2 = np.mean(feats[2:4], axis=0, keepdims=True) # mean value of reagent and catalysts
        final_feats = np.concatenate([x1, x2], axis=-1) # 2*dim
    return final_feats.squeeze()


class PhotoDatasets(Dataset):
    def __init__(self, file,
                 feat_ops='-,+,||,0.5,0.5',
                 feats_dict1=None,
                 feats_dict2=None,
                 mixing='add',
                 weights=[0.5, 0.5],
                 categorical=False):
        super(PhotoDatasets, self).__init__()

        self.reactions = get_raw_dataset(file)

        self.dataset = gen_model_data(self.reactions,
                                      feat_ops,
                                      feats_dict1,
                                      feats_dict2,
                                      mixing,
                                      weights,
                                      categorical_targets=categorical)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)

    def get_feat_dim(self):
        return self.dataset[0][0].shape[-1]


def split(dataset, mode='rule', ratios=[0.8, 0.1, -1]):

    if mode=='random':
        N = dataset.__len__()
        n_train, n_valid = int(ratios[0]*dataset.__len__()), int(ratios[1]*dataset.__len__())
        n_test = N - n_train - n_valid
        train_db, valid_db, test_db = random_split(dataset, [n_train, n_valid, n_test])

    if mode == 'rule':
        splits = np.load('../data/split_ids.npy', allow_pickle=True).item()
        train_db = Subset(dataset, splits['train'])
        valid_db = Subset(dataset, splits['valid'])
        test_db = Subset(dataset, splits['test']) # training 7781, validation 1640, test 1648
    return train_db, valid_db, test_db


def loaders(dataset, ratios=[0.8, 0.1, -1],
            batch_sizes=[512,64,64],
            data_split_mode='rule',
            shuffle=True):

    train_db, valid_db, test_db = split(dataset, mode=data_split_mode, ratios=ratios)
    train_loader = DataLoader(train_db, batch_size=batch_sizes[0], shuffle=shuffle)
    valid_loader = DataLoader(valid_db, batch_size=batch_sizes[1], shuffle=shuffle)
    test_loader = DataLoader(test_db, batch_size=batch_sizes[2], shuffle=shuffle)
    return train_loader, valid_loader, test_loader


def extractFC2(path):
    dct = torch.load(path)
    train_data = dct['train']
    vali_data = dct['vali']
    test_data = dct['test']
    device = 'cpu'
    train_set = MultifeatCatRegDataset(train_data.iloc[:, :4], train_data.iloc[:, 4], device)
    vali_set = MultifeatCatRegDataset(vali_data.iloc[:, :4], vali_data.iloc[:, 4], device)
    test_set = MultifeatCatRegDataset(test_data.iloc[:, :4], test_data.iloc[:, 4], device)
    model_path = 'mlp/no_embed_all_multifeat/optuna_res/' \
                 'no_embed-rule-dim_reduce-pgd-linux/drop280691'
    with open(f'{model_path}/config.yml', 'r') as hyper_f:
        hypers = yaml.safe_load(hyper_f)
    if hypers['pgd']:
        dct['feat_molclr'].requires_grad_()
        dct['feat_morgan'].requires_grad_()
    net = NoEmbedAllMultiFeat(
        rp_dim=100, out_dim=1,
        dim_reduce=hypers['dim_reduce'],
        feat_molclr=dct['feat_molclr'],
        feat_morgan=dct['feat_morgan'],
        w_molclr=hypers['w_molclr'],
        drop=hypers['drop'], act=hypers['act']
    ).to(device)
    net.load_state_dict(torch.load(
        f'{model_path}/best_model.pth', map_location=device))
    net.eval()
    eval_train_dataloader = DataLoader(
        train_set, batch_size=min(1024, len(train_set)), shuffle=False
    )
    eval_vali_dataloader = DataLoader(
        vali_set, batch_size=min(1024, len(vali_set)), shuffle=False
    )
    test_dataloader = DataLoader(
        test_set, batch_size=min(1024, len(train_set)), shuffle=False
    )
    with torch.no_grad():
        t_train_all, pred_train_all = [], []
        t_vali_all, pred_vali_all = [], []
        t_test_all, pred_test_all = [], []
        for feature, t_train in eval_train_dataloader:
            t_train_all.append(t_train)
            pred_train_all.append(net(feature, to_ml=True))
        for feature, t_vali in eval_vali_dataloader:
            t_vali_all.append(t_vali)
            pred_vali_all.append(net(feature, to_ml=True))
        for feature, t_test in test_dataloader:
            t_test_all.append(t_test)
            pred_test_all.append(net(feature, to_ml=True))
    pred_train_all = torch.vstack(pred_train_all).numpy()
    pred_vali_all = torch.vstack(pred_vali_all).numpy()
    pred_test_all = torch.vstack(pred_test_all).numpy()
    t_train_all = torch.vstack(t_train_all).squeeze().numpy()
    t_vali_all = torch.vstack(t_vali_all).squeeze().numpy()
    t_test_all = torch.vstack(t_test_all).squeeze().numpy()
    # pred_train_vali_all = np.vstack((pred_train_all, pred_vali_all))
    # t_train_vali_all = np.r_[t_train_all, t_vali_all]

    # scaler = StandardScaler()
    # pred_train_all = scaler.fit_transform(pred_train_all)
    # pred_vali_all = scaler.transform(pred_vali_all)
    # pred_test_all = scaler.transform(pred_test_all)
    np.savez(
        'yield_data/split_data.npz',
        feat_train=pred_train_all, feat_vali=pred_vali_all, feat_test=pred_test_all,
        t_train=t_train_all, t_vali=t_vali_all, t_test=t_test_all,
        # feat_train_vali=pred_train_vali_all, t_train_vali=t_train_vali_all
    )

def main():
    from argparse import ArgumentParser
    from utils import simple_check_feats

    parser = ArgumentParser()
    parser.add_argument('--file_dir', type=str, default='../data/scifinder_clean.csv', help='Path to raw_data')

    # use following 7 parameters to tune feats dimensions
    parser.add_argument('--regenerate_molclr', type=bool, default=False, help='regenerate molclr 512 feats') # fixed
    parser.add_argument('--molclr_do_pca', type=bool, default=True, help='dim reduction')
    parser.add_argument('--molclr_pca_out', type=int, default=96, help='molclr pca out dim')

    parser.add_argument('--regenerate_fp', type=bool, default=True, help='regenerate fp feats')
    parser.add_argument('--fpsize', type=int, default=512, help='morgan_fp_size') # variable
    parser.add_argument('--morgan_do_pca', type=bool, default=True, help='dim reduction')
    parser.add_argument('--morgan_pca_out', type=int, default=96, help='morgan pca out dim')

    # use following 3 parameters to tune feats combination
    parser.add_argument('--feat_operations', type=str, default='-,+,||,0.5,0.5',
                        help='no, no operations; -,+,||,0.5,0.5 or -,+,+,+ for reactants, products, catalysts, reagents')
    parser.add_argument('--feat_mixing', type=str, default='add',
                        help='no, only the first one; add: w0*molclr+w1*morgan; cat: [w0*molclr||w1*morgan]')

    parser.add_argument('--feat_weights', type=float, default=[0.5, 0.5],
                        help='Feat weights. If mixing is add, then feat=w0*molclr+w1*morgan; if mixing is concat, feat=[w0*molclr||w1*morgan]. w0+w1 is not restricted to be 1')

    # use following 3 parameters to prepare datasets for train, validation, test
    parser.add_argument('--split_mode',
                        type=str,
                        default='rule',
                        help='random or rule. how to split the dataset. rule guarantees molecules in test dataset have been seen during training')
    parser.add_argument('--split_ratios',
                        type=list,
                        default=[0.7, 0.15, -1],
                        help='ratios of training, validation, and test set')
    parser.add_argument('--batch_sizes',
                        type=list,
                        default=[1024, 64, 64],
                        help='batch_sizes for training, validation and testing')

    args = parser.parse_args()

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
        print('Generating fp features...')  # always. this is quick
        morgan_feats = gen_morgan_feats(in_file='../data/scifinder_clean_all_mols.csv',
                                        out_file='../data/feats/all_mol_morgan_feats_{}.npy'.format(args.fpsize),
                                        fpsize=args.fpsize)
    else:
        print('Loading precalculated fp features...')
        morgan_feats = np.load('../data/feats/all_mol_morgan_feats_{}.npy'.format(args.fpsize),
                               allow_pickle=True).item()

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
    test_mol = 'CC(=O)c1ccc(Br)cc1'
    simple_check_feats(molclr_feats[test_mol])
    simple_check_feats(morgan_feats[test_mol])


    # prepare all datasets
    dataset = PhotoDatasets(args.file_dir,
                            feat_ops=args.feat_operations,
                            feats_dict1=molclr_feats,
                            feats_dict2=morgan_feats,
                            mixing=args.feat_mixing,
                            weights=args.feat_weights,
                            categorical=True)
    args.in_dim = dataset.get_feat_dim()  # in feat dimension
    print('Dataset length: ', dataset.__len__())
    print('feature dim is {}'.format(args.in_dim))
    train_loader, valid_loader, test_loader = loaders(dataset,
                                                            ratios=args.split_ratios,
                                                            batch_sizes=args.batch_sizes,
                                                            data_split_mode=args.split_mode,
                                                            shuffle=True)
    print('Dataset lenghts: training {}, validation {}, test {}'.format(len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset)))

    return train_loader, valid_loader, test_loader



if __name__ == '__main__':
    import os
    import time

    start = time.time()
    train_loader, valid_loader, test_loader = main()

    print('{:4.2f}s spent to prapare all data...'.format(time.time()-start))

    for d, y in test_loader:
        print(d.shape)
        # print(y)
        # print(y)
        # break


