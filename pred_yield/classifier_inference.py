# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 13:11
# @Author  : TXH
# @File    : classifier_inference.py
# @Software: PyCharm

import os
import numpy as np
import torch
from torch import nn

from functools import partial
from torch.utils.data import Dataset, DataLoader
from utils import seed_torch

from utils import ACT_FUNCTIONS
from yield_model_dnn import MLP_YIELD_Classifier
from train_classifier import main
from MolCLRInfer import gen_molclr_feats
from MolCLRInfer import pca_decomposition as molclr_pca_decomposition
from gen_morgan import gen_morgan_feats
from gen_morgan import pca_decomposition as morgan_pca_decomposition

seed_torch(seed=2023)

class PhotoDataInferOne(Dataset):
    def __init__(self, reactions,
                 feat_ops='-,+,||,0.5,0.5',
                 feats_dict1=None,
                 feats_dict2=None,
                 mixing='add',
                 weights=[0.5, 0.5],
                 categorical=False):
        super(PhotoDataInferOne, self).__init__()
        self.reactions = reactions
        self.dataset = gen_model_data(self.reactions,
                                      feat_ops,
                                      feats_dict1,
                                      feats_dict2,
                                      mixing,
                                      weights)
        print('self.dataset: ', len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def get_feat_dim(self):
        return self.dataset[0][0].shape[-1]


def gen_model_data(raw_data,
                   operations='-,+,+,+',
                   molclr_feats=None,
                   morgan_feats=None,
                   mixing='add',
                   weights=[0.5, 0.5]):
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
    for i, rxn in enumerate(raw_data):
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
        # data.append([all_feats])
    # return data
    return all_feats


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


if __name__ == '__main__':
    yield_categories = {0: 'Low', 1: 'Medium', 2: 'High'}

    # input data: reactants, products, catalysts, reagents
    reactants = ['CCOC(=O)C1=C(C)NC(C)=C(C(=O)OCC)C1C(=O)NC(C)C(=O)OC', 'C=C(CBr)C(=O)OCC']
    products = ['C=C(CC(=O)NC(C)C(=O)OC)C(=O)OCC']
    catalysts = ['CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.FC1=C[C-](F)C(c2ccc(C(F)(F)F)cn2)=C=C1.FC1=C[C-](F)C(c2ccc(C(F)(F)F)cn2)=C=C1.F[P-](F)(F)(F)(F)F.[Ru+3]']
    reagents = ['OCC(F)(F)F']

    # raw reactions is a list of reaction; reaction is a list of different components
    reactions = [[reactants, products, catalysts, reagents]]

    # generate feats metadata for this reaction
    all_smiles = reactants+products+catalysts+reagents
    molclr_feats = gen_molclr_feats(all_smiles)
    molclr_feats = molclr_pca_decomposition(molclr_feats)
    morgan_feats = gen_molclr_feats(all_smiles)
    morgan_feats = morgan_pca_decomposition(morgan_feats)

    device = torch.device("cpu")
    args = main()
    # # prepare all datasets
    # dataset = PhotoDataInferOne(reactions,
    #                         feat_ops=args.feat_operations,
    #                         feats_dict1=molclr_feats,
    #                         feats_dict2=morgan_feats,
    #                         mixing=args.feat_mixing,
    #                         weights=args.feat_weights)
    #
    # test_loader = DataLoader(dataset, batch_size=1)
    # args.in_dim = dataset.get_feat_dim() # in feat dimension

    data = gen_model_data(reactions,
                          args.feat_operations,
                          molclr_feats,
                          morgan_feats,
                          args.feat_mixing,
                          args.feat_weights)[None, ...]
    args.in_dim = data.shape[-1]

    model = MLP_YIELD_Classifier(in_dim=args.in_dim,
                      fc_dims=args.fc_dims,
                      act_funs=[ACT_FUNCTIONS[act] for act in args.activation])

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    with torch.no_grad():
        # for data in test_loader:
        #     print(data[0].shape)
        y_pred = model(torch.tensor(data)).detach().numpy()
        print('Yield of this reaction: {}'.format(yield_categories[np.argmax(y_pred)]))




