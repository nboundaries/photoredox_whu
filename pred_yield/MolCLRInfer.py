# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 14:29
# @Author  : TXH
# @File    : MolCLRInfer.py
# @Software: PyCharm

import numpy as np
import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import yaml
from tqdm import tqdm
import csv

from sklearn.decomposition import PCA


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def read_smiles(data_path):
    smiles_data = []
    noneSmiles = []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in (enumerate(csv_reader)):
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol != None:
                smiles_data.append(smiles)
            else:
                noneSmiles.append(smiles)
    print('len of dataset_test:', len(smiles_data))
    print('Unable to convert smiles to mol quantity:', len(noneSmiles))
    print(noneSmiles)
    return smiles_data


class InferDataSet(Dataset):
    def __init__(self, smiles):
        super(InferDataSet, self).__init__()
        self.smiles = smiles

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles[index])
        mol = Chem.AddHs(mol)

        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    smiles=self.smiles[index]
                    )
        return data

    def __len__(self):
        return len(self.smiles)


class MolCLRInferOne(object):
    def __init__(self, dataset, config):
        super(MolCLRInferOne, self).__init__()
        self.config = config
        self.device = self._get_device()
        self.dataset = dataset
        print('Using GIN model.')
        from models.ginet_finetune import GINet
        self.model = GINet(**self.config["model"]).to(self.device)
        self.model = self._load_pre_trained_weights(self.model)
        self.model.eval()
        print('MolCLR model: ', self.model)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)
        return device

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
        return model

    def gen_feats(self, batch_size_=1):
        data_loader = DataLoader(self.dataset, batch_size=batch_size_, drop_last=False)
        # extract MolCLR feats
        all_feats = {}
        for data in tqdm(data_loader, desc='Extracting MolCLR features...'):
            data = data.to(self.device)
            with torch.no_grad():
                h, _ = self.model(data)
            all_feats[data.smiles[0]] = h.to('cpu').numpy()
        return all_feats


def gen_molclr_feats(smiles=None,
                     in_file=None,
                     out_file=None,
                     config_file='./config.yaml'):
    config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    ## generate dataset
    if smiles is None:
        smiles = read_smiles(in_file)
    if len(smiles)==0:
        return None

    dataset = InferDataSet(smiles)
    MolCLRInfer = MolCLRInferOne(dataset, config)
    molclr_feats = MolCLRInfer.gen_feats()
    if out_file is not None:
        np.save(out_file, molclr_feats)
    print('Total smiles: ', len(list(molclr_feats.keys())))

    return molclr_feats


def molclr_pca(feats_dict, out_dim=96, save_model_path=None):
    feats = np.array(list(feats_dict.values())).squeeze()
    pca = PCA(n_components=out_dim, whiten=True)
    feats = pca.fit_transform(feats)
    if save_model_path is not None:
        from joblib import dump
        dump(pca, save_model_path)

    new_feats_dict = {k: m[None, ...] for k, m in zip(list(feats_dict.keys()), feats)} # care the dim
    return new_feats_dict


def pca_decomposition(feats_dict, model='./model/molclr_pca.pkl'):
    from joblib import load
    pca_model = load(model)
    new_feats_dict = {k: pca_model.transform(v) for k, v in feats_dict.items()}
    return new_feats_dict



if __name__ == '__main__':

    do_pca = True

    # # whole dataset
    # molclr_feats = gen_molclr_feats(in_file='../data/scifinder_clean_all_mols.csv',
    #                                 out_file='../data/feats/all_mol_molclr_feats_512.npy',
    #                                 config_file='./config.yaml')
    # if do_pca:
    #     feats_dict = molclr_pca(molclr_feats,
    #                             out_dim=96,
    #                             save_model_path='./model/molclr_pca.pkl')
    #

    ## inference
    smiles = ['c1cc2ccc3cccc4ccc(c1)c2c34',
              'OC(=O)[C@@H]1CSC(=N1)C1=NC2=CC=C(O)C=C2S1',
              '[H][C@@](O)(COCC1=CC=C(OC)C=C1)[C@@]([H])(O)COC1=CC=C(OC)C=C1',
              'CC(=O)OC=C',
              'C=C(CC(=O)NC(C)C(=O)OC)C(=O)OCC']
    molclr_feats = gen_molclr_feats(smiles=smiles)

    if do_pca:
        molclr_feats = pca_decomposition(molclr_feats)

    print(molclr_feats.keys())

