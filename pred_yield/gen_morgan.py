import joblib
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
from tqdm import tqdm
import csv
from sklearn.decomposition import PCA

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
    # print('len of dataset_test:', len(smiles_data))
    # print('Unable to convert smiles to mol quantity:', len(noneSmiles))
    # print(noneSmiles)
    return smiles_data

# def morgan512(path):
#     mol_molclr = torch.load(path)
#     mol_morgan = {}
#     for chem in tqdm(mol_molclr.keys()):
#         m = Chem.MolFromSmiles(chem)
#         mol_morgan[chem] = [int(each) for each in AllChem.GetMorganFingerprintAsBitVect(m, 2).ToBitString()]
#     ar_fp = np.asarray(list(mol_morgan.values()))
#
#     # pca = PCA(n_components=512)
#     # ar_fp_pca = pca.fit_transform(ar_fp.astype(np.float32))  # 注意这里要获取PCA变换矩阵，用于推断时对新进来的特征降维
#     # joblib.dump(pca, 'pca.m')
#     pca = joblib.load('pca.m')
#     ar_fp_pca = pca.transform(ar_fp)
#     fps_pca = torch.split(torch.from_numpy(ar_fp_pca).type(torch.FloatTensor),
#                           split_size_or_sections=1, dim=0)
#     del ar_fp
#     gc.collect()
#     pca_morgan = dict(zip(mol_morgan, fps_pca))
#     torch.save(pca_morgan, 'yield_data/morgan512.pt')

def get_fp(smiles, fpsize=2048):
    if isinstance(smiles, str):
        smiles = [smiles]
    morgan_dict = {}
    for s in tqdm(smiles, desc='Processing all smiles...'):
        mol = Chem.MolFromSmiles(s)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fpsize)
        array = np.zeros((1,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, array)
        morgan_dict[s] = array.reshape(1, -1)
    return morgan_dict

    # pca = PCA(n_components=512)
    # ar_fp_pca = pca.fit_transform(ar_fp.astype(np.float32))  # 注意这里要获取PCA变换矩阵，用于推断时对新进来的特征降维
    # joblib.dump(pca, 'pca.m')

def gen_morgan_feats(smiles=None,
                     fpsize=512,
                     in_file=None,
                     out_file=None):
    ## generate dataset
    if smiles is None:
        smiles = read_smiles(in_file)
    if len(smiles)==0:
        return None

    morgan_dict = get_fp(smiles, fpsize=fpsize)
    if out_file is not None:
        np.save(out_file, morgan_dict)
    print('Total smiles: ', len(list(morgan_dict.keys())))
    return morgan_dict


def morgan_pca(feats_dict, out_dim=96, save_model_path=None):
    feats = np.array(list(feats_dict.values())).squeeze()
    pca = PCA(n_components=out_dim, whiten=True)
    feats = pca.fit_transform(feats)
    if save_model_path is not None:
        from joblib import dump
        dump(pca, save_model_path)

    new_feats_dict = {k: m[None, ...] for k, m in zip(list(feats_dict.keys()), feats)} # care the dim
    return new_feats_dict


def pca_decomposition(feats_dict, model='./model/morgan_pca.pkl'):
    from joblib import load
    pca_model = load(model)
    new_feats_dict = {k: pca_model.transform(v) for k, v in feats_dict.items()}
    return new_feats_dict


if __name__ == '__main__':

    do_pca = True
    fpsize = 512
    # whole datasets
    morgan_feats = gen_morgan_feats(in_file='../data/scifinder_clean_all_mols.csv',
                                   out_file='../data/feats/all_mol_morgan_feats_{}.npy'.format(fpsize),
                                   fpsize=fpsize)

    if do_pca:
        feats_dict = morgan_pca(morgan_feats,
                                out_dim=96,
                                save_model_path='./model/morgan_pca.pkl')

    ## inference
    smiles = ['c1cc2ccc3cccc4ccc(c1)c2c34',
              'OC(=O)[C@@H]1CSC(=N1)C1=NC2=CC=C(O)C=C2S1',
              '[H][C@@](O)(COCC1=CC=C(OC)C=C1)[C@@]([H])(O)COC1=CC=C(OC)C=C1',
              'CC(=O)OC=C']
    morgan_feats = gen_morgan_feats(smiles=smiles, fpsize=fpsize)

    if do_pca:
        morgan_feats = pca_decomposition(morgan_feats)


