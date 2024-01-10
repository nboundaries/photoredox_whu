import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import MultiLabelBinarizer


def processSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    fp_array = np.zeros((0,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    # print(fp_array)
    return fp_array


def process_rrp(reactant=None, product=None, reagent=None):
    reactants = processSmiles('.'.join(eval(reactant)))
    reagents = processSmiles('.'.join(eval(reagent)))
    products = processSmiles('.'.join(eval(product)))

    return reactants, products, reagents


def process_rp(reactant=None, product=None):
    reactants = processSmiles('.'.join(eval(reactant)))
    products = processSmiles('.'.join(eval(product)))

    return reactants, products

#if use rpr model
def predict_rpr(reactants="", products="", reagents="", model_file="RF_model_rpr.pkl"):
    model = joblib.load(model_file)
    r, p, rg = process_rrp(reactants, products, reagents)
    r, p, rg = np.array([r]), np.array([p]), np.array([rg])
    X = np.concatenate([r, p, rg], axis=1)

    labels = np.load('catalysts_labels.npy', allow_pickle=True).tolist()
    labels = [[l] for l in labels]
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(labels)
    prob = model.predict_proba(X)
    pp = np.zeros([len(labels), 2], dtype=np.float32)
    for i, p in enumerate(prob):
        pp[i:i + 1, :] = np.array([p[0][0], 1 - p[0][0]])
    mols = mlb.inverse_transform(model.predict(X))
    return mols, pp

#if use rp model
def predict_rp(reactants="", products="", model_file="RF_model_rp.pkl"):
    model = joblib.load(model_file)
    r, p = process_rp(reactants, products)
    r, p = np.array([r]), np.array([p])
    X = np.concatenate([r, p], axis=1)

    labels = np.load('catalysts_labels.npy', allow_pickle=True).tolist()
    labels = [[l] for l in labels]
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(labels)
    prob = model.predict_proba(X)
    pp = np.zeros([len(labels), 2], dtype=np.float32)
    for i, p in enumerate(prob):
        pp[i:i + 1, :] = np.array([p[0][0], 1 - p[0][0]])
    mols = mlb.inverse_transform(model.predict(X))
    return mols, pp


if __name__=='__main__':
    file_in = r"your_file_input_path" # reactants/products/reagents SMILES splited by '.', such as 'CO.CC#N'
    file_out = open(r"your_file_output_path", 'w', newline='')

    writer = csv.writer(file_out)
    writer.writerow(['id', 'reactant', 'product', 'reagent', 'catalyst'])

    f = open(file_in, 'r')
    data = csv.reader(f)
    header = next(data)
    id_index = header.index('id')
    r_index = header.index('reactant')
    p_index = header.index('product')
    rg_index = header.index('reagent')
    # yield_index = header.index('YIELD')
    for k, row in enumerate(data):
        id = row[id_index]
        # print('Processing ', id)

        reactants = '[\''+row[r_index]+'\']'
        products = '[\''+row[p_index]+'\']'

        # # use r+p
        # catalysts, pp = predict_rp(reactants, products, model_file="DT_model_rp.pkl")
        # writer.writerow([id, reactants, products, catalysts])

        # use r+p+r
        reagents = '[\''+row[rg_index]+'\']'
        catalysts, pp = predict_rpr(reactants, products, reagents, model_file="RF_model_rpr.pkl")
        writer.writerow([id, reactants, products, reagents, catalysts])

        print('-'*30)
        print('Sample{}'.format(k))
        for i, p in enumerate(pp):
            if pp[i][1]>=0.1:
                print('cat_id={:d}, p={:.3f}'.format(i, p[1]))

    f.close()
    print('Job completed!')
