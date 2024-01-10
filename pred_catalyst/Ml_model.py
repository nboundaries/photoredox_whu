import csv
import numpy as np
import pandas as pd
import joblib
import os

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold

def getMorgan(smiles, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
    fp_array = np.zeros((0,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    return fp_array

def get_data(data_path):
    reactants = []
    reagents = []
    products = []
    catalysts = []
    cat_labels = []
    reg_labels = []
    with open(data_path, encoding='gbk') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            catalyst = '.'.join(list(set(eval(row['Catalysts_Smiles']))))
            reagent = '.'.join(list(set(eval(row['Reagents_Smiles']))))
            if reagent == '' or catalyst == '':  # 丢弃空数据
                continue
            reactant = '.'.join(list(set(eval(row['Reactants_Smiles']))))
            product = '.'.join(list(set(eval(row['Products_Smiles']))))

            reactants.append(getMorgan(reactant))
            reagents.append(getMorgan(reagent))
            products.append(getMorgan(product))
            catalysts.append(getMorgan(catalyst))
            # labels
            cat_label = list(set(eval(row['Catalysts_Smiles'])))
            cat_labels.append(cat_label)
            reg_label = list(set(eval(row['Reagents_Smiles'])))
            reg_labels.append(reg_label)

    return (np.array(reactants), np.array(products), np.array(reagents), np.array(catalysts)), (cat_labels, reg_labels)

class MLmodel(object):
    def __init__(self, data_file, model_flag, inputs):
        super(MLmodel, self).__init__()
        self.model_flag = model_flag
        self.inputs = inputs
        print('1. Data processing...')
        (reactants, products, reagents, _), (catalysts, _) = get_data(data_file)

        # labels
        self.mlb = MultiLabelBinarizer()
        Y = self.mlb.fit_transform(catalysts)
        np.save('./catalysts_labels.npy', self.mlb.classes_)
        print('Number of catalysts: ', len(self.mlb.classes_))

        # inputs
        if self.inputs == 'rp':  
            X = np.concatenate([reactants, products], axis=1)
        if self.inputs == 'rpr':  
            X = np.concatenate([reactants, products, reagents], axis=1)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X,
                                                                                Y,
                                                                                test_size=0.2,
                                                                                random_state=42)

        # models
        if self.model_flag=='DT':
            self.model = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=10,
                                       min_samples_split=2,
                                       min_samples_leaf=1)
        if self.model_flag=='RF':
            self.model = RandomForestClassifier(n_estimators=12,
                                    criterion='entropy',
                                    max_depth=40,
                                    random_state=42)


    def train_model(self):
        print('2. Training model...')
        self.model.fit(self.X_train, self.Y_train)
        scores = cross_val_score(self.model, self.X_train, self.Y_train, scoring='f1_samples', cv=KFold(10))
        print("Cross validation scores:{}".format(scores))
        print("Mean cross validation score:{:2f}".format(scores.mean()))

        joblib.dump(self.model, self.model_flag+'_model_'+self.inputs+'.pkl')
        print('3. Model saved.')

    def test_model(self):
        print('Testing model...')
        labels = np.load('catalysts_labels.npy', allow_pickle=True).tolist()
        labels = [[l] for l in labels]
        self.mlb.fit_transform(labels)

        model = joblib.load(self.model_flag+'_model_'+self.inputs+'.pkl')
        # this is the binary classification probabilities for each catalyst label
        prob = model.predict_proba(self.X_test)
        pp = np.zeros([len(labels), 2], dtype=np.float32)
        for i, p in enumerate(prob):
            pp[i:i + 1, :] = np.array([p[0][0], 1 - p[0][0]])

        Y_pred = model.predict(self.X_test)
        f1 = f1_score(self.Y_test, Y_pred, average="samples")

        # print true and predicted smiles
        Y_test_class = self.mlb.inverse_transform(self.Y_test)
        Y_pred_class = self.mlb.inverse_transform(Y_pred)
        # for x, y in zip(Y_test_class, Y_pred_class):
        #     print('true:', x)
        #     print('pred:', y)
        #     print()
        print('Test f1_score:', f1)

if __name__ == '__main__':
    data_file = r"..\data\scifinder_clean.csv"
    ml = MLmodel(data_file, 'RF', 'rpr')
    ml.train_model()
    ml.test_model()
