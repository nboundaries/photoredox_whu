目的：提取MolCLR分子表征
**模型环境：Remote Python 3.7.13 (sftp://common@192.168.0.61:22/home/common/anaconda3/envs/MolCLR_weilu/bin/python)**

ckpt
    pretrained_gcn文件夹   GCN的预训练模型参数
    pretrained_gin文件夹   GIN的模型预训练参数

data 需要提取特征的文件夹，当需要提取特征时，请把分子处理成smiles的集合set，后缀为.pkl，放到data文件夹下。
例如，data/catalyst_reagent_set.pkl 其中 catalyst_reagent_set.pkl是分子表征的集合
运行之后生成catalyst_reagent_set.pt, 这是一个字典类型，key 对应 smiles, value 对应 MolCLR特征

dataset_test
    dataset_test.py 数据集处理（不建议修改）

models
    gcn_finetune.py gcn模型
    ginet_finetune.py gin模型

config.yaml 模型的配置文件，因为是导入是模型的参数，所以一些配置参数不可以修改
    其中可以修改的为batch_size、fine_tune_from、gpu、pool，建议只修改gpu为 gpu: cuda:7,其余的修改没有意义（建议不修改）

main.py 主函数
将你的.pkl文件放到/data目录下，然后将filename全名替换成你的文件名，点击运行即可，生成的特征存放在/data目录下