import torch
import transformers
import pandas as pd
import time
import dgl
from rdkit import Chem
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from dgllife.utils import smiles_to_bigraph,CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import gc

class MolecularDataset(Dataset):
    """自定义 PyTorch Dataset，用于存储化合物和蛋白质特征"""
    def __init__(self, molecule_data):
        self.molecule_data = molecule_data

    def __len__(self):
        return len(self.molecule_data)

    def __getitem__(self, idx):
        return self.molecule_data[idx]


    @staticmethod
    def collate_fn(batch):
        """批量处理 DataLoader 提供的数据"""
        graphs = [item['graph'] for item in batch]
        batched_graph = dgl.batch(graphs)  # 合并 DGLGraph

        node_feats = torch.cat([item['node_feats'] for item in batch], dim=0)  # [batch_size, 节点特征维度]
        edge_feats = torch.cat([item['edge_feats'] for item in batch], dim=0)  # [batch_size, 边特征维度]
        protein_feats = pad_sequence([item['protein_feats'] for item in batch], batch_first=True, padding_value=0)


        adj_matrices = []
        max_rows = max([item['adj_matrix'].shape[0] for item in batch])  # 最大行数
        max_cols = max([item['adj_matrix'].shape[1] for item in batch])  # 最大列数
        # 将最大行列数存储在 max_size
        max_size = (max_rows, max_cols)
        for item in batch:
            adj_matrix = item['adj_matrix']

            rows, cols = adj_matrix.shape

            # 计算需要填充的行和列
            padding_rows = max_size[0] - rows
            padding_cols = max_size[1] - cols

            # 使用 0 填充
            padded_matrix = F.pad(adj_matrix, (0, padding_cols, 0, padding_rows), value=0)
            adj_matrices.append(padded_matrix)
        adj_matrix=torch.stack(adj_matrices, dim=0)

        return batched_graph, node_feats, edge_feats, protein_feats,adj_matrix

    @staticmethod
    def get_protein_features(prosequences_list,i,tokenizer,model,device,pro_feats_dic):

        #处理蛋白序列
        protein_sequence = prosequences_list[i]
        protein_tokenizer = tokenizer(protein_sequence, return_tensors="pt").to(device)
        #提取蛋白特征
        with torch.no_grad():
            outputs = model(**protein_tokenizer)
            protein_feats = outputs.last_hidden_state.mean(dim=0)  # 取平均作为蛋白特征
            protein_feats=protein_feats.to(torch.device('cpu'))
            pro_feats_dic[protein_sequence] = protein_feats

        return protein_feats,pro_feats_dic       #（L，C）序列个数，隐藏层维度

    @staticmethod
    def loading_data(seq,smiles, device):
        """
        从内存列表加载数据，生成 MolecularDataset
        :param smiles_list: list[str]，分子 SMILES
        :param pro_sequences_list: list[str]，对应的蛋白序列
        :param device: torch.device
        :return: MolecularDataset 实例
        """
        atom_featurizer = CanonicalAtomFeaturizer()  # 分子特征提取器
        bond_featurizer = CanonicalBondFeaturizer()

        # 加载 ESM 蛋白模型
        model_path = "/ldata/databases/folding/esm/esm2_t33_650M_UR50D.pt"
        tokenizer_path = "tokenizer"

        '''huggingface下载模型'''
        # model_name = "facebook/esm2_t30_150M_UR50D"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name).to(device)

        '''本地加载模型'''

        # model_path = "esm2_t33_650M_UR50D.pt"
        # tokenizer_path ="tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = AutoConfig.from_pretrained(tokenizer_path)
        model = AutoModel.from_config(config)
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.to(device).eval()

        atom_featurizer = CanonicalAtomFeaturizer()
        bond_featurizer = CanonicalBondFeaturizer()

        # 分子图
        graph = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
        node_feats = graph.ndata['h']
        edge_feats = graph.edata['e']
        mol = Chem.MolFromSmiles(smiles)
        adj_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol)).float()

        # 蛋白特征
        pro_feats_dic = {}
        if seq not in pro_feats_dic:
            protein_tokenizer = tokenizer(seq, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**protein_tokenizer)
                protein_feats = outputs.last_hidden_state.mean(dim=0).to(torch.device('cpu'))
            pro_feats_dic[seq] = protein_feats
        else:
            protein_feats = pro_feats_dic[seq]

        molecule_data = [{
            'graph': graph,
            'node_feats': node_feats,
            'edge_feats': edge_feats,
            'protein_feats': protein_feats,
            'adj_matrix': adj_matrix
        }]

        return MolecularDataset(molecule_data)





# device = torch.device("cpu")
# print('using device:{}'.format(device))

# file_path='/Users/zephyr/Documents/PycharmProjects/KRASG12D_Inhibitors/Compound_activity_prediction/DataSet/csv/final_decoys.csv'
# loading_data=loading_data(file_path,device)
