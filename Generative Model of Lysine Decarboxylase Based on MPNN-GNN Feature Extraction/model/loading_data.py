import torch
import transformers
import pandas as pd
import time
import dgl
import numpy as np
from rdkit import Chem
from torch.utils.data import Sampler
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from dgllife.utils import smiles_to_bigraph,CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# from KRASG12D_Inhibitors.Compound_activity_prediction.model_副本.loading_data import MolecularDataset


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
        # value = torch.stack([torch.tensor([item['value']], dtype=torch.float32) for item in batch], dim=0) # [batch_size]
        value = torch.tensor([item['value'] for item in batch], )
        # adj_matrix = pad_sequence([item['adj_matrix'] for item in batch], batch_first=True, padding_value=0)

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

        return batched_graph, node_feats, edge_feats, protein_feats, value,adj_matrix

    @staticmethod
    def get_protein_features(prosequences_list,i,tokenizer,model,device,pro_feats_dic):

        #处理蛋白序列
        protein_sequence = prosequences_list[i]
        protein_tokenizer = tokenizer(protein_sequence, return_tensors="pt").to(device)
        #提取蛋白特征
        with torch.no_grad():
            outputs = model(**protein_tokenizer)
            # protein_feats = outputs.last_hidden_state[0]   # 取平均作为蛋白特征
            protein_feats = outputs.last_hidden_state.mean(dim=0)   #pool
            protein_feats=protein_feats.to(torch.device('cpu'))
            pro_feats_dic[protein_sequence] = protein_feats
            # print('protein_feats:',protein_feats.shape)
        return protein_feats,pro_feats_dic       #（L，C）序列个数，隐藏层维度

    @staticmethod
    def mutation( mutation, label, protein_feats,seq_len):
        AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

        def aa_onehot(aa):
            """将氨基酸单个字母编码为 one-hot（20维）"""
            onehot = np.zeros((20,), dtype=np.float32)
            if aa in AMINO_ACIDS:
                onehot[AMINO_ACIDS.index(aa)] = 1.0
            return onehot

        def parse_mutation(mutation_str):
            """解析突变字符串，例如 a879l → ('A', 879, 'L')"""
            orig = mutation_str[0].upper()
            target = mutation_str[-1].upper()
            pos = int(mutation_str[1:-1])
            return orig, pos, target

        def mutation_feature_vector(mutation, seq_len):
            """构造包含原始aa,突变aa,位置的特征向量"""

            dic = {}
            mutation = mutation.split(',')
            for mu in mutation:
                orig, pos, target =parse_mutation(mu)
                orig_vec = aa_onehot(orig)
                target_vec = aa_onehot(target)
                dic[pos] = np.concatenate([orig_vec, target_vec])
            return dic  # 共40维

            '''输出的序列特征为681维'''
        if label==0 or label==1:
            L = protein_feats.size(0)  # 获取序列长度
            zero_feat = torch.zeros((L, 40), dtype=protein_feats.dtype, device=protein_feats.device)
            protein_feats = torch.cat([protein_feats, zero_feat], dim=1)  # (L, 681)
            return protein_feats

        # elif label == 1:
        #     muta_dic = mutation_feature_vector(mutation, seq_len)  # {pos:[orig_vec, target_vec]}
        #     mut_feat_tensor = torch.zeros((protein_feats.size(0), 40), dtype=protein_feats.dtype,
        #                                   device=protein_feats.device)
        #     for pos, feat in muta_dic.items():
        #         if 0 <= pos < protein_feats.size(0):
        #             feat_tensor = torch.from_numpy(feat).to(protein_feats.device).to(protein_feats.dtype)
        #             mut_feat_tensor[pos] = feat_tensor  # 替换该位置的突变向量
        #
        #     # 拼接原特征 + 突变特征，得到 (seq_len, 660)
        #     protein_feats = torch.cat([protein_feats, mut_feat_tensor], dim=1)
        #     return protein_feats


    @staticmethod
    def loading_data(*file_paths,device):
        smiles_list=[]
        pro_sequences_list=[]
        label_list=[]
        value_list=[]
        mutation_list=[]

        transformers.logging.set_verbosity_error()
        """从csv加载数据"""
        for file_path in file_paths:
            df = pd.read_csv(file_path,low_memory=False )
            smiles_list.extend(df['Smiles'].tolist())
            pro_sequences_list.extend(df['Sequence'].tolist())
            label_list.extend(df['Label'].tolist())
            mutation_list.extend(df['Mutation'].tolist())
            # value_list.extend(df['Value'].tolist())
            value_list.extend(np.log10(1 + df['Value'].values))



        atom_featurizer = CanonicalAtomFeaturizer()     #分子特征提取函数实体化
        bond_featurizer = CanonicalBondFeaturizer()

        '''huggingface下载模型'''
        # model_name = "facebook/esm2_t30_150M_UR50D"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name).to(device)

        '''本地加载模型'''

        # model_path = "esm2_t33_650M_UR50D.pt"
        # tokenizer_path ="tokenizer"

        '''服务器地址'''
        model_path = "/ldata/databases/folding/esm/esm2_t33_650M_UR50D.pt"
        tokenizer_path = "tokenizer"



        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 加载模型配置文件
        config = AutoConfig.from_pretrained(tokenizer_path)
        # 初始化模型架构
        model =AutoModel.from_config(config)
        checkpoint = torch.load(model_path,weights_only=False, map_location=device)  # 读取 .pt 权重文件
        model_weights = checkpoint['model']  # 获取模型的权重
        # 将权重加载到模型
        model.load_state_dict(model_weights, strict=False)
        model = (model.to(device)).eval()



        molecule_data = []  # 存储所有分子数据
        pro_feats_dic = {}
        protein_feats = None

        print('———————————————— Start loading dataset ————————————————')
        start = time.time()
        for i, smiles in enumerate(smiles_list):
            '''化合物特征提取'''
            graph = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
            node_feats = graph.ndata['h']       #（L，C）  原子，特征数
            edge_feats = graph.edata['e']       #（L，C）  化学键，特征数
            mol = Chem.MolFromSmiles(smiles)
            # 获取邻接矩阵
            adj_matrix = Chem.GetAdjacencyMatrix(mol)       #（L，L）原子，原子
            adj_matrix=torch.tensor(adj_matrix)

            '''蛋白质特征提取'''

            if not pro_feats_dic:
                protein_feats, pro_feats_dic = MolecularDataset.get_protein_features(pro_sequences_list, i, tokenizer, model, device,
                                                                     pro_feats_dic)
            else:
                if pro_sequences_list[i] not in pro_feats_dic:
                    protein_feats,pro_feats_dic=MolecularDataset.get_protein_features(pro_sequences_list,i,tokenizer,model,device,
                                                                      pro_feats_dic)
                else:
                    protein_feats=pro_feats_dic[pro_sequences_list[i]]

            '''value提取'''
            value = value_list[i]


            '''mutation提取'''
            # protein_feats=MolecularDataset.mutation(mutation = mutation_list[i],label = label_list[i],
            #                        protein_feats = protein_feats,seq_len = len(pro_sequences_list[i]))


            # 存储到字典
            molecule_info = {
                'graph': graph,  # 直接存 DGLGraph
                'node_feats': node_feats,
                'edge_feats': edge_feats,
                'protein_feats': protein_feats,  # 形状 [len, 1280+40=1320]，之后 `collate_fn` 处理
                'value': value,
                'adj_matrix': adj_matrix.float()
            }
            #添加到列表
            molecule_data.append(molecule_info)

            if i % 1000 == 0:
                print(f'———————————— {i} sets of data loaded, time consuming: {time.time() - start:.2f}s ————————————')

        end = time.time()
        print('——————————Loading data completed, total {} sets of data,time consuming：{:.2f}s——————————'.format(
            len(smiles_list), end - start))
        return MolecularDataset(molecule_data)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('using device:{}'.format(device))
# file_path='/Users/zephyr/Documents/PycharmProjects/2025igem/GNN预测/dlkcat_data/test.csv'
# loading_data=MolecularDataset.loading_data(file_path,device=device)



#
# class ProportionalBatchSampler(Sampler):
#     def __init__(self, generic_len, kras_len, batch_size, kras_ratio=0.2):
#         self.generic_len = generic_len
#         self.kras_len = kras_len
#         self.batch_size = batch_size
#         self.kras_ratio = kras_ratio
#
#         self.num_kras = int(batch_size * kras_ratio)
#         self.num_generic = batch_size - self.num_kras
#
#         self.generic_indices = list(range(generic_len))
#         self.kras_indices = list(range(kras_len))
#
#     def __iter__(self):
#         # 每次打乱
#         random.shuffle(self.generic_indices)
#         random.shuffle(self.kras_indices)
#
#         g_ptr = 0
#         k_ptr = 0
#
#         while g_ptr + self.num_generic <= self.generic_len:
#             if k_ptr + self.num_kras > self.kras_len:
#                 # 重复使用 KRAS 索引
#                 random.shuffle(self.kras_indices)
#                 k_ptr = 0
#
#             generic_batch = self.generic_indices[g_ptr: g_ptr + self.num_generic]
#             kras_batch = self.kras_indices[k_ptr: k_ptr + self.num_kras]
#
#             g_ptr += self.num_generic
#             k_ptr += self.num_kras
#
#             # 返回组合 batch 索引，前面是 generic，后面是 KRAS（或混合）
#             yield [('generic', i) for i in generic_batch] + [('kras', i) for i in kras_batch]
#
#     def __len__(self):
#         return self.generic_len // self.num_generic
#
# class MixedDataset(torch.utils.data.Dataset):
#     def __init__(self, generic_dataset, kras_dataset):
#         self.generic_dataset = generic_dataset
#         self.kras_dataset = kras_dataset
#
#     def __getitem__(self, index_tuple):
#         kind, idx = index_tuple
#         if kind == 'generic':
#             return self.generic_dataset[idx]
#         else:
#             return self.kras_dataset[idx]
#
#     def __len__(self):
#         return len(self.generic_dataset) + len(self.kras_dataset)  # 用不到




