import pandas as pd
import torch
import os
import numpy as np
import random
import dgl
from torch.utils.data import DataLoader
import argparse
import sys
from rdkit import Chem
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from mpnn_predictor import MPNNPredictorWithProtein
from loading_experiment import MolecularDataset


def set_seed(seed=42):
    """固定随机种子，确保每次训练结果一致"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def append_columns_to_csv(csv_path, list1, list2, col_name1, col_name2):
    # 如果 CSV 已存在，读取它；否则创建空的 DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # 创建两个新的列（自动对齐行数，多余部分补 NaN）
    df[col_name1] = pd.Series(list1)
    df[col_name2] = pd.Series(list2)

    # 保存回 CSV
    df.to_csv(csv_path, index=False)
    print(f"已将列 '{col_name1}' 和 '{col_name2}' 添加到 {csv_path}")


def experiment(model, dataset, batch_size, device, seed=None):
    """
    对单组数据进行推理，直接返回回归值列表
    """
    model.eval()
    output_lst = []

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=MolecularDataset.collate_fn)
    counter = 0

    if seed is None:
        set_seed()
    else:
        set_seed(seed)

    with torch.no_grad():
        for i in dataloader:
            graph, node_feats, edge_feats, protein_feats, adj_matrix = i  # 解包 tuple
            graph = graph.to(device)
            protein_feats = protein_feats.to(device)
            node_feats = node_feats.to(device)
            edge_feats = edge_feats.to(device)
            adj_matrix = adj_matrix.to(device)

            output = torch.sigmoid(model(graph, node_feats, edge_feats, protein_feats, adj_matrix))
            output_lst.extend([round(float(x), 4) for x in output.float()])
            output = 10 ** output_lst[0]




    return output


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
def mutate_sequence(seq, positions):
    """随机改变指定位置的氨基酸"""
    seq_list = list(seq)
    for pos in positions:
        seq_list[pos] = random.choice(AMINO_ACIDS)
    return ''.join(seq_list)


def genetic_algorithm(seq, smiles, positions, model, device, generations=50, population_size=10, top_k=3):
    """
    seq: 原始蛋白序列
    smiles: 化合物SMILES
    positions: 可变位点的索引列表
    model: 已加载MPNN模型
    device: 计算设备
    """

    import random

    # 初始化种群
    population = [mutate_sequence(seq, positions) for _ in range(population_size)]
    best_seq = seq
    best_value = -float('inf')

    for gen in range(generations):
        print(f"\n=== Generation {gen + 1} ===")
        fitness_scores = []

        for idx, individual in enumerate(population):
            print(f"Evaluating individual {idx + 1}/{population_size}...", end=' ')
            dataset = MolecularDataset.loading_data(individual, smiles, device=device)
            value = experiment(model=model, dataset=dataset, batch_size=1, device=device, seed=48)
            fitness_scores.append((individual, value))
            print(f"Done. Value = {value:.4f}")

            if value > best_value:
                best_value = value
                best_seq = individual

        # 按得分排序
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # 打印本代 top_k
        print("Top individuals this generation:")
        for i, (seq_val, val) in enumerate(fitness_scores[:top_k]):
            print(f"  Rank {i+1}: Value = {val:.4f}, Seq = {seq_val}")

        # 选择 top_k 做交叉
        next_population = [seq for seq, val in fitness_scores[:top_k]]

        # 生成新个体
        while len(next_population) < population_size:
            parent = random.choice(next_population)
            child = mutate_sequence(parent, positions)
            next_population.append(child)

        population = next_population
        print(f"Best value so far: {best_value:.4f}, Best seq so far: {best_seq}")

    return best_seq, best_value





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:', device)

if __name__ == '__main__':
    mpnn = MPNNPredictorWithProtein().to(device)
    mpnn.load_state_dict(torch.load('best_model.pth',weights_only=True) ) # 加载训练好的权重
    mpnn.eval()

    seq = "MNVIAILNHMGVYFKEEPIRELHRALERLNFQIVYPNDRDDLLKLIENNARLCGVIFDWDKYNLELCEEISKMNENLPLYAFANTYSTLDVSLNDLRLQISFFEYALGAAEDIANKIKQTTDEYINTILPPLTKALFKYVREGKYTFCTPGHMGGTAFQKSPVGSLFYDFFGPNTMKSDISISVSELGSLLDHSGPHKEAEQYIARVFNADRSYMVTNGTSTANKIVGMYSAPAGSTILIDRNCHKSLTHLMMMSDVTPIYFRPTRNAYGILGGIPQSEFQHATIAKRVKETPNATWPVHAVITNSTYDGLLYNTDFIKKTLDVKSIHFDSAWVPYTNFSPIYEGKCGMSGGRVEGKVIYETQSTHLLAAFSQASMIHVKGDVNEETFNEAYMMHTTTSPHYGIVASTETAAAMMKGNAGKRLINGSIERAIKFRKEIKRLRTESDGWFFDVWQPDHIDTTECWPLRSDSTWHGFKNIDNEHMYLDPIKVTLLTPGMEKDGTMSDFGIPASIVAKYLDEHGIVVEKTGPYNLLFLFSIGIDKTKALSLLRALTDFKRAFDLNLRVKNMLPSLYREDPEFYENMRIQELAQNIHKLIVHHNLPDLMYRAFEVLPTMVMTPYAAFQKELHGMTEEVYLDEMVGRINANMILPYPPGVPLVMPGEMITEESRPVLEFLQMLCEIGAHYPGFETDIHGAYRQADGRYTVKVLKE"
    smiles = "C(CCN)C[C@@H](C(=O)O)N"
    positions = [181, 244, 245, 332, 524, 525 ,650]  # 想改变的序号
    best_seq, best_value = genetic_algorithm(seq, smiles, positions, model=mpnn, device=device)
    print("Best sequence:", best_seq)
    print("Predicted value:", best_value)