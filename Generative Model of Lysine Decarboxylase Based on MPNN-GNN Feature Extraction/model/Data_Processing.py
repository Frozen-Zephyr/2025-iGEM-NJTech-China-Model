import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit.Chem import Descriptors
from difflib import SequenceMatcher
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

def compute_properties(smiles):
    """计算分子理化性质"""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MW": round(Descriptors.MolWt(mol), 2),  # 分子量
            "LogP": round(Descriptors.MolLogP(mol), 2),  # LogP
            "HBA": Descriptors.NumHAcceptors(mol),  # 氢键受体数
            "HBD": Descriptors.NumHDonors(mol),  # 氢键供体数
            "TPSA": round(Descriptors.TPSA(mol), 2)  # 极性表面积
        }
    return None


def dealing_csv(active_df,output_csv):
    '''处理虚假配体库'''
    # 处理 CSV 文件
    results = []
    for smiles in tqdm(active_df['SMILES']):
        # 确保 CSV 文件包含 "smiles" 列
        properties = compute_properties(smiles)
        if properties:
            results.append([smiles, properties["MW"], properties["LogP"], properties["HBA"], properties["HBD"], properties["TPSA"]])
        else:
            results.append([smiles, "Invalid", "Invalid", "Invalid", "Invalid", "Invalid"])

    # 保存到 CSV
    columns = ["SMILES", "MW", "LogP", "HBA", "HBD", "TPSA"]
    pd.DataFrame(results, columns=columns).to_csv(output_csv, index=False)

    print(f"理化性质计算完成，结果已保存至 {output_csv}")


def DUDE(input_file,output_file):
    '''删除少于五个正样本的蛋白（DUDE）'''
    # 读取 CSV，假设 CSV 有表头
    df = pd.read_csv(input_file,low_memory=False)

    # 统计每个 Protein_name 对应的 label=1 的数量
    positive_counts = df[df["Label"] == 1].groupby("Protein Name").size()

    # 找出正样本数 < 5 的蛋白
    proteins_to_remove = positive_counts[positive_counts < 5].index

    # 过滤掉这些蛋白对应的所有行
    filtered_df = df[~df["Protein Name"].isin(proteins_to_remove)]

    # 保存结果到新的 CSV 文件
    filtered_df.to_csv(output_file, index=False)

    print(f"筛选后数据已保存至 {output_file}")


def count_labels(*csv_files):
    '''计算正负样本数量'''
    count_0=0
    count_1=0
    count_total=0
    for csv_file in csv_files:
        # 读取 CSV 文件
        df = pd.read_csv(csv_file,low_memory=False)

        # 确保 'label' 列存在
        if 'Label' not in df.columns:
            raise ValueError("CSV 文件中没有 'Label' 列")

        # 统计 0 和 1 的数量
        count_0 += (df['Label'] == 0).sum()
        count_1 += (df['Label'] == 1).sum()
    count_total += count_0 + count_1

    # 计算 0 是 1 的几倍
    ratio = count_0 / count_1 if count_1 > 0 else float('inf')

    print('一共数据有：{}'.format(count_total))
    print(f"负样本有: {count_0}")
    print(f"正样本有: {count_1}")
    print(f"负样本数是正样本的 {ratio:.2f} 倍")


def count_proteins(*file_path):
    '''计算蛋白数量'''
    unique_proteins=0
    for csv_file in file_path:
        # 读取CSV文件
        df = pd.read_csv(csv_file,low_memory=False)

        # 确保列名正确
        if 'Sequence' not in df.columns:
            raise ValueError("CSV文件中未找到 'Sequence' 列")

        # 计算唯一蛋白数量
        unique_proteins += df['Sequence'].nunique()
    print('有{}个蛋白'.format(unique_proteins))


def is_similar(seq1, seq2, threshold=0.8):
    """判断两个蛋白质序列是否相似，使用相似度阈值"""
    return SequenceMatcher(None, seq1, seq2).ratio() >= threshold


def remove_duplicates(file1, file2, output_file):
    '''正样本间查重'''
    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 确保列名正确
    if 'SMILES' not in df1.columns or 'Protein Sequence' not in df1.columns:
        raise ValueError("CSV文件1中缺少 'SMILES' 或 'Protein Sequence' 列")
    if 'SMILES' not in df2.columns or 'Protein_sequence' not in df2.columns:
        raise ValueError("CSV文件2中缺少 'SMILES' 或 'Protein Sequence' 列")

    # 记录重复项
    duplicates = set()

    for i, row1 in tqdm(df1.iterrows(), total=len(df1), desc="Processing File 1"):
        for j, row2 in df2.iterrows():
            if row1['SMILES'] == row2['SMILES'] and is_similar(row1['Protein Sequence'], row2['Protein Sequence']):
                duplicates.add(j)  # 记录df2中重复的行索引

    # 过滤掉df2中的重复项
    df2_filtered = df2.drop(index=list(duplicates))

    # 合并不重复的部分
    merged_df = pd.concat([df1, df2_filtered], ignore_index=True)

    # 保存到新CSV文件
    merged_df.to_csv(output_file, index=False)
    print(f'去重后的数据已保存至 {output_file}')


def save_large_csv(data, base_filename, max_rows=1_000_000):
    """当 CSV 超过 max_rows 时，拆分多个文件"""
    num_parts = (len(data) // max_rows) + (1 if len(data) % max_rows != 0 else 0)

    for i in range(num_parts):
        start = i * max_rows
        end = start + max_rows
        filename = f"{base_filename}_{i + 1}.csv" if num_parts > 1 else f"{base_filename}.csv"
        data.iloc[start:end].to_csv(filename, index=False)


def split_csv(*file_paths, label_col='Label', train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    '''多个csv文件数据集分割'''
    total_data = []

    # 读取所有 CSV 文件并合并
    for file_path in file_paths:
        df = pd.read_csv(file_path,low_memory=False)
        total_data.append(df)

    df = pd.concat(total_data, ignore_index=True)  # 合并数据

    # 按 label 分层抽样
    train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), stratify=df[label_col], random_state=42)
    valid_size = valid_ratio / (valid_ratio + test_ratio)
    valid_data, test_data = train_test_split(temp_data, test_size=(1 - valid_size), stratify=temp_data[label_col],
                                             random_state=42)

    # **按 100 万行拆分保存**
    save_large_csv(train_data, '/Users/zephyr/Documents/PycharmProjects/2025igem/GNN预测/new_data/集中/trainset.csv')
    save_large_csv(valid_data, "/Users/zephyr/Documents/PycharmProjects/2025igem/GNN预测/new_data/集中/validset.csv")
    save_large_csv(test_data, "/Users/zephyr/Documents/PycharmProjects/2025igem/GNN预测/new_data/集中/testset.csv")

    print("数据集拆分完成！")


def Tanimoto(input_path, output_path):
    # 读取你的CSV文件
    df = pd.read_csv(input_path)  # 替换为你的实际文件名
    smiles_list = df["SMILES"]  # 替换为你的实际列名

    # 两个参考化合物的 SMILES
    MRTX133 = Chem.MolFromSmiles("C#Cc1c(ccc2c1c(cc(c2)O)c3c(c4c(cn3)c(nc(n4)OC[C@@]56CCCN5C[C@@H](C6)F)N7C[C@H]8CC[C@@H](C7)N8)F)F")  # 替换为你的参考化合物1
    BI_2852 = Chem.MolFromSmiles("c1nc2c(n1[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@](=O)(O)O[P@@](=O)(CP(=O)(O)O)O)O)O)N=C(NC2=O)N")  # 替换为你的参考化合物2

    # 生成参考化合物的指纹
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp_ref1 = AllChem.GetMorganFingerprintAsBitVect(MRTX133, radius=2, nBits=2048)
    fp_ref2 = AllChem.GetMorganFingerprintAsBitVect(BI_2852, radius=2, nBits=2048)

    # 计算每个化合物与参考化合物的相似度
    similarities_1 = []
    similarities_2 = []

    for smi in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = morgan_gen.GetFingerprint(mol)
            sim1 = DataStructs.TanimotoSimilarity(fp, fp_ref1)
            sim2 = DataStructs.TanimotoSimilarity(fp, fp_ref2)
        else:
            sim1, sim2 = None, None  # 或用0、-1表示解析失败
        similarities_1.append(round(sim1,5))
        similarities_2.append(round(sim2,5))

    # 添加到 DataFrame 并保存
    df["similarity_to_MRTX1133"] = similarities_1
    df["similarity_to_BI2852"] = similarities_2
    df.to_csv(output_path, index=False)


def fix_pdb(input_pdb, output_pdb, ph=7.0):
    '''修补蛋白质氨基酸原子'''
    fixer = PDBFixer(filename=input_pdb)

    print("Finding missing residues and atoms...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    print("Adding hydrogens...")
    fixer.addMissingHydrogens(pH=ph)

    print(f"Saving fixed PDB to: {output_pdb}")
    with open(output_pdb, 'w') as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, out)


def smile_to_pdb(smiles,out):
    smiles = smiles
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, out)


def mean_substrate(inputpath, outputpath):
    # 读取原始CSV文件
    df = pd.read_csv(inputpath)  # 替换为你的CSV文件路径

    # 确保列名正确（区分大小写）
    assert 'Smiles' in df.columns and 'Value' in df.columns, "列名必须是 'Smiles' 和 'value'"

    # 将value列转换为数值型（防止是字符串）
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # 去掉缺失值（若有）
    df = df.dropna(subset=['Smiles', 'Value'])

    # 按Smiles分组取平均
    df_avg = df.groupby('Smiles', as_index=False)['Value'].mean()

    # 重命名为MeanValue更清晰（可选）
    df_avg = df_avg.rename(columns={'Value': 'MeanValue'})

    # 保存为新CSV
    df_avg.to_csv(outputpath, index=False)

    print("完成！已保存为 smiles_mean_values.csv")



input_path = '/Users/zephyr/Documents/PycharmProjects/2025igem/GNN预测/new_data/集中/EITLEMoutput_filtered.csv'
output_path = '/Users/zephyr/Documents/PycharmProjects/2025igem/GNN预测/dlkcat_data/去极端/smile_mean.csv'

if __name__ == "__main__":
    while True:
        code=input('mission:')

        if code == 'DUDE':

            print('正在处理数据，请稍后......')
            DUDE(input_path,output_path)
            break

        elif code == 'count':
            '计算正负样本数，蛋白数量'
            print('正在处理数据，请稍后......')
            count_labels(input_path)
            count_proteins(input_path)
            break

        elif code == 'split':
            '分割为训练、验证、测试集'
            print('正在处理数据，请稍后......')
            split_csv(input_path)
            break

        elif code == 'tanimoto':
            Tanimoto(input_path, output_path)
            break

        elif code == 'fix':
            input_pdb = '/Users/zephyr/Downloads/ligand1complex.pdb'
            output_pdb = '/Users/zephyr/Downloads/ligand1complex_fixed.pdb'
            fix_pdb(input_pdb, output_pdb)
            break

        elif code == 'to_pdb':
            smiles = 'CC(=O)O'
            out = '/Users/zephyr/Downloads/prot7.pdb'
            smile_to_pdb(smiles, out)
            break

        elif code == 'mean':
            mean_substrate(input_path, output_path)
            break

        else:
            continue

