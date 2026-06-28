import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from config import config
from utils import save_pickle, load_pickle


def load_features(path, id_column):
    """
    🔧 修复：加载特征文件（智能处理ID类型）
    
    Args:
        path: 特征文件路径
        id_column: ID列名（'RNA_ID' 或 'CID'）
    
    Returns:
        feature_dict: {id: feature_vector}
    """
    df = pd.read_excel(path)
    feature_dict = {}
    
    for _, row in df.iterrows():
        # 🆕 智能转换：统一转为字符串
        entity_id_raw = row[id_column]
        
        # 处理不同类型
        if pd.isna(entity_id_raw):
            # 跳过空值
            continue
        elif isinstance(entity_id_raw, (int, float)):
            # 数字类型：转为整数再转字符串
            entity_id = str(int(entity_id_raw))
        else:
            # 字符串类型：直接去除空格
            entity_id = str(entity_id_raw).strip()
        
        # 提取特征
        features = row.drop(id_column).values.astype(np.float32)
        feature_dict[entity_id] = features
    
    return feature_dict


def compute_jaccard_similarity(vec1, vec2):
    """
    计算两个二值向量的Jaccard相似度
    
    Args:
        vec1: numpy array, 二值向量
        vec2: numpy array, 二值向量
    
    Returns:
        float: Jaccard相似度 [0, 1]
    """
    # 确保是二值向量
    vec1_binary = (vec1 > 0).astype(int)
    vec2_binary = (vec2 > 0).astype(int)
    
    # 计算交集和并集
    intersection = np.sum(vec1_binary & vec2_binary)
    union = np.sum(vec1_binary | vec2_binary)
    
    # 避免除以0
    if union == 0:
        return 0.0
    
    return intersection / union


def precompute_similarity_matrices(rna_onehot_dict, drug_onehot_dict, threshold=0.9):
    """
    🆕 预计算RNA-RNA和Drug-Drug的Jaccard相似性矩阵
    
    Args:
        rna_onehot_dict: {rna_id: onehot_vector}
        drug_onehot_dict: {drug_id: onehot_vector}
        threshold: 相似度阈值
    
    Returns:
        rna_similar_dict: {rna_id: [similar_rna_ids]}
        drug_similar_dict: {drug_id: [similar_drug_ids]}
    """
    print(f"\n🔍 预计算Jaccard相似性矩阵（阈值={threshold})...")
    
    # RNA相似性
    rna_ids = list(rna_onehot_dict.keys())
    rna_similar_dict = {}
    
    print(f"   计算RNA-RNA相似性 (共{len(rna_ids)}个RNA)...")
    for i, rna_i in enumerate(rna_ids):
        if (i + 1) % 100 == 0:
            print(f"      进度: {i+1}/{len(rna_ids)}")
        
        similar_rnas = []
        vec_i = rna_onehot_dict[rna_i]
        
        for rna_j in rna_ids:
            if rna_i == rna_j:
                continue
            
            vec_j = rna_onehot_dict[rna_j]
            sim = compute_jaccard_similarity(vec_i, vec_j)
            
            if sim > threshold:
                similar_rnas.append(rna_j)
        
        rna_similar_dict[rna_i] = similar_rnas
    
    # 统计RNA相似性
    total_similar_pairs = sum(len(v) for v in rna_similar_dict.values())
    avg_similar = total_similar_pairs / len(rna_ids) if rna_ids else 0
    print(f"   ✅ RNA: 平均每个RNA有 {avg_similar:.2f} 个相似RNA")
    
    # Drug相似性
    drug_ids = list(drug_onehot_dict.keys())
    drug_similar_dict = {}
    
    print(f"   计算Drug-Drug相似性 (共{len(drug_ids)}个Drug)...")
    for i, drug_i in enumerate(drug_ids):
        if (i + 1) % 50 == 0:
            print(f"      进度: {i+1}/{len(drug_ids)}")
        
        similar_drugs = []
        vec_i = drug_onehot_dict[drug_i]
        
        for drug_j in drug_ids:
            if drug_i == drug_j:
                continue
            
            vec_j = drug_onehot_dict[drug_j]
            sim = compute_jaccard_similarity(vec_i, vec_j)
            
            if sim > threshold:
                similar_drugs.append(drug_j)
        
        drug_similar_dict[drug_i] = similar_drugs
    
    # 统计Drug相似性
    total_similar_pairs = sum(len(v) for v in drug_similar_dict.values())
    avg_similar = total_similar_pairs / len(drug_ids) if drug_ids else 0
    print(f"   ✅ Drug: 平均每个Drug有 {avg_similar:.2f} 个相似Drug")
    
    return rna_similar_dict, drug_similar_dict


def load_onehot_matrices():
    """
    🆕 加载onehot矩阵用于计算Jaccard相似性
    
    Returns:
        rna_onehot_dict: {rna_id: onehot_vector}
        drug_onehot_dict: {drug_id: onehot_vector}
    """
    print(f"\n📂 加载onehot矩阵用于相似性计算...")
    
    # 加载RNA onehot
    rna_onehot_dict = load_features(config.RNA_ONEHOT_MATRIX_PATH, 'RNA_ID')
    print(f"   ✅ RNA onehot: {len(rna_onehot_dict)} 条")
    
    # 加载Drug onehot
    drug_onehot_dict = load_features(config.DRUG_ONEHOT_MATRIX_PATH, 'CID')
    print(f"   ✅ Drug onehot: {len(drug_onehot_dict)} 条")
    
    return rna_onehot_dict, drug_onehot_dict


def is_valid_negative_sample(rna_id, drug_id, positive_pairs_set,
                             rna_similar_dict, drug_similar_dict):
    """
    🆕 判断候选负样本是否有效（基于Jaccard相似性）
    
    策略：
    1. 检查与rna_id相似的RNA是否与drug_id有正样本关联
    2. 检查与drug_id相似的Drug是否与rna_id有正样本关联
    
    Args:
        rna_id: RNA ID
        drug_id: Drug ID
        positive_pairs_set: set of (rna_id, drug_id)
        rna_similar_dict: {rna_id: [similar_rna_ids]}
        drug_similar_dict: {drug_id: [similar_drug_ids]}
    
    Returns:
        bool: True表示可以作为负样本，False表示拒绝
    """
    # 检查1: RNA相似性
    similar_rnas = rna_similar_dict.get(rna_id, [])
    for similar_rna in similar_rnas:
        if (similar_rna, drug_id) in positive_pairs_set:
            # 存在相似的RNA与该Drug有关联，拒绝
            return False
    
    # 检查2: Drug相似性
    similar_drugs = drug_similar_dict.get(drug_id, [])
    for similar_drug in similar_drugs:
        if (rna_id, similar_drug) in positive_pairs_set:
            # 存在相似的Drug与该RNA有关联，拒绝
            return False
    
    # 通过所有检查，可以作为负样本
    return True


def random_negative_sampling_with_jaccard(positive_pairs, all_rna_ids, all_drug_ids, 
                                         rna_similar_dict, drug_similar_dict,
                                         ratio=1, seed=42):
    """
    🆕 基于Jaccard相似性过滤的负采样
    
    Args:
        positive_pairs: list of tuple, 正样本对
        all_rna_ids: list, 所有RNA ID
        all_drug_ids: list, 所有Drug ID
        rna_similar_dict: dict, RNA相似性字典
        drug_similar_dict: dict, Drug相似性字典
        ratio: int, 负样本比例
        seed: int, 随机种子
    
    Returns:
        negative_pairs: list of tuple
    """
    random.seed(seed)
    
    # 确保所有ID都是字符串类型
    all_rna_ids = [str(x) for x in all_rna_ids]
    all_drug_ids = [str(x) for x in all_drug_ids]
    
    positive_pairs_str = [(str(rna), str(drug)) for rna, drug in positive_pairs]
    positive_set = set(positive_pairs_str)
    
    negative_set = set()
    negative_pairs = []
    
    num_negatives = len(positive_pairs) * ratio
    
    # 统计信息
    total_attempts = 0
    rejected_by_similarity = 0
    rejected_by_duplicate = 0
    
    print(f"\n   📊 负采样（带Jaccard过滤）:")
    print(f"      - 正样本数: {len(positive_set)}")
    print(f"      - 目标负样本数: {num_negatives}")
    print(f"      - Jaccard阈值: {config.JACCARD_THRESHOLD}")
    
    # 使用更合理的最大尝试次数
    max_total_attempts = num_negatives * config.MAX_SAMPLING_ATTEMPTS
    
    while len(negative_pairs) < num_negatives and total_attempts < max_total_attempts:
        rna_id = random.choice(all_rna_ids)
        drug_id = random.choice(all_drug_ids)
        
        rna_id = str(rna_id)
        drug_id = str(drug_id)
        
        pair = (rna_id, drug_id)
        total_attempts += 1
        
        # 检查是否为正样本
        if pair in positive_set:
            rejected_by_duplicate += 1
            continue
        
        # 检查是否已经采样过
        if pair in negative_set:
            rejected_by_duplicate += 1
            continue
        
        # 🆕 检查Jaccard相似性
        if not is_valid_negative_sample(rna_id, drug_id, positive_set,
                                       rna_similar_dict, drug_similar_dict):
            rejected_by_similarity += 1
            continue
        
        # 通过所有检查，添加为负样本
        negative_set.add(pair)
        negative_pairs.append(pair)
        
        # 每1000个样本打印一次进度
        if len(negative_pairs) % 1000 == 0:
            print(f"      进度: {len(negative_pairs)}/{num_negatives}")
    
    # 打印统计信息
    print(f"\n   ✅ 负采样完成:")
    print(f"      - 成功采样: {len(negative_pairs)}/{num_negatives}")
    print(f"      - 总尝试次数: {total_attempts}")
    print(f"      - 被相似性拒绝: {rejected_by_similarity} ({100*rejected_by_similarity/total_attempts:.2f}%)")
    print(f"      - 被重复拒绝: {rejected_by_duplicate} ({100*rejected_by_duplicate/total_attempts:.2f}%)")
    
    if len(negative_pairs) < num_negatives:
        print(f"   ⚠️  警告: 仅采样到 {len(negative_pairs)}/{num_negatives} 个负样本")
        print(f"      可能需要降低Jaccard阈值或增加最大尝试次数")
    
    # 验证负样本质量
    overlap = set(negative_pairs) & positive_set
    if overlap:
        print(f"   ❌ 错误: 发现 {len(overlap)} 个负样本与正样本重复!")
    else:
        print(f"   ✅ 负样本质量检查通过：无重复")
    
    return negative_pairs


def prepare_cv_splits(positive_pairs, all_rna_ids, all_drug_ids, n_folds=5, 
                      negative_ratio=1, seed=42, save_dir=None):
    """
    🔧 修复版：准备交叉验证的数据划分（带Jaccard相似性过滤）
    
    Args:
        positive_pairs: list of tuple
        all_rna_ids: list
        all_drug_ids: list
        n_folds: int
        negative_ratio: int
        seed: int
        save_dir: str, 缓存目录
    
    Returns:
        cv_splits: list of dict
    """
    # 确保所有ID都是字符串类型
    positive_pairs = [(str(rna), str(drug)) for rna, drug in positive_pairs]
    all_rna_ids = [str(x) for x in all_rna_ids]
    all_drug_ids = [str(x) for x in all_drug_ids]
    
    # 检查缓存
    if save_dir and os.path.exists(save_dir):
        cache_file = os.path.join(save_dir, 
                                 f'cv_splits_jaccard{config.JACCARD_THRESHOLD}_ratio{negative_ratio}_seed{seed}.pkl')
        if os.path.exists(cache_file):
            print(f"\n📂 从缓存加载CV划分: {cache_file}")
            return load_pickle(cache_file)
    
    print(f"\n🔄 生成{n_folds}折交叉验证划分（Jaccard过滤）...")
    
    # ===== 步骤1: 加载onehot矩阵并预计算相似性 =====
    rna_onehot_dict, drug_onehot_dict = load_onehot_matrices()
    rna_similar_dict, drug_similar_dict = precompute_similarity_matrices(
        rna_onehot_dict, drug_onehot_dict, threshold=config.JACCARD_THRESHOLD
    )
    
    # ===== 步骤2: 使用Jaccard过滤进行负采样 =====
    print(f"\n📊 步骤2: 使用Jaccard相似性过滤进行负采样...")
    negative_pairs = random_negative_sampling_with_jaccard(
        positive_pairs, all_rna_ids, all_drug_ids,
        rna_similar_dict, drug_similar_dict,
        ratio=negative_ratio, seed=seed
    )
    
    # ===== 步骤3: 合并所有样本并打上标签 =====
    print(f"\n📊 步骤3: 合并正负样本...")
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    print(f"   ✅ 总样本数: {len(all_pairs)} (正:{len(positive_pairs)} + 负:{len(negative_pairs)})")
    
    # ===== 步骤4: K折划分 =====
    print(f"\n📊 步骤4: 进行{n_folds}折划分...")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_pairs)):
        print(f"\n   处理第 {fold_idx} 折...")
        
        train_pairs = [all_pairs[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        
        val_pairs = [all_pairs[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        train_pos = [pair for pair, label in zip(train_pairs, train_labels) if label == 1]
        train_neg = [pair for pair, label in zip(train_pairs, train_labels) if label == 0]
        
        val_pos = [pair for pair, label in zip(val_pairs, val_labels) if label == 1]
        val_neg = [pair for pair, label in zip(val_pairs, val_labels) if label == 0]
        
        cv_splits.append({
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'train_labels': train_labels,
            'val_labels': val_labels,
            'train_pos': train_pos,
            'train_neg': train_neg,
            'val_pos': val_pos,
            'val_neg': val_neg
        })
        
        print(f"      ✅ 训练集: {len(train_pos)} 正样本 + {len(train_neg)} 负样本")
        print(f"      ✅ 验证集: {len(val_pos)} 正样本 + {len(val_neg)} 负样本")
    
    # ===== 验证：确保没有数据泄露 =====
    print(f"\n🔍 验证数据划分...")
    positive_set = set(positive_pairs)
    negative_set = set(negative_pairs)
    
    overlap = positive_set & negative_set
    if overlap:
        print(f"   ⚠️  警告: 发现 {len(overlap)} 个正负样本重叠!")
    else:
        print(f"   ✅ 正负样本无重叠")
    
    for fold_idx, split in enumerate(cv_splits):
        val_neg_set = set(split['val_neg'])
        leak = val_neg_set & positive_set
        if leak:
            print(f"   ⚠️  Fold {fold_idx}: 验证集中发现 {len(leak)} 个泄露样本!")
        else:
            print(f"   ✅ Fold {fold_idx}: 无数据泄露")
    
    # 保存缓存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cache_file = os.path.join(save_dir, 
                                 f'cv_splits_jaccard{config.JACCARD_THRESHOLD}_ratio{negative_ratio}_seed{seed}.pkl')
        save_pickle(cv_splits, cache_file)
        print(f"\n💾 CV划分已缓存: {cache_file}")
    
    return cv_splits


class RNADrugDataset(Dataset):
    """
    RNA-Drug交互数据集（通用版，支持lncRNA/miRNA/circRNA）
    """
    def __init__(self, pairs, labels, rna_features_dict, drug_features_dict):
        """
        Args:
            pairs: list of tuple, [(rna_id, drug_id), ...]
            labels: list of int, 标签列表 [1, 0, 1, ...]
            rna_features_dict: dict of dict, 
                {
                    'seq': {rna_id: feature},
                    'struct': {rna_id: feature},
                    'disease': {rna_id: feature}
                }
            drug_features_dict: dict of dict,
                {
                    'seq': {drug_id: feature},
                    'graph': {drug_id: feature},
                    'ecfp': {drug_id: feature},
                    'disease': {drug_id: feature}
                }
        """
        self.pairs = pairs
        self.labels = labels
        self.rna_features_dict = rna_features_dict
        self.drug_features_dict = drug_features_dict
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        rna_id, drug_id = self.pairs[idx]
        label = self.labels[idx]
        
        # 确保ID是字符串类型
        rna_id = str(rna_id).strip()
        drug_id = str(drug_id).strip()
        
        # 构建样本字典
        sample = {
            'rna_id': rna_id,
            'drug_id': drug_id,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # 添加RNA特征
        if 'seq' in self.rna_features_dict and True:
            sample['rna_seq'] = torch.tensor(
                self.rna_features_dict['seq'][rna_id], dtype=torch.float32
            )
        
        if 'struct' in self.rna_features_dict and True:
            sample['rna_struct'] = torch.tensor(
                self.rna_features_dict['struct'][rna_id], dtype=torch.float32
            )
        
        if 'disease' in self.rna_features_dict and True:
            sample['rna_disease'] = torch.tensor(
                self.rna_features_dict['disease'][rna_id], dtype=torch.float32
            )
        
        # 添加Drug特征
        if 'seq' in self.drug_features_dict and True:
            sample['drug_seq'] = torch.tensor(
                self.drug_features_dict['seq'][drug_id], dtype=torch.float32
            )
        
        if 'graph' in self.drug_features_dict and True:
            sample['drug_graph'] = torch.tensor(
                self.drug_features_dict['graph'][drug_id], dtype=torch.float32
            )
        
        if 'ecfp' in self.drug_features_dict and True:
            sample['drug_ecfp'] = torch.tensor(
                self.drug_features_dict['ecfp'][drug_id], dtype=torch.float32
            )
        
        if 'disease' in self.drug_features_dict and True:
            sample['drug_disease'] = torch.tensor(
                self.drug_features_dict['disease'][drug_id], dtype=torch.float32
            )
        
        return sample


def load_all_features():
    """
    加载所有特征数据（包括归一化的疾病关联特征）
    
    Returns:
        rna_features_dict: dict of dict
        drug_features_dict: dict of dict
    """
    print("\n📊 加载特征数据...")
    
    rna_features_dict = {}
    drug_features_dict = {}
    
    # 加载RNA特征
    if True and os.path.exists(config.RNA_SEQ_FEATURE_PATH):
        rna_features_dict['seq'] = load_features(config.RNA_SEQ_FEATURE_PATH, 'RNA_ID')
        print(f"   ✅ RNA序列特征: {len(rna_features_dict['seq'])} 条")
    
    if True and os.path.exists(config.RNA_STRUCT_FEATURE_PATH):
        rna_features_dict['struct'] = load_features(config.RNA_STRUCT_FEATURE_PATH, 'RNA_ID')
        print(f"   ✅ RNA结构特征: {len(rna_features_dict['struct'])} 条")
    
    if True and os.path.exists(config.RNA_DISEASE_FEATURE_PATH):
        rna_features_dict['disease'] = load_features(config.RNA_DISEASE_FEATURE_PATH, 'RNA_ID')
        print(f"   ✅ RNA疾病特征(归一化): {len(rna_features_dict['disease'])} 条")
    
    # 加载Drug特征
    if True and os.path.exists(config.DRUG_SEQ_FEATURE_PATH):
        drug_features_dict['seq'] = load_features(config.DRUG_SEQ_FEATURE_PATH, 'CID')
        print(f"   ✅ Drug序列特征: {len(drug_features_dict['seq'])} 条")
    
    if True:
        if os.path.exists(config.DRUG_GRAPH_FEATURE_PATH):
            drug_features_dict['graph'] = load_features(config.DRUG_GRAPH_FEATURE_PATH, 'CID')
            print(f"   ✅ Drug图特征: {len(drug_features_dict['graph'])} 条")
        
        if os.path.exists(config.DRUG_ECFP_FEATURE_PATH):
            drug_features_dict['ecfp'] = load_features(config.DRUG_ECFP_FEATURE_PATH, 'CID')
            print(f"   ✅ Drug ECFP特征: {len(drug_features_dict['ecfp'])} 条")
    
    if True and os.path.exists(config.DRUG_DISEASE_FEATURE_PATH):
        drug_features_dict['disease'] = load_features(config.DRUG_DISEASE_FEATURE_PATH, 'CID')
        print(f"   ✅ Drug疾病特征(归一化): {len(drug_features_dict['disease'])} 条")
    
    return rna_features_dict, drug_features_dict


def load_positive_pairs():
    """
    加载正样本对（智能处理ID类型）
    
    Returns:
        positive_pairs: list of tuple
        all_rna_ids: list
        all_drug_ids: list
    """
    print(f"\n📂 加载正样本对: {config.POSITIVE_PAIRS_PATH}")
    
    df = pd.read_excel(config.POSITIVE_PAIRS_PATH)
    
    # 智能转换ID为字符串
    def convert_id(x):
        if pd.isna(x):
            return None
        elif isinstance(x, (int, float)):
            return str(int(x))
        else:
            return str(x).strip()
    
    df['RNA_ID'] = df['RNA_ID'].apply(convert_id)
    df['CID'] = df['CID'].apply(convert_id)
    
    # 过滤掉空值
    df = df.dropna(subset=['RNA_ID', 'CID'])
    
    positive_pairs = list(zip(df['RNA_ID'], df['CID']))
    all_rna_ids = df['RNA_ID'].unique().tolist()
    all_drug_ids = df['CID'].unique().tolist()
    
    print(f"   ✅ 正样本对: {len(positive_pairs)} 个")
    print(f"   ✅ RNA总数: {len(all_rna_ids)}")
    print(f"   ✅ Drug总数: {len(all_drug_ids)}")
    
    return positive_pairs, all_rna_ids, all_drug_ids