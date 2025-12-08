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


def random_negative_sampling(positive_pairs, all_rna_ids, all_drug_ids, ratio=1, seed=42):
    """
    🔧 修复版随机负采样：确保ID类型一致，提高效率
    
    Args:
        positive_pairs: list of tuple, 正样本对 [(rna_id, drug_id), ...]
        all_rna_ids: list, 所有RNA ID
        all_drug_ids: list, 所有Drug ID
        ratio: int, 负样本比例
        seed: int, 随机种子
    
    Returns:
        negative_pairs: list of tuple
    """
    random.seed(seed)
    
    # 🔧 修复1：确保所有ID都是字符串类型
    all_rna_ids = [str(x) for x in all_rna_ids]
    all_drug_ids = [str(x) for x in all_drug_ids]
    
    # 🔧 修复2：确保正样本对中的ID也是字符串
    positive_pairs_str = [(str(rna), str(drug)) for rna, drug in positive_pairs]
    positive_set = set(positive_pairs_str)
    
    # 🔧 修复3：使用set存储负样本，提高查找效率
    negative_set = set()
    negative_pairs = []
    
    num_negatives = len(positive_pairs) * ratio
    
    # 添加最大尝试次数,避免无限循环
    max_attempts = num_negatives * 100
    attempts = 0
    
    # 🆕 添加调试信息
    print(f"\n   📊 负采样调试信息:")
    print(f"      - 正样本数: {len(positive_set)}")
    print(f"      - RNA ID数: {len(all_rna_ids)}")
    print(f"      - Drug ID数: {len(all_drug_ids)}")
    print(f"      - 理论最大负样本数: {len(all_rna_ids) * len(all_drug_ids) - len(positive_set)}")
    
    while len(negative_pairs) < num_negatives and attempts < max_attempts:
        rna_id = random.choice(all_rna_ids)
        drug_id = random.choice(all_drug_ids)
        
        # 确保ID是字符串类型
        rna_id = str(rna_id)
        drug_id = str(drug_id)
        
        pair = (rna_id, drug_id)
        
        # 检查是否为新的负样本
        if pair not in positive_set and pair not in negative_set:
            negative_set.add(pair)
            negative_pairs.append(pair)
        
        attempts += 1
    
    if len(negative_pairs) < num_negatives:
        print(f"   ⚠️  警告: 仅采样到 {len(negative_pairs)}/{num_negatives} 个负样本")
    
    # 🆕 验证负样本质量
    overlap = set(negative_pairs) & positive_set
    if overlap:
        print(f"   ❌ 错误: 发现 {len(overlap)} 个负样本与正样本重复!")
        print(f"      重复的样本: {list(overlap)[:5]}")  # 显示前5个
    else:
        print(f"   ✅ 负样本质量检查通过：无重复")
    
    return negative_pairs


def prepare_cv_splits(positive_pairs, all_rna_ids, all_drug_ids, n_folds=5, 
                      negative_ratio=1, seed=42, save_dir=None):
    """
    🔧 修复版：准备交叉验证的数据划分
    
    核心改进：
    1. 先统一从所有正样本之外采样负样本
    2. 将负样本与正样本一起进行K折划分
    3. 确保任何负样本都不会与任何正样本重复
    4. 确保ID类型一致性
    
    Args:
        positive_pairs: list of tuple
        all_rna_ids: list
        all_drug_ids: list
        n_folds: int
        negative_ratio: int
        seed: int
        save_dir: str, 缓存目录
    
    Returns:
        cv_splits: list of dict, 每个dict包含 train_pairs 和 val_pairs
    """
    # 🔧 修复：确保所有ID都是字符串类型
    positive_pairs = [(str(rna), str(drug)) for rna, drug in positive_pairs]
    all_rna_ids = [str(x) for x in all_rna_ids]
    all_drug_ids = [str(x) for x in all_drug_ids]
    
    # 检查缓存
    if save_dir and os.path.exists(save_dir):
        cache_file = os.path.join(save_dir, f'cv_splits_unified_ratio{negative_ratio}_fixed.pkl')
        if os.path.exists(cache_file):
            print(f"\n📂 从缓存加载CV划分: {cache_file}")
            return load_pickle(cache_file)
    
    print(f"\n🔄 生成{n_folds}折交叉验证划分（统一负采样策略）...")
    
    # ===== 步骤1: 统一采样所有负样本 =====
    print(f"\n📊 步骤1: 从所有正样本之外统一采样负样本...")
    print(f"   - 正样本数: {len(positive_pairs)}")
    print(f"   - 负样本比例: {negative_ratio}:1")
    print(f"   - 目标负样本数: {len(positive_pairs) * negative_ratio}")
    
    negative_pairs = random_negative_sampling(
        positive_pairs, all_rna_ids, all_drug_ids,
        ratio=negative_ratio, seed=seed
    )
    
    print(f"   ✅ 成功采样 {len(negative_pairs)} 个负样本")
    
    # ===== 步骤2: 合并所有样本并打上标签 =====
    print(f"\n📊 步骤2: 合并正负样本...")
    all_pairs = positive_pairs + negative_pairs
    all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    
    print(f"   ✅ 总样本数: {len(all_pairs)} (正:{len(positive_pairs)} + 负:{len(negative_pairs)})")
    
    # ===== 步骤3: K折划分 =====
    print(f"\n📊 步骤3: 进行{n_folds}折划分...")
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_pairs)):
        print(f"\n   处理第 {fold_idx} 折...")
        
        # 划分训练集和验证集
        train_pairs = [all_pairs[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        
        val_pairs = [all_pairs[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        # 分离正负样本（用于统计）
        train_pos = [pair for pair, label in zip(train_pairs, train_labels) if label == 1]
        train_neg = [pair for pair, label in zip(train_pairs, train_labels) if label == 0]
        
        val_pos = [pair for pair, label in zip(val_pairs, val_labels) if label == 1]
        val_neg = [pair for pair, label in zip(val_pairs, val_labels) if label == 0]
        
        cv_splits.append({
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'train_pos': train_pos,
            'train_neg': train_neg,
            'val_pos': val_pos,
            'val_neg': val_neg
        })
        
        print(f"      ✅ 训练集: {len(train_pos)} 正样本 + {len(train_neg)} 负样本 = {len(train_pairs)} 总样本")
        print(f"      ✅ 验证集: {len(val_pos)} 正样本 + {len(val_neg)} 负样本 = {len(val_pairs)} 总样本")
    
    # ===== 验证：确保没有数据泄露 =====
    print(f"\n🔍 验证数据划分...")
    positive_set = set(positive_pairs)
    negative_set = set(negative_pairs)
    
    # 检查正负样本是否有重叠
    overlap = positive_set & negative_set
    if overlap:
        print(f"   ⚠️  警告: 发现 {len(overlap)} 个正负样本重叠!")
        print(f"      重叠样本示例: {list(overlap)[:5]}")
    else:
        print(f"   ✅ 正负样本无重叠")
    
    # 检查每折的验证集负样本是否在正样本中
    for fold_idx, split in enumerate(cv_splits):
        val_neg_set = set(split['val_neg'])
        leak = val_neg_set & positive_set
        if leak:
            print(f"   ⚠️  Fold {fold_idx}: 验证集中发现 {len(leak)} 个泄露样本!")
            print(f"      泄露样本示例: {list(leak)[:5]}")
        else:
            print(f"   ✅ Fold {fold_idx}: 无数据泄露")
    
    # 保存缓存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cache_file = os.path.join(save_dir, f'cv_splits_unified_ratio{negative_ratio}_fixed.pkl')
        save_pickle(cv_splits, cache_file)
        print(f"\n💾 CV划分已缓存: {cache_file}")
    
    return cv_splits


class RNADrugDataset(Dataset):
    """
    RNA-Drug交互数据集（通用版，支持lncRNA/miRNA/circRNA）
    """
    def __init__(self, pairs, rna_features_dict, drug_features_dict):
        """
        Args:
            pairs: list of tuple, [(rna_id, drug_id), ...]
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
        self.rna_features_dict = rna_features_dict
        self.drug_features_dict = drug_features_dict
        
        # 标签：前半部分是正样本(1)，后半部分是负样本(0)
        n_pos = len(pairs) // 2
        self.labels = [1] * n_pos + [0] * (len(pairs) - n_pos)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        rna_id, drug_id = self.pairs[idx]
        label = self.labels[idx]
        
        # 🆕 确保ID是字符串类型
        rna_id = str(rna_id).strip()
        drug_id = str(drug_id).strip()
        
        # 构建样本字典
        sample = {
            'rna_id': rna_id,
            'drug_id': drug_id,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # 添加RNA特征
        if 'seq' in self.rna_features_dict and config.USE_RNA_SEQ:
            sample['rna_seq'] = torch.tensor(
                self.rna_features_dict['seq'][rna_id], dtype=torch.float32
            )
        
        if 'struct' in self.rna_features_dict and config.USE_RNA_STRUCT:
            sample['rna_struct'] = torch.tensor(
                self.rna_features_dict['struct'][rna_id], dtype=torch.float32
            )
        
        if 'disease' in self.rna_features_dict and config.USE_RNA_DISEASE:
            sample['rna_disease'] = torch.tensor(
                self.rna_features_dict['disease'][rna_id], dtype=torch.float32
            )
        
        # 添加Drug特征
        if 'seq' in self.drug_features_dict and config.USE_DRUG_SEQ:
            sample['drug_seq'] = torch.tensor(
                self.drug_features_dict['seq'][drug_id], dtype=torch.float32
            )
        
        if 'graph' in self.drug_features_dict and config.USE_DRUG_STRUCT:
            sample['drug_graph'] = torch.tensor(
                self.drug_features_dict['graph'][drug_id], dtype=torch.float32
            )
        
        if 'ecfp' in self.drug_features_dict and config.USE_DRUG_STRUCT:
            sample['drug_ecfp'] = torch.tensor(
                self.drug_features_dict['ecfp'][drug_id], dtype=torch.float32
            )
        
        if 'disease' in self.drug_features_dict and config.USE_DRUG_DISEASE:
            sample['drug_disease'] = torch.tensor(
                self.drug_features_dict['disease'][drug_id], dtype=torch.float32
            )
        
        return sample


def load_all_features():
    """
    加载所有特征数据
    
    Returns:
        rna_features_dict: dict of dict
        drug_features_dict: dict of dict
    """
    print("\n📊 加载特征数据...")
    
    rna_features_dict = {}
    drug_features_dict = {}
    
    # 加载RNA特征
    if config.USE_RNA_SEQ and os.path.exists(config.RNA_SEQ_FEATURE_PATH):
        rna_features_dict['seq'] = load_features(config.RNA_SEQ_FEATURE_PATH, 'RNA_ID')
        print(f"   ✅ RNA序列特征: {len(rna_features_dict['seq'])} 条")
    
    if config.USE_RNA_STRUCT and os.path.exists(config.RNA_STRUCT_FEATURE_PATH):
        rna_features_dict['struct'] = load_features(config.RNA_STRUCT_FEATURE_PATH, 'RNA_ID')
        print(f"   ✅ RNA结构特征: {len(rna_features_dict['struct'])} 条")
    
    if config.USE_RNA_DISEASE and os.path.exists(config.RNA_DISEASE_FEATURE_PATH):
        rna_features_dict['disease'] = load_features(config.RNA_DISEASE_FEATURE_PATH, 'RNA_ID')
        print(f"   ✅ RNA疾病特征: {len(rna_features_dict['disease'])} 条")
    
    # 加载Drug特征
    if config.USE_DRUG_SEQ and os.path.exists(config.DRUG_SEQ_FEATURE_PATH):
        drug_features_dict['seq'] = load_features(config.DRUG_SEQ_FEATURE_PATH, 'CID')
        print(f"   ✅ Drug序列特征: {len(drug_features_dict['seq'])} 条")
    
    if config.USE_DRUG_STRUCT:
        if os.path.exists(config.DRUG_GRAPH_FEATURE_PATH):
            drug_features_dict['graph'] = load_features(config.DRUG_GRAPH_FEATURE_PATH, 'CID')
            print(f"   ✅ Drug图特征: {len(drug_features_dict['graph'])} 条")
        
        if os.path.exists(config.DRUG_ECFP_FEATURE_PATH):
            drug_features_dict['ecfp'] = load_features(config.DRUG_ECFP_FEATURE_PATH, 'CID')
            print(f"   ✅ Drug ECFP特征: {len(drug_features_dict['ecfp'])} 条")
    
    if config.USE_DRUG_DISEASE and os.path.exists(config.DRUG_DISEASE_FEATURE_PATH):
        drug_features_dict['disease'] = load_features(config.DRUG_DISEASE_FEATURE_PATH, 'CID')
        print(f"   ✅ Drug疾病特征: {len(drug_features_dict['disease'])} 条")
    
    return rna_features_dict, drug_features_dict


def load_positive_pairs():
    """
    🔧 修复：加载正样本对（智能处理ID类型）
    
    Returns:
        positive_pairs: list of tuple
        all_rna_ids: list
        all_drug_ids: list
    """
    print(f"\n📂 加载正样本对: {config.POSITIVE_PAIRS_PATH}")
    
    df = pd.read_excel(config.POSITIVE_PAIRS_PATH)
    
    # 🆕 智能转换ID为字符串
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