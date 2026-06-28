import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path


def save_fused_features(rna_features, drug_features, rna_ids, drug_ids, fold, save_dir, config):
    """
    保存融合后的RNA和Drug特征（维度动态获取）
    
    Args:
        rna_features: torch.Tensor, shape [N, LLM_HIDDEN_DIM]
        drug_features: torch.Tensor, shape [M, LLM_HIDDEN_DIM]
        rna_ids: list of str, RNA IDs
        drug_ids: list of str, Drug CIDs
        fold: int, 折数
        save_dir: str, 保存目录
        config: Config对象，用于获取LLM_HIDDEN_DIM
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 🔧 修复：从config动态获取维度
    hidden_dim = config.LLM_HIDDEN_DIM
    
    # 保存RNA特征
    rna_df = pd.DataFrame(
        rna_features.cpu().numpy(),
        columns=[f'dim_{i}' for i in range(hidden_dim)]
    )
    rna_df.insert(0, 'RNA_ID', rna_ids)
    rna_path = os.path.join(save_dir, f'fold_{fold}_rna_fused_features.csv')
    rna_df.to_csv(rna_path, index=False)
    
    # 保存Drug特征
    drug_df = pd.DataFrame(
        drug_features.cpu().numpy(),
        columns=[f'dim_{i}' for i in range(hidden_dim)]
    )
    drug_df.insert(0, 'CID', drug_ids)
    drug_path = os.path.join(save_dir, f'fold_{fold}_drug_fused_features.csv')
    drug_df.to_csv(drug_path, index=False)


def save_raw_modality_features(batch_data, fold, save_dir, config):
    """
    🆕 保存原始模态特征
    
    Args:
        batch_data: list of dict, 每个dict包含一个样本的所有特征
        fold: int, 折数
        save_dir: str, 保存目录
        config: Config对象
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取所有数据
    rna_ids = [item['rna_id'] for item in batch_data]
    drug_ids = [item['drug_id'] for item in batch_data]
    labels = [item['label'] for item in batch_data]
    
    # 构建RNA特征DataFrame
    rna_data = {'RNA_ID': rna_ids, 'label': labels}
    
    if True:
        rna_seq_features = np.array([item['rna_seq'] for item in batch_data])
        for i in range(rna_seq_features.shape[1]):
            rna_data[f'rna_seq_dim_{i}'] = rna_seq_features[:, i]
    
    if True:
        rna_struct_features = np.array([item['rna_struct'] for item in batch_data])
        for i in range(rna_struct_features.shape[1]):
            rna_data[f'rna_struct_dim_{i}'] = rna_struct_features[:, i]
    
    if True:
        rna_disease_features = np.array([item['rna_disease'] for item in batch_data])
        for i in range(rna_disease_features.shape[1]):
            rna_data[f'rna_disease_dim_{i}'] = rna_disease_features[:, i]
    
    rna_df = pd.DataFrame(rna_data)
    rna_path = os.path.join(save_dir, f'fold_{fold}_rna_raw_features.csv')
    rna_df.to_csv(rna_path, index=False)
    print(f"   💾 已保存RNA原始特征: {rna_path}")
    
    # 构建Drug特征DataFrame
    drug_data = {'CID': drug_ids, 'label': labels}
    
    if True:
        drug_seq_features = np.array([item['drug_seq'] for item in batch_data])
        for i in range(drug_seq_features.shape[1]):
            drug_data[f'drug_seq_dim_{i}'] = drug_seq_features[:, i]
    
    if True:
        drug_graph_features = np.array([item['drug_graph'] for item in batch_data])
        drug_ecfp_features = np.array([item['drug_ecfp'] for item in batch_data])
        for i in range(drug_graph_features.shape[1]):
            drug_data[f'drug_graph_dim_{i}'] = drug_graph_features[:, i]
        for i in range(drug_ecfp_features.shape[1]):
            drug_data[f'drug_ecfp_dim_{i}'] = drug_ecfp_features[:, i]
    
    if True:
        drug_disease_features = np.array([item['drug_disease'] for item in batch_data])
        for i in range(drug_disease_features.shape[1]):
            drug_data[f'drug_disease_dim_{i}'] = drug_disease_features[:, i]
    
    drug_df = pd.DataFrame(drug_data)
    drug_path = os.path.join(save_dir, f'fold_{fold}_drug_raw_features.csv')
    drug_df.to_csv(drug_path, index=False)
    print(f"   💾 已保存Drug原始特征: {drug_path}")


def save_predictions(rna_ids, drug_ids, y_true, y_pred, y_prob, fold, save_dir):
    """
    保存预测结果
    
    Args:
        rna_ids: list of str
        drug_ids: list of str
        y_true: numpy array
        y_pred: numpy array
        y_prob: numpy array, shape [N, 2]
        fold: int
        save_dir: str
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存 predictions_simple.csv
    simple_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_score': y_prob[:, 1]  # 正类概率
    })
    simple_path = os.path.join(save_dir, f'fold_{fold}_predictions_simple.csv')
    simple_df.to_csv(simple_path, index=False)
    
    # 保存 details_predictions_simple.csv
    details_df = pd.DataFrame({
        'RNA_ID': rna_ids,
        'CID': drug_ids,
        'true_label': y_true,
        'predicted_score': y_prob[:, 1]
    })
    details_path = os.path.join(save_dir, f'fold_{fold}_details_predictions_simple.csv')
    details_df.to_csv(details_path, index=False)


def save_modality_weights(model, save_dir):
    """
    保存模态权重
    
    Args:
        model: 模型对象
        save_dir: str, 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 🔧 修复：调用模型的get_modality_weights方法
    if hasattr(model, 'get_modality_weights'):
        weights_dict = model.get_modality_weights()
        
        if not weights_dict:
            print(f"\n⚠️  当前pooling方法不支持权重导出")
            return
        
        # 保存为JSON
        weights_path = os.path.join(save_dir, 'modality_weights.json')
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(weights_dict, f, indent=4, ensure_ascii=False)
        
        print(f"\n💾 已保存模态权重: {weights_path}")
        if 'rna_weights' in weights_dict:
            print(f"   RNA权重: {weights_dict['rna_weights']}")
        if 'drug_weights' in weights_dict:
            print(f"   Drug权重: {weights_dict['drug_weights']}")
    else:
        print(f"\n⚠️  模型不支持权重导出")


def aggregate_cv_features(save_dir, n_folds=5):
    """
    拼接五折的特征文件
    
    Args:
        save_dir: str, 保存目录
        n_folds: int, 折数
    """
    print("\n🔗 拼接五折特征文件...")
    
    # 拼接RNA特征
    rna_dfs = []
    for fold in range(n_folds):
        rna_path = os.path.join(save_dir, f'fold_{fold}_rna_fused_features.csv')
        if os.path.exists(rna_path):
            df = pd.read_csv(rna_path)
            rna_dfs.append(df)
    
    if rna_dfs:
        all_rna = pd.concat(rna_dfs, ignore_index=True)
        all_rna_path = os.path.join(save_dir, 'all_rna_fused_features.csv')
        all_rna.to_csv(all_rna_path, index=False)
        print(f"   ✅ 拼接完成: {all_rna_path} (共 {len(all_rna)} 条)")
    
    # 拼接Drug特征
    drug_dfs = []
    for fold in range(n_folds):
        drug_path = os.path.join(save_dir, f'fold_{fold}_drug_fused_features.csv')
        if os.path.exists(drug_path):
            df = pd.read_csv(drug_path)
            drug_dfs.append(df)
    
    if drug_dfs:
        all_drug = pd.concat(drug_dfs, ignore_index=True)
        all_drug_path = os.path.join(save_dir, 'all_drug_fused_features.csv')
        all_drug.to_csv(all_drug_path, index=False)
        print(f"   ✅ 拼接完成: {all_drug_path} (共 {len(all_drug)} 条)")


def aggregate_cv_raw_features(save_dir, n_folds=5):
    """
    🆕 拼接五折的原始特征文件
    
    Args:
        save_dir: str, 保存目录
        n_folds: int, 折数
    """
    print("\n🔗 拼接五折原始特征文件...")
    
    # 拼接RNA原始特征
    rna_dfs = []
    for fold in range(n_folds):
        rna_path = os.path.join(save_dir, f'fold_{fold}_rna_raw_features.csv')
        if os.path.exists(rna_path):
            df = pd.read_csv(rna_path)
            rna_dfs.append(df)
    
    if rna_dfs:
        all_rna = pd.concat(rna_dfs, ignore_index=True)
        all_rna_path = os.path.join(save_dir, 'all_rna_raw_features.csv')
        all_rna.to_csv(all_rna_path, index=False)
        print(f"   ✅ 拼接完成: {all_rna_path} (共 {len(all_rna)} 条)")
    
    # 拼接Drug原始特征
    drug_dfs = []
    for fold in range(n_folds):
        drug_path = os.path.join(save_dir, f'fold_{fold}_drug_raw_features.csv')
        if os.path.exists(drug_path):
            df = pd.read_csv(drug_path)
            drug_dfs.append(df)
    
    if drug_dfs:
        all_drug = pd.concat(drug_dfs, ignore_index=True)
        all_drug_path = os.path.join(save_dir, 'all_drug_raw_features.csv')
        all_drug.to_csv(all_drug_path, index=False)
        print(f"   ✅ 拼接完成: {all_drug_path} (共 {len(all_drug)} 条)")


def aggregate_cv_predictions(save_dir, n_folds=5):
    """
    拼接五折的预测结果
    
    Args:
        save_dir: str, 保存目录
        n_folds: int, 折数
    """
    print("\n🔗 拼接五折预测结果...")
    
    # 拼接 predictions_simple
    pred_dfs = []
    for fold in range(n_folds):
        pred_path = os.path.join(save_dir, f'fold_{fold}_predictions_simple.csv')
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            pred_dfs.append(df)
    
    if pred_dfs:
        all_pred = pd.concat(pred_dfs, ignore_index=True)
        all_pred_path = os.path.join(save_dir, 'predictions_simple.csv')
        all_pred.to_csv(all_pred_path, index=False)
        print(f"   ✅ 拼接完成: {all_pred_path} (共 {len(all_pred)} 条)")
    
    # 拼接 details_predictions_simple
    details_dfs = []
    for fold in range(n_folds):
        details_path = os.path.join(save_dir, f'fold_{fold}_details_predictions_simple.csv')
        if os.path.exists(details_path):
            df = pd.read_csv(details_path)
            details_dfs.append(df)
    
    if details_dfs:
        all_details = pd.concat(details_dfs, ignore_index=True)
        all_details_path = os.path.join(save_dir, 'details_predictions_simple.csv')
        all_details.to_csv(all_details_path, index=False)
        print(f"   ✅ 拼接完成: {all_details_path} (共 {len(all_details)} 条)")


def save_fold_results(fold_results, save_dir):
    """
    保存每折的评估指标
    
    Args:
        fold_results: list of dict
        save_dir: str
    """
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, 'fold_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(fold_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 已保存每折结果: {results_path}")