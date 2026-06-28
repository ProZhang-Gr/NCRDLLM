import os
import json
import pickle
import random
import shutil
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    average_precision_score,
    confusion_matrix
)


def set_seed(seed):
    """设置随机种子,确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir="./results"):
    """
    创建带时间戳的实验目录
    
    Returns:
        experiment_dir: 实验目录路径,例如 ./results/exp_20250116_143522/
    """
    timestamp = get_timestamp()
    experiment_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n📁 实验目录已创建: {experiment_dir}")
    return experiment_dir


def backup_code(experiment_dir):
    """
    备份当前运行的代码到实验目录
    
    Args:
        experiment_dir: 实验目录路径
    """
    code_backup_dir = os.path.join(experiment_dir, 'code_snapshot')
    os.makedirs(code_backup_dir, exist_ok=True)
    
    # 需要备份的文件列表
    files_to_backup = [
        'train.py',
        'model.py',
        'baseline.py',  # 🆕 新增
        'dataset.py',
        'config.py',
        'utils.py',
        'visualize.py',
    ]
    
    print(f"\n💾 备份代码到: {code_backup_dir}")
    
    for filename in files_to_backup:
        if os.path.exists(filename):
            shutil.copy2(filename, os.path.join(code_backup_dir, filename))
            print(f"   ✅ {filename}")
        else:
            print(f"   ⚠️  {filename} 不存在,跳过")
    
    print(f"✅ 代码备份完成\n")


def save_experiment_info(experiment_dir, config):
    """
    保存实验配置和环境信息
    
    Args:
        experiment_dir: 实验目录路径
        config: 配置对象
    """
    info = {
        'timestamp': get_timestamp(),
        'experiment_dir': experiment_dir,
        
        # 配置信息
        'config': {
            'model_type': config.MODEL_TYPE,
            'n_folds': config.N_FOLDS,
            'negative_ratio': config.NEGATIVE_RATIO,
            'batch_size': config.BATCH_SIZE,
            'accumulation_steps': config.ACCUMULATION_STEPS,  # 🆕
            'effective_batch_size': config.BATCH_SIZE * config.ACCUMULATION_STEPS,  # 🆕
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'max_epochs': config.MAX_EPOCHS,
            'early_stop_patience': config.EARLY_STOP_PATIENCE,
            'random_seed': config.RANDOM_SEED,
            'mixed_precision': config.USE_MIXED_PRECISION,  # 🆕
        },
        
        # 环境信息
        'environment': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device': str(config.DEVICE),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        
        # 🔧 修复：使用正确的变量名
        'data_paths': {
            'rna_seq_feature': config.RNA_SEQ_FEATURE_PATH,  # ✅ 修复
            'drug_seq_feature': config.DRUG_SEQ_FEATURE_PATH,  # ✅ 修复
            'rna_struct_feature': config.RNA_STRUCT_FEATURE_PATH,
            'drug_graph_feature': config.DRUG_GRAPH_FEATURE_PATH,  # ✅ 修复
            'drug_ecfp_feature': config.DRUG_ECFP_FEATURE_PATH,  # ✅ 修复
            'rna_disease_feature': config.RNA_DISEASE_FEATURE_PATH,  # 🆕
            'drug_disease_feature': config.DRUG_DISEASE_FEATURE_PATH,  # 🆕
            'positive_pairs': config.POSITIVE_PAIRS_PATH,
        },
        
        # 模态信息（固定使用全部 6 个模态）
        'modalities': {
            'enabled': config.get_enabled_modalities(),
        }
    }
    
    # 根据模型类型添加特定配置
    if config.MODEL_TYPE == 'llm':
        info['config'].update({
            'llm_model_id': config.LLM_MODEL_ID,
            'llm_hidden_dim': config.LLM_HIDDEN_DIM,  # 🆕
            'use_lora': config.USE_LORA,
            'lora_r': config.LORA_R if config.USE_LORA else None,
            'lora_alpha': config.LORA_ALPHA if config.USE_LORA else None,
            'lora_target_modules': config.LORA_TARGET_MODULES if config.USE_LORA else None,
            'pooling_method': config.POOLING_METHOD,
            'classifier_hidden_dim': config.CLASSIFIER_HIDDEN_DIM,  # 🆕
        })
    elif config.MODEL_TYPE == 'baseline':
        info['config'].update({
            'total_input_dim': config.get_total_input_dim(),
        })
    
    # 保存为JSON
    info_path = os.path.join(experiment_dir, 'experiment_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    
    print(f"📝 实验信息已保存: {info_path}\n")


def calculate_metrics(y_true, y_pred, y_prob):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签 (numpy array)
        y_pred: 预测标签 (numpy array)
        y_prob: 预测概率 (numpy array, shape: [N, 2])
    
    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_prob[:, 1]),
        'pr_auc': average_precision_score(y_true, y_prob[:, 1]),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = {
        'TN': int(tn), 'FP': int(fp),
        'FN': int(fn), 'TP': int(tp)
    }
    
    return metrics


def print_metrics(metrics, phase="Validation"):
    """打印指标"""
    print(f"\n{'='*50}")
    print(f"{phase} Metrics:")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['TN']:4d}  |  FP: {cm['FP']:4d}")
    print(f"  FN: {cm['FN']:4d}  |  TP: {cm['TP']:4d}")
    print(f"{'='*50}\n")


def save_results(results, save_path):
    """保存结果到JSON文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def load_results(load_path):
    """从JSON文件加载结果"""
    with open(load_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def save_pickle(data, path):
    """
    保存数据为pickle格式
    
    Args:
        data: 要保存的数据
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 已保存: {path}")


def load_pickle(path):
    """
    从pickle文件加载数据
    
    Args:
        path: pickle文件路径
    
    Returns:
        加载的数据
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def aggregate_cv_results(fold_results):
    """
    汇总交叉验证结果
    
    Args:
        fold_results: list of dict, 每折的结果
    
    Returns:
        dict: 平均值和标准差
    """
    metrics_names = ['accuracy', 'auc_roc', 'pr_auc', 'f1', 'precision', 'recall']
    aggregated = {}
    
    for metric in metrics_names:
        values = [fold[metric] for fold in fold_results]
        aggregated[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    return aggregated


def print_cv_summary(aggregated_results):
    """打印交叉验证汇总结果"""
    print("\n" + "="*60)
    print("🎯 5-Fold Cross-Validation Summary")
    print("="*60)
    
    metrics_order = ['accuracy', 'auc_roc', 'pr_auc', 'f1', 'precision', 'recall']
    metrics_display = {
        'accuracy': 'Accuracy',
        'auc_roc': 'AUC-ROC',
        'pr_auc': 'PR-AUC',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    for metric in metrics_order:
        if metric in aggregated_results:
            mean = aggregated_results[metric]['mean']
            std = aggregated_results[metric]['std']
            print(f"{metrics_display[metric]:12s}: {mean:.4f} ± {std:.4f}")
    
    print("="*60 + "\n")


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, mode='max', delta=0):
        """
        Args:
            patience: 容忍的轮数
            mode: 'max' 或 'min'
            delta: 最小变化阈值
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False