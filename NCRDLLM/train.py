import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # 🔧 使用torch.amp.GradScaler
from torch.cuda.amp import GradScaler  # 🔧 修复: 使用torch.cuda.amp.GradScaler
from tqdm import tqdm
import torch.nn.functional as F  # 🆕 添加这行
from config import config
from args_parser import parse_args
from dataset import (
    load_all_features,
    load_positive_pairs,
    prepare_cv_splits,
    RNADrugDataset
)
from model import create_model
from utils import (
    set_seed,
    create_experiment_dir,
    backup_code,
    save_experiment_info,
    calculate_metrics,
    print_metrics,
    aggregate_cv_results,
    print_cv_summary,
    EarlyStopping
)
from export_utils import (
    save_fused_features,
    save_predictions,
    save_modality_weights,
    aggregate_cv_features,
    aggregate_cv_raw_features,
    aggregate_cv_predictions,
    save_fold_results
)
from visualize import (
    plot_roc_curve,
    plot_pr_curve,
    plot_cv_roc_curves,
    plot_cv_pr_curves,
    plot_metrics_comparison,
    plot_modality_tsne_features
)

# ========== 🆕 损失函数定义 ==========
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    🔧 修复: alpha现在正确支持类别权重
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # alpha可以是:
        # - 单个float: 正类权重, 负类权重自动为1-alpha
        # - list/tuple: [负类权重, 正类权重]
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1-alpha, alpha])
        else:
            self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 2] logits
            targets: [B] labels (0 or 1)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 根据目标类别选择对应的alpha
        at = self.alpha.to(inputs.device)[targets]
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    
    🔧 改进: 使用更稳定的实现
    """
    def __init__(self, classes=2, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.classes = classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 2] logits
            target: [B] labels (0 or 1)
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            # 构建平滑后的真实分布
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 计算KL散度损失
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        return loss.mean()

def create_loss_function(config, train_dataset=None):
    """
    创建损失函数
    
    🔧 修复: weighted_ce的标签提取逻辑
    """
    if config.LOSS_TYPE == 'ce':
        return nn.CrossEntropyLoss()
    
    elif config.LOSS_TYPE == 'focal':
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    
    elif config.LOSS_TYPE == 'label_smoothing':
        return LabelSmoothingLoss(classes=2, smoothing=config.LABEL_SMOOTHING)
    
    elif config.LOSS_TYPE == 'weighted_ce':
        if config.CLASS_WEIGHTS is None and train_dataset is not None:
            # 🔧 修复: 直接使用dataset的labels属性
            labels = train_dataset.labels  # ✅ 正确!
            class_counts = torch.bincount(torch.tensor(labels))
            class_weights = 1.0 / class_counts.float()
            class_weights = class_weights / class_weights.sum() * 2  # 归一化
            
            print(f"   📊 自动计算类别权重: 负类={class_weights[0]:.4f}, 正类={class_weights[1]:.4f}")
        else:
            class_weights = torch.tensor(config.CLASS_WEIGHTS or [1.0, 1.0])
        
        return nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    
    else:
        raise ValueError(f"Unknown loss type: {config.LOSS_TYPE}")


def create_optimizer(model, config):
    """
    创建优化器
    
    ✅ 这个函数是正确的,不需要修改
    """
    if config.OPTIMIZER_TYPE == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.ADAM_BETAS,
            eps=config.ADAM_EPS
        )
    
    elif config.OPTIMIZER_TYPE == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=config.ADAM_BETAS,
            eps=config.ADAM_EPS
        )
    
    elif config.OPTIMIZER_TYPE == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            nesterov=config.SGD_NESTEROV
        )
    
    elif config.OPTIMIZER_TYPE == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.LEARNING_RATE,
            alpha=0.99,
            eps=1e-8,
            weight_decay=config.WEIGHT_DECAY
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {config.OPTIMIZER_TYPE}")

def prepare_batch_features(batch, config):
    """
    从batch中提取特征并组织成模型输入格式
    
    Args:
        batch: dict, 来自DataLoader
        config: Config对象
    
    Returns:
        feature_dict: dict, 模型forward所需的参数
    """
    feature_dict = {}
    
    # RNA特征
    if config.USE_RNA_SEQ and 'rna_seq' in batch:
        feature_dict['rna_seq_features'] = batch['rna_seq'].to(config.DEVICE)
    
    if config.USE_RNA_STRUCT and 'rna_struct' in batch:
        feature_dict['rna_struct_features'] = batch['rna_struct'].to(config.DEVICE)
    
    if config.USE_RNA_DISEASE and 'rna_disease' in batch:
        feature_dict['rna_disease_features'] = batch['rna_disease'].to(config.DEVICE)
    
    # Drug特征
    if config.USE_DRUG_SEQ and 'drug_seq' in batch:
        feature_dict['drug_seq_features'] = batch['drug_seq'].to(config.DEVICE)
    
    if config.USE_DRUG_STRUCT:
        if 'drug_graph' in batch and 'drug_ecfp' in batch:
            drug_graph = batch['drug_graph'].to(config.DEVICE)
            drug_ecfp = batch['drug_ecfp'].to(config.DEVICE)
            feature_dict['drug_struct_features'] = torch.cat([drug_graph, drug_ecfp], dim=-1)
    
    if config.USE_DRUG_DISEASE and 'drug_disease' in batch:
        feature_dict['drug_disease_features'] = batch['drug_disease'].to(config.DEVICE)
    
    return feature_dict


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch, config):
    """训练一个epoch - 修复bfloat16兼容性"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in pbar:
        feature_dict = prepare_batch_features(batch, config)
        labels = batch['label'].to(config.DEVICE)
        
        # 确定autocast的dtype
        if config.USE_MIXED_PRECISION:
            if torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.float32
        
        with autocast('cuda', enabled=config.USE_MIXED_PRECISION, dtype=autocast_dtype):
            logits = model(**feature_dict)
            loss = criterion(logits, labels)
            loss = loss / config.ACCUMULATION_STEPS
        
        # bfloat16不需要GradScaler
        if config.USE_MIXED_PRECISION and autocast_dtype == torch.bfloat16:
            loss.backward()  # 直接backward
        elif config.USE_MIXED_PRECISION and autocast_dtype == torch.float16:
            scaler.scale(loss).backward()  # float16需要scaler
        else:
            loss.backward()  # float32正常backward
        
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # bfloat16不需要GradScaler
            if config.USE_MIXED_PRECISION and autocast_dtype == torch.float16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.ACCUMULATION_STEPS
        pbar.set_postfix({'loss': f'{loss.item() * config.ACCUMULATION_STEPS:.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate_and_extract(model, dataloader, criterion, fold, save_dir, config, extract_features=True):
    """
    验证并提取特征(用于可视化)
    
    Args:
        model: 模型
        dataloader: DataLoader
        criterion: 损失函数
        fold: int, 折数
        save_dir: str, 保存目录
        config: Config对象
        extract_features: bool, 是否提取特征
    
    Returns:
        metrics: dict, 评估指标
        fold_data: dict, 包含y_true和y_prob,用于绘制曲线
        feature_data: dict, 包含各模态特征和标签,用于t-SNE可视化
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_rna_ids = []
    all_drug_ids = []
    
    # 用于存储融合特征
    all_rna_fused = []
    all_drug_fused = []
    
    # 🆕 用于存储各模态特征(用于t-SNE可视化)
    all_modality_features = {
        'rna_seq': [],
        'rna_struct': [],
        'rna_disease': [],
        'drug_seq': [],
        'drug_struct': [],
        'drug_disease': [],
        'labels': []
    }
    
    total_loss = 0
    num_batches = len(dataloader)
    
    # 🔧 修复: 确定autocast的dtype
    if config.USE_MIXED_PRECISION:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating Fold {fold}"):
            feature_dict = prepare_batch_features(batch, config)
            labels = batch['label'].to(config.DEVICE)
            
            # 🔧 修复: 添加dtype参数
            with autocast('cuda', enabled=config.USE_MIXED_PRECISION, dtype=autocast_dtype):
                logits = model(**feature_dict)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().float().numpy())  # 🔧 添加.float()转换
            all_rna_ids.extend(batch['rna_id'])
            all_drug_ids.extend(batch['drug_id'])
            
            # 🆕 提取各模态特征(用于t-SNE可视化)
            if extract_features and config.SAVE_FEATURES:
                modality_feats = model.get_modality_features(**feature_dict)
                
                if modality_feats['rna_seq'] is not None:
                    all_modality_features['rna_seq'].append(modality_feats['rna_seq'].cpu())
                if modality_feats['rna_struct'] is not None:
                    all_modality_features['rna_struct'].append(modality_feats['rna_struct'].cpu())
                if modality_feats['rna_disease'] is not None:
                    all_modality_features['rna_disease'].append(modality_feats['rna_disease'].cpu())
                if modality_feats['drug_seq'] is not None:
                    all_modality_features['drug_seq'].append(modality_feats['drug_seq'].cpu())
                if modality_feats['drug_struct'] is not None:
                    all_modality_features['drug_struct'].append(modality_feats['drug_struct'].cpu())
                if modality_feats['drug_disease'] is not None:
                    all_modality_features['drug_disease'].append(modality_feats['drug_disease'].cpu())
                
                all_modality_features['labels'].append(labels.cpu())
                
                # 提取融合特征
                if config.POOLING_METHOD == 'learnable_weight':
                    rna_fused, drug_fused = model.get_fused_features(**feature_dict)
                    all_rna_fused.append(rna_fused.cpu())
                    all_drug_fused.append(drug_fused.cpu())
    
    # 拼接所有结果
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics['loss'] = total_loss / num_batches
    
    # 保存预测结果
    save_predictions(all_rna_ids, all_drug_ids, y_true, y_pred, y_prob, fold, save_dir)
    
    # 保存融合特征
    if extract_features and config.SAVE_FEATURES and config.POOLING_METHOD == 'learnable_weight' and all_rna_fused:
        rna_fused_tensor = torch.cat(all_rna_fused, dim=0)
        drug_fused_tensor = torch.cat(all_drug_fused, dim=0)
        save_fused_features(
            rna_fused_tensor, drug_fused_tensor,
            all_rna_ids, all_drug_ids,
            fold, save_dir, config
        )
    
    # 🆕 整理特征数据用于t-SNE可视化
    feature_data = None
    if extract_features and config.SAVE_FEATURES:
        feature_data = {}
        
        if all_modality_features['rna_seq']:
            feature_data['rna_seq'] = torch.cat(all_modality_features['rna_seq'], dim=0).cpu().float().numpy()
        if all_modality_features['rna_struct']:
            feature_data['rna_struct'] = torch.cat(all_modality_features['rna_struct'], dim=0).cpu().float().numpy()
        if all_modality_features['rna_disease']:
            feature_data['rna_disease'] = torch.cat(all_modality_features['rna_disease'], dim=0).cpu().float().numpy()
        if all_modality_features['drug_seq']:
            feature_data['drug_seq'] = torch.cat(all_modality_features['drug_seq'], dim=0).cpu().float().numpy()
        if all_modality_features['drug_struct']:
            feature_data['drug_struct'] = torch.cat(all_modality_features['drug_struct'], dim=0).cpu().float().numpy()
        if all_modality_features['drug_disease']:
            feature_data['drug_disease'] = torch.cat(all_modality_features['drug_disease'], dim=0).cpu().float().numpy()
        if all_rna_fused:
            feature_data['fused'] = np.concatenate([
                torch.cat(all_rna_fused, dim=0).cpu().float().numpy(),
                torch.cat(all_drug_fused, dim=0).cpu().float().numpy()
            ], axis=1)
        
        feature_data['labels'] = np.concatenate([l.numpy() for l in all_modality_features['labels']])

    
    # 用于绘制ROC/PR曲线的数据
    fold_data = {
        'y_true': y_true,
        'y_prob': y_prob[:, 1],
        'fold': fold
    }
    
    return metrics, fold_data, feature_data


def train_one_fold(fold, train_dataset, val_dataset, experiment_dir, config):
    """
    训练一折
    
    Args:
        fold: int, 折数
        train_dataset: Dataset
        val_dataset: Dataset
        experiment_dir: str, 实验目录
        config: Config对象
    
    Returns:
        best_metrics: dict, 最佳验证指标
        best_fold_data: dict, 最佳验证数据(用于绘制曲线)
        best_feature_data: dict, 最佳特征数据(用于t-SNE可视化)
    """
    print(f"\n{'='*60}")
    print(f"🔄 训练 Fold {fold}")
    print(f"{'='*60}")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )
    
    # 创建模型
    model = create_model()
    
    # 损失函数和优化器
    # 🆕 使用工厂函数创建损失函数和优化器
    criterion = create_loss_function(config, train_dataset)
    optimizer = create_optimizer(model, config)
    

    # 🔧 修复: 混合精度训练的GradScaler
    scaler = GradScaler('cuda', enabled=config.USE_MIXED_PRECISION) 
    

    # 早停
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        mode='max',
        delta=0.001
    )
    
    # 训练循环
    best_val_auc = 0
    best_metrics = None
    best_fold_data = None
    best_feature_data = None
    best_model_state = None

    for epoch in range(1, config.MAX_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.MAX_EPOCHS} ---")
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, config
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        val_metrics, fold_data, feature_data = validate_and_extract(
            model, val_loader, criterion, fold, experiment_dir, config,
            extract_features=False  # 训练过程中不提取特征
        )
        
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['auc_roc']:.4f}")
        
        # 保存最佳模型
        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc = val_metrics['auc_roc']
            best_metrics = val_metrics
            best_fold_data = fold_data
            print(f"💾 达到最佳指标")
            if config.SAVE_MODEL:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"💾 达到最佳指标，已保存模型状态")
        
        # 早停检查
        if early_stopping(val_metrics['auc_roc']):
            print(f"⏹️  早停触发! 最佳AUC: {best_val_auc:.4f}")
            break
    
    print(f"\n✅ Fold {fold} 训练完成! 最佳AUC: {best_val_auc:.4f}")

    # 🆕 加载最佳模型并提取特征
    if best_model_state is not None:
        print(f"\n🔄 加载最佳模型并提取特征...")
        model.load_state_dict(best_model_state)
        model.eval()
        
        # 提取特征
        best_metrics_final, best_fold_data_final, best_feature_data = validate_and_extract(
            model, val_loader, criterion, fold, experiment_dir, config,
            extract_features=True  # 强制提取特征
        )
        
        # 更新为最终的fold_data
        best_fold_data = best_fold_data_final
        
        # 保存模态权重
        if config.SAVE_WEIGHTS and config.POOLING_METHOD == 'learnable_weight':
            fold_dir = os.path.join(experiment_dir, f'fold_{fold}')
            save_modality_weights(model, fold_dir)
    
    return best_metrics, best_fold_data, best_feature_data


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 RNA-Drug Interaction Prediction with Multi-Modal LLM")
    print("="*70)
    
    # 1. 解析命令行参数
    parse_args()
    print(config)
    
    # 2. 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    # 3. 创建实验目录
    experiment_dir = create_experiment_dir(config.RESULTS_DIR)
    
    # 4. 备份代码
    backup_code(experiment_dir)
    
    # 5. 保存实验配置
    save_experiment_info(experiment_dir, config)
    
    # 6. 加载数据
    print("\n" + "="*60)
    print("📊 加载数据")
    print("="*60)
    
    rna_features_dict, drug_features_dict = load_all_features()
    positive_pairs, all_rna_ids, all_drug_ids = load_positive_pairs()
    
    cv_splits = prepare_cv_splits(
        positive_pairs, all_rna_ids, all_drug_ids,
        n_folds=config.N_FOLDS,
        negative_ratio=config.NEGATIVE_RATIO,
        seed=config.RANDOM_SEED,
        save_dir=config.SPLITS_DIR
    )
    
    # 7. 交叉验证训练
    print("\n" + "="*60)
    print("🔄 开始5折交叉验证")
    print("="*60)
    
    fold_results = []
    fold_data_list = []
    fold_feature_data_list = []  # 🆕 存储每折的特征数据
    
    for fold in range(config.N_FOLDS):
        train_dataset = RNADrugDataset(
            cv_splits[fold]['train_pairs'],
            rna_features_dict,
            drug_features_dict
        )
        
        val_dataset = RNADrugDataset(
            cv_splits[fold]['val_pairs'],
            rna_features_dict,
            drug_features_dict
        )
        
        # 训练
        best_metrics, fold_data, feature_data = train_one_fold(
            fold, train_dataset, val_dataset, experiment_dir, config
        )
        
        fold_results.append(best_metrics)
        fold_data_list.append(fold_data)
        fold_feature_data_list.append(feature_data)  # 🆕 保存特征数据
        
        # 打印当前折结果
        print_metrics(best_metrics, phase=f"Fold {fold} Best Validation")
    
    # 8. 汇总结果
    print("\n" + "="*60)
    print("📊 汇总交叉验证结果")
    print("="*60)
    
    aggregated = aggregate_cv_results(fold_results)
    print_cv_summary(aggregated)
    
    results = {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'config': {
            'dataset': config.DATASET_NAME,
            'model_type': config.MODEL_TYPE,
            'pooling_method': config.POOLING_METHOD,
            'enabled_modalities': config.get_enabled_modalities(),
            'use_lora': config.USE_LORA,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    save_fold_results(results, experiment_dir)
    
    # 9. 拼接五折特征和预测结果
    print("\n" + "="*60)
    print("🔗 拼接五折数据")
    print("="*60)
    
    if config.SAVE_FEATURES:
        aggregate_cv_features(experiment_dir, config.N_FOLDS)
        aggregate_cv_raw_features(experiment_dir, config.N_FOLDS)
    else:
        print("   ⚠️  特征保存已禁用，跳过特征拼接")

    aggregate_cv_predictions(experiment_dir, config.N_FOLDS)
    
    # 10. 绘制五折汇总曲线
    print("\n" + "="*60)
    print("📊 绘制五折汇总曲线")
    print("="*60)
    
    plot_cv_roc_curves(fold_data_list, os.path.join(experiment_dir, 'cv_roc_curves.png'))
    plot_cv_pr_curves(fold_data_list, os.path.join(experiment_dir, 'cv_pr_curves.png'))
    plot_metrics_comparison(fold_results, os.path.join(experiment_dir, 'metrics_comparison.png'))
    
    # 🆕 11. 绘制t-SNE可视化
    if config.SAVE_FEATURES and any(fd is not None for fd in fold_feature_data_list):
        print("\n" + "="*60)
        print("🎨 绘制t-SNE特征可视化")
        print("="*60)
        
        for fold, feature_data in enumerate(fold_feature_data_list):
            if feature_data is not None:
                fold_dir = os.path.join(experiment_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)
                plot_modality_tsne_features(feature_data, fold, fold_dir, config)
    
    # 12. 完成
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print(f"📁 结果保存在: {experiment_dir}")
    print("="*70)
    
    print(f"\n📊 最终结果:")
    print(f"   AUC-ROC: {aggregated['auc_roc']['mean']:.4f} ± {aggregated['auc_roc']['std']:.4f}")
    print(f"   PR-AUC:  {aggregated['pr_auc']['mean']:.4f} ± {aggregated['pr_auc']['std']:.4f}")
    print(f"   F1:      {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    
    print(f"\n💾 输出文件:")
    if config.SAVE_FEATURES:
        print(f"   - 融合特征: all_rna_fused_features.csv, all_drug_fused_features.csv")
        print(f"   - 原始特征: all_rna_raw_features.csv, all_drug_raw_features.csv")
        print(f"   - t-SNE可视化: fold_*/tsne_*.png")
    else:
        print(f"   - 融合特征: 未保存（使用 --no_save_features 禁用）")
        print(f"   - 原始特征: 未保存（使用 --no_save_features 禁用）")
    print(f"   - 预测结果: predictions_simple.csv, details_predictions_simple.csv")
    print(f"   - ROC/PR曲线: cv_roc_curves.png, cv_pr_curves.png")
    if config.SAVE_WEIGHTS:
        print(f"   - 模态权重: fold_*/modality_weights.json")
    else:
        print(f"   - 模态权重: 未保存（使用 --no_save_weights 禁用）")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)