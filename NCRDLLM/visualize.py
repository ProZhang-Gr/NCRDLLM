import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curve(y_true, y_prob, save_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_prob, save_path):
    """绘制PR曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cv_roc_curves(fold_data_list, save_path):
    """绘制五折交叉验证的ROC曲线(5条折线 + 1条平均线)"""
    plt.figure(figsize=(10, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold_data in fold_data_list:
        y_true = fold_data['y_true']
        y_prob = fold_data['y_prob']
        fold_idx = fold_data['fold']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        
        plt.plot(fpr, tpr, lw=1.5, alpha=0.6, 
                label=f'Fold {fold_idx} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', 
             label='Random', alpha=0.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='red', lw=3,
            label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='red', 
                     alpha=0.2, label='± 1 std. dev.')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('5-Fold Cross-Validation ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 五折ROC曲线已保存: {save_path}")


def plot_cv_pr_curves(fold_data_list, save_path):
    """绘制五折交叉验证的PR曲线(5条折线 + 1条平均线)"""
    plt.figure(figsize=(10, 8))
    
    precisions = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    
    for fold_data in fold_data_list:
        y_true = fold_data['y_true']
        y_prob = fold_data['y_prob']
        fold_idx = fold_data['fold']
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        aucs.append(pr_auc)
        
        precision = precision[::-1]
        recall = recall[::-1]
        
        precision_interp = np.interp(mean_recall, recall, precision)
        precisions.append(precision_interp)
        
        plt.plot(recall, precision, lw=1.5, alpha=0.6,
                label=f'Fold {fold_idx} (AP = {pr_auc:.4f})')
    
    mean_precision = np.mean(precisions, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_recall, mean_precision, color='red', lw=3,
            label=f'Mean PR (AP = {mean_auc:.4f} ± {std_auc:.4f})')
    
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, 
                     color='red', alpha=0.2, label='± 1 std. dev.')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('5-Fold Cross-Validation Precision-Recall Curves', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 五折PR曲线已保存: {save_path}")


def plot_metrics_comparison(fold_results, save_path):
    """绘制各折指标对比图"""
    metrics_names = ['accuracy', 'auc_roc', 'pr_auc', 'f1', 'precision', 'recall']
    metrics_display = {
        'accuracy': 'Accuracy',
        'auc_roc': 'AUC-ROC',
        'pr_auc': 'PR-AUC',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    n_folds = len(fold_results)
    fold_indices = list(range(n_folds))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = [fold_results[i][metric] for i in range(n_folds)]
        mean_val = np.mean(values)
        
        ax.bar(fold_indices, values, alpha=0.7, color='steelblue')
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean = {mean_val:.4f}')
        ax.set_xlabel('Fold', fontsize=10)
        ax.set_ylabel(metrics_display[metric], fontsize=10)
        ax.set_title(metrics_display[metric], fontsize=12, fontweight='bold')
        ax.set_xticks(fold_indices)
        ax.set_xticklabels([f'{i}' for i in fold_indices])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 指标对比图已保存: {save_path}")