import argparse
from config import config

def parse_args():
    """解析命令行参数并更新config"""
    parser = argparse.ArgumentParser(description='RNA-Drug Interaction Prediction with LLM')
    
    # ========== 数据集选择 ==========
    parser.add_argument('--dataset', type=str, 
                       choices=['lncRNA-drug', 'miRNA-drug', 'circRNA-drug'],
                       default='miRNA-drug',
                       help='选择数据集')
    
    # ========== 模态开关 ==========
    parser.add_argument('--use_rna_seq', action='store_true',
                       help='启用RNA序列特征')
    parser.add_argument('--use_rna_struct', action='store_true',
                       help='启用RNA结构特征')
    parser.add_argument('--use_rna_disease', action='store_true',
                       help='启用RNA疾病关联特征')
    parser.add_argument('--use_drug_seq', action='store_true',
                       help='启用Drug序列特征')
    parser.add_argument('--use_drug_struct', action='store_true',
                       help='启用Drug结构特征')
    parser.add_argument('--use_drug_disease', action='store_true',
                       help='启用Drug疾病关联特征')
    
    # ========== Pooling方法 ==========
    parser.add_argument('--pooling', type=str, 
                       choices=['cls', 'learnable_weight'],
                       default='learnable_weight',
                       help='特征融合方法')
    
    # ========== 训练参数 ==========
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='早停耐心值')
    
    # ========== LoRA参数 ==========
    parser.add_argument('--use_lora', action='store_true',
                       help='启用LoRA微调')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    
    # ========== 优化器和损失函数 ========== 🆕
    parser.add_argument('--optimizer', type=str, 
                       choices=['adamw', 'adam', 'sgd', 'rmsprop'],
                       default='adamw',
                       help='优化器类型')
    
    parser.add_argument('--loss', type=str,
                       choices=['ce', 'focal', 'label_smoothing', 'weighted_ce'],
                       default='ce',
                       help='损失函数类型')
    
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                       help='Focal loss alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma参数')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing参数')
    
    # ========== 其他 ==========
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader工作进程数')
    parser.add_argument('--no_save_features', action='store_true',
                       help='不保存特征文件（加快训练速度）')
    parser.add_argument('--no_save_weights', action='store_true',
                       help='不保存模态权重（加快训练速度）')
    parser.add_argument('--no_save_model', action='store_true',
                       help='不保存模型检查点（加快训练速度，但无法恢复最佳模型）')
    
    args = parser.parse_args()  # 这行必须在所有add_argument之后
    
    # ========== 更新config ==========
    config.DATASET_NAME = args.dataset
    config.update_paths()
    
    config.SAVE_FEATURES = not args.no_save_features
    config.SAVE_WEIGHTS = not args.no_save_weights
    config.SAVE_MODEL = not args.no_save_model
    
    # 模态开关
    config.USE_RNA_SEQ = args.use_rna_seq
    config.USE_RNA_STRUCT = args.use_rna_struct
    config.USE_RNA_DISEASE = args.use_rna_disease
    config.USE_DRUG_SEQ = args.use_drug_seq
    config.USE_DRUG_STRUCT = args.use_drug_struct
    config.USE_DRUG_DISEASE = args.use_drug_disease
    
    # Pooling方法
    config.POOLING_METHOD = args.pooling
    
    # 训练参数
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.WEIGHT_DECAY = args.weight_decay
    config.MAX_EPOCHS = args.max_epochs
    config.EARLY_STOP_PATIENCE = args.early_stop_patience
    
    # LoRA参数
    config.USE_LORA = args.use_lora
    config.LORA_R = args.lora_r
    config.LORA_ALPHA = args.lora_alpha
    
    # 其他
    config.RANDOM_SEED = args.seed
    config.NUM_WORKERS = args.num_workers
    
    # 🆕 优化器和损失函数
    config.OPTIMIZER_TYPE = args.optimizer
    config.LOSS_TYPE = args.loss
    config.FOCAL_ALPHA = args.focal_alpha
    config.FOCAL_GAMMA = args.focal_gamma
    config.LABEL_SMOOTHING = args.label_smoothing
    
    return config


if __name__ == '__main__':
    # 测试
    cfg = parse_args()
    print(cfg)