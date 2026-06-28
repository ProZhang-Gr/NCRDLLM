import torch

class Config:
    """配置类（固定配置版：所有超参数均已锁定为论文最终设置）"""

    # ========== 数据集配置 ==========
    DATASET_NAME = "miRNA-drug"  # 可选: 'lncRNA-drug', 'miRNA-drug', 'circRNA-drug'

    # ========== 路径配置 ==========
    DATA_DIR = f"./data/{DATASET_NAME}"
    RESULTS_DIR = "./results"

    # 原始序列路径
    RNA_SEQUENCE_PATH = f"{DATA_DIR}/ALLRNA-seq.xlsx"
    DRUG_SMILES_PATH = f"{DATA_DIR}/ALLdrug-smiles.xlsx"

    # 序列特征路径
    RNA_SEQ_FEATURE_PATH = f"{DATA_DIR}/Features_RNAFM_RNA_640D.xlsx"
    DRUG_SEQ_FEATURE_PATH = f"{DATA_DIR}/Features_ChemBERTa_Drug_768D.xlsx"

    # 结构特征路径
    RNA_STRUCT_FEATURE_PATH = f"{DATA_DIR}/secondary_feature_RNA.xlsx"
    DRUG_GRAPH_FEATURE_PATH = f"{DATA_DIR}/ALLdrug-graph-features.xlsx"
    DRUG_ECFP_FEATURE_PATH = f"{DATA_DIR}/ALLdrug-ECFP-features.xlsx"

    # 疾病关联路径（归一化后的语义相似性矩阵）
    _rna_type = DATASET_NAME.split('-')[0]  # 提取 'miRNA', 'lncRNA', 'circRNA'
    RNA_DISEASE_FEATURE_PATH = f"{DATA_DIR}/semantic_{_rna_type}_matrix_normalized.xlsx"
    DRUG_DISEASE_FEATURE_PATH = f"{DATA_DIR}/semantic_Drug_matrix_normalized.xlsx"

    # 正样本对路径
    POSITIVE_PAIRS_PATH = f"{DATA_DIR}/responsed_RNA-drug.xlsx"

    # 负样本缓存路径
    SPLITS_DIR = f"{DATA_DIR}/splits/{DATASET_NAME}"

    # ========== 数据配置 ==========
    N_FOLDS = 5
    NEGATIVE_RATIO = 1
    RANDOM_SEED = 42

    # 负采样配置（Jaccard相似性过滤）
    JACCARD_THRESHOLD = 0.9
    MAX_SAMPLING_ATTEMPTS = 100

    # Onehot矩阵路径（仅用于计算Jaccard相似性）
    RNA_ONEHOT_MATRIX_PATH = f"{DATA_DIR}/onehot_RNA_matrix.xlsx"
    DRUG_ONEHOT_MATRIX_PATH = f"{DATA_DIR}/onehot_Drug_matrix.xlsx"

    # ========== 模型配置 ==========
    MODEL_TYPE = 'llm'

    # 特征维度
    RNA_SEQ_DIM = 640
    DRUG_SEQ_DIM = 768
    RNA_STRUCT_DIM = 128
    DRUG_GRAPH_DIM = 512
    DRUG_ECFP_DIM = 512
    DRUG_STRUCT_DIM = 1024      # Graph + ECFP
    RNA_DISEASE_DIM = 1690
    DRUG_DISEASE_DIM = 1690

    SAVE_FEATURES = True
    SAVE_WEIGHTS = True
    SAVE_MODEL = True

    # ========== LLM配置 ==========
    LLM_MODEL_ID = './llama3.1/LLM-Research/Llama-3___2-3B-Instruct'
    LLM_HIDDEN_DIM = 3072

    USE_LORA = True
    LORA_R = 64
    LORA_ALPHA = 64
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = [
        "q_proj", "v_proj",
        "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    # 分类头配置
    POOLING_METHOD = 'learnable_weight'
    CLASSIFIER_HIDDEN_DIM = 1024
    CLASSIFIER_DROPOUT = 0.3

    # ========== 训练配置 ==========
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 5

    # 优化器参数（AdamW）
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8

    # 混合精度训练
    USE_MIXED_PRECISION = True

    # DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2

    # ========== 设备配置 ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 日志配置 ==========
    LOG_INTERVAL = 10
    SAVE_BEST_MODEL = True

    # ========== 固定的模态与token顺序（不可配置） ==========
    # 模型始终使用全部 6 个模态，顺序固定
    MODALITIES = ['RNA_SEQ', 'RNA_STRUCT', 'RNA_DISEASE',
                  'DRUG_SEQ', 'DRUG_STRUCT', 'DRUG_DISEASE']

    def update_paths(self):
        """更新所有路径（当DATASET_NAME改变时调用）"""
        self.DATA_DIR = f"./data/{self.DATASET_NAME}"
        self.RNA_SEQUENCE_PATH = f"{self.DATA_DIR}/ALLRNA-seq.xlsx"
        self.DRUG_SMILES_PATH = f"{self.DATA_DIR}/ALLdrug-smiles.xlsx"
        self.RNA_SEQ_FEATURE_PATH = f"{self.DATA_DIR}/Features_RNAFM_RNA_640D.xlsx"
        self.DRUG_SEQ_FEATURE_PATH = f"{self.DATA_DIR}/Features_ChemBERTa_Drug_768D.xlsx"
        self.RNA_STRUCT_FEATURE_PATH = f"{self.DATA_DIR}/secondary_feature_RNA.xlsx"
        self.DRUG_GRAPH_FEATURE_PATH = f"{self.DATA_DIR}/ALLdrug-graph-features.xlsx"
        self.DRUG_ECFP_FEATURE_PATH = f"{self.DATA_DIR}/ALLdrug-ECFP-features.xlsx"

        _rna_type = self.DATASET_NAME.split('-')[0]
        self.RNA_DISEASE_FEATURE_PATH = f"{self.DATA_DIR}/semantic_{_rna_type}_matrix_normalized.xlsx"
        self.DRUG_DISEASE_FEATURE_PATH = f"{self.DATA_DIR}/semantic_Drug_matrix_normalized.xlsx"

        self.RNA_ONEHOT_MATRIX_PATH = f"{self.DATA_DIR}/onehot_RNA_matrix.xlsx"
        self.DRUG_ONEHOT_MATRIX_PATH = f"{self.DATA_DIR}/onehot_Drug_matrix.xlsx"

        self.POSITIVE_PAIRS_PATH = f"{self.DATA_DIR}/responsed_RNA-drug.xlsx"
        self.SPLITS_DIR = f"{self.DATA_DIR}/splits/{self.DATASET_NAME}"

    def get_enabled_modalities(self):
        """返回启用的模态列表（固定为全部 6 个）"""
        return list(self.MODALITIES)

    def get_token_order(self):
        """返回token顺序（固定顺序）"""
        return list(self.MODALITIES)

    def get_total_input_dim(self):
        """计算总输入维度"""
        return (self.RNA_SEQ_DIM + self.RNA_STRUCT_DIM + self.RNA_DISEASE_DIM +
                self.DRUG_SEQ_DIM + self.DRUG_STRUCT_DIM + self.DRUG_DISEASE_DIM)

    def __repr__(self):
        """打印配置信息"""
        config_str = "\n" + "="*60 + "\n"
        config_str += "🔧 配置信息\n"
        config_str += "="*60 + "\n"

        config_str += f"\n📊 数据集: {self.DATASET_NAME}\n"
        config_str += f"\n🎯 模态: {', '.join(self.get_enabled_modalities())}\n"

        config_str += f"\n🤖 LLM配置:\n"
        config_str += f"   - 模型: LLaMA-3.2-3B\n"
        config_str += f"   - LoRA Rank: {self.LORA_R}\n"
        config_str += f"   - LoRA模块: {len(self.LORA_TARGET_MODULES)}个\n"

        config_str += f"\n⚙️  训练配置:\n"
        config_str += f"   - Batch Size: {self.BATCH_SIZE}\n"
        config_str += f"   - 梯度累积: {self.ACCUMULATION_STEPS}步\n"
        config_str += f"   - 学习率: {self.LEARNING_RATE}\n"
        config_str += f"   - 混合精度: {'启用' if self.USE_MIXED_PRECISION else '禁用'}\n"

        config_str += f"\n💾 数据配置:\n"
        config_str += f"   - 交叉验证折数: {self.N_FOLDS}\n"
        config_str += f"   - 负样本比例: {self.NEGATIVE_RATIO}:1\n"

        config_str += "\n" + "="*60 + "\n"
        return config_str

# 创建全局配置实例
config = Config()
