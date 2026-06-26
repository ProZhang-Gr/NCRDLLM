import torch

class Config:
    """配置类 - 重构版"""
    
    # ========== 数据集配置 ========== 
    DATASET_NAME = "miRNA-drug"  # 🆕 可选: 'lncRNA-drug', 'miRNA-drug', 'circRNA-drug'
    
    # ========== 路径配置 ==========
    DATA_DIR = f"./data/{DATASET_NAME}"  # 🆕 动态路径
    RESULTS_DIR = "./results"
    
    # 原始序列路径
    RNA_SEQUENCE_PATH = f"{DATA_DIR}/ALLRNA-seq.xlsx"  # 🆕 改名
    DRUG_SMILES_PATH = f"{DATA_DIR}/ALLdrug-smiles.xlsx"

    # 序列特征路径
    RNA_SEQ_FEATURE_PATH = f"{DATA_DIR}/Features_RNAFM_RNA_640D.xlsx"  # 🆕 改名
    DRUG_SEQ_FEATURE_PATH = f"{DATA_DIR}/Features_ChemBERTa_Drug_768D.xlsx"
    
    # 结构特征路径
    RNA_STRUCT_FEATURE_PATH = f"{DATA_DIR}/secondary_feature_RNA.xlsx"  # 🆕 改名
    DRUG_GRAPH_FEATURE_PATH = f"{DATA_DIR}/ALLdrug-graph-features.xlsx"
    DRUG_ECFP_FEATURE_PATH = f"{DATA_DIR}/ALLdrug-ECFP-features.xlsx"
    
    # 🆕 疾病关联路径（归一化后的语义相似性矩阵）
    # 根据数据集类型动态选择对应的归一化矩阵
    _rna_type = DATASET_NAME.split('-')[0]  # 提取 'miRNA', 'lncRNA', 'circRNA'
    RNA_DISEASE_FEATURE_PATH = f"{DATA_DIR}/semantic_{_rna_type}_matrix_normalized.xlsx"
    DRUG_DISEASE_FEATURE_PATH = f"{DATA_DIR}/semantic_Drug_matrix_normalized.xlsx"

    # 正样本对路径
    POSITIVE_PAIRS_PATH = f"{DATA_DIR}/responsed_RNA-drug.xlsx"  # 🆕 改名
    
    # 负样本缓存路径
    SPLITS_DIR = f"{DATA_DIR}/splits/{DATASET_NAME}"  # 🆕 动态路径
    
    # ========== 数据配置 ==========
    N_FOLDS = 5
    NEGATIVE_RATIO = 1
    RANDOM_SEED = 42
    
    # 🆕 负采样配置（Jaccard相似性过滤）
    JACCARD_THRESHOLD = 0.9  # Jaccard相似度阈值，超过此值拒绝作为负样本
    MAX_SAMPLING_ATTEMPTS = 100  # 单个负样本的最大尝试次数
    
    # 🆕 Onehot矩阵路径（用于计算Jaccard相似性，不用于特征）
    RNA_ONEHOT_MATRIX_PATH = f"{DATA_DIR}/onehot_RNA_matrix.xlsx"
    DRUG_ONEHOT_MATRIX_PATH = f"{DATA_DIR}/onehot_Drug_matrix.xlsx"
    
    # ========== 模型配置 ==========
    MODEL_TYPE = 'llm'  # 'baseline' 或 'llm'
    
    # 🆕 Baseline版本配置（仅当MODEL_TYPE='baseline'时有效）
    BASELINE_VERSION = 'weak'  # 'strong' / 'medium' / 'weak' / 'simple'
    
    # 🆕 Token顺序配置（用于位置编码消融实验）
    # 0: 默认顺序, 1-19: 代表性排列, -1: 随机顺序
    TOKEN_ORDER_ID = 0
    
    # 🆕 模态开关配置（用于消融实验）
    USE_RNA_SEQ = True          # RNA序列embedding
    USE_RNA_STRUCT = True       # RNA结构特征
    USE_RNA_DISEASE = True      # RNA疾病关联
    USE_DRUG_SEQ = True         # Drug序列embedding
    USE_DRUG_STRUCT = True      # Drug结构特征（Graph+ECFP）
    USE_DRUG_DISEASE = True     # Drug疾病关联
    
    # 特征维度
    RNA_SEQ_DIM = 640
    DRUG_SEQ_DIM = 768
    RNA_STRUCT_DIM = 128
    DRUG_GRAPH_DIM = 512
    DRUG_ECFP_DIM = 512
    DRUG_STRUCT_DIM = 1024      # Graph + ECFP
    RNA_DISEASE_DIM = 1690      # 🆕 疾病关联维度
    DRUG_DISEASE_DIM = 1690     # 🆕 疾病关联维度
    
    SAVE_FEATURES = True         # 是否保存特征文件
    SAVE_WEIGHTS = True          # 是否保存模态权重
    SAVE_MODEL = True            # 是否保存模型检查点
    # ========== LLM配置 ==========
    #LLM_MODEL_ID = "./llama3.1/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    #LLM_HIDDEN_DIM = 4096       # LLaMA-3.1-8B的embedding维度
    LLM_MODEL_ID = './llama3.1/LLM-Research/Llama-3___2-3B-Instruct'
    LLM_HIDDEN_DIM = 3072
    # LLM_MODEL_ID = './llama3.1/LLM-Research/Llama-3___2-1B-Instruct'
    # LLM_HIDDEN_DIM = 2048
    # 🆕 LoRA配置（扩展版）
    USE_LORA = True
    LORA_R = 64                  
    LORA_ALPHA = 64              
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = [
        "q_proj", "v_proj",      
        "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj" 
    ]
    # "q_proj", "v_proj",      
        # "k_proj", "o_proj",
    # 🆕 分类头配置
    POOLING_METHOD = 'learnable_weight'  # 🆕 改名: 'learnable_weight' / 'cls'
    CLASSIFIER_HIDDEN_DIM = 1024 
    CLASSIFIER_DROPOUT = 0.3

    # ========== 训练配置 ==========
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 4       
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 5
        # 🆕 优化器配置
    OPTIMIZER_TYPE = 'adamw'  # 可选: 'adamw', 'adam', 'sgd', 'rmsprop'
    
    # SGD特定参数
    SGD_MOMENTUM = 0.9
    SGD_NESTEROV = True
    
    # Adam/AdamW特定参数
    ADAM_BETAS = (0.9, 0.999)
    ADAM_EPS = 1e-8
    
    # 🆕 损失函数配置
    LOSS_TYPE = 'ce'  # 可选: 'ce', 'focal', 'label_smoothing', 'weighted_ce'
    
    # Focal Loss参数
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Label Smoothing参数
    LABEL_SMOOTHING = 0.1
    
    # Weighted CE参数
    CLASS_WEIGHTS = None  # None表示自动计算
    # 🆕 混合精度训练
    USE_MIXED_PRECISION = True   
    
    # 🆕 DataLoader优化
    NUM_WORKERS = 4              
    PIN_MEMORY = True            
    PREFETCH_FACTOR = 2          
    
    # ========== 设备配置 ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== 日志配置 ==========
    LOG_INTERVAL = 10
    SAVE_BEST_MODEL = True
    
    def update_paths(self):
        """🆕 更新所有路径（当DATASET_NAME改变时调用）"""
        self.DATA_DIR = f"./data/{self.DATASET_NAME}"
        self.RNA_SEQUENCE_PATH = f"{self.DATA_DIR}/ALLRNA-seq.xlsx"
        self.DRUG_SMILES_PATH = f"{self.DATA_DIR}/ALLdrug-smiles.xlsx"
        self.RNA_SEQ_FEATURE_PATH = f"{self.DATA_DIR}/Features_RNAFM_RNA_640D.xlsx"
        self.DRUG_SEQ_FEATURE_PATH = f"{self.DATA_DIR}/Features_ChemBERTa_Drug_768D.xlsx"
        self.RNA_STRUCT_FEATURE_PATH = f"{self.DATA_DIR}/secondary_feature_RNA.xlsx"
        self.DRUG_GRAPH_FEATURE_PATH = f"{self.DATA_DIR}/ALLdrug-graph-features.xlsx"
        self.DRUG_ECFP_FEATURE_PATH = f"{self.DATA_DIR}/ALLdrug-ECFP-features.xlsx"
        
        # 🆕 归一化疾病关联矩阵（用于特征）
        _rna_type = self.DATASET_NAME.split('-')[0]
        self.RNA_DISEASE_FEATURE_PATH = f"{self.DATA_DIR}/semantic_{_rna_type}_matrix_normalized.xlsx"
        self.DRUG_DISEASE_FEATURE_PATH = f"{self.DATA_DIR}/semantic_Drug_matrix_normalized.xlsx"
        
        # 🆕 Onehot矩阵（用于Jaccard相似性计算）
        self.RNA_ONEHOT_MATRIX_PATH = f"{self.DATA_DIR}/onehot_RNA_matrix.xlsx"
        self.DRUG_ONEHOT_MATRIX_PATH = f"{self.DATA_DIR}/onehot_Drug_matrix.xlsx"
        
        self.POSITIVE_PAIRS_PATH = f"{self.DATA_DIR}/responsed_RNA-drug.xlsx"
        self.SPLITS_DIR = f"{self.DATA_DIR}/splits/{self.DATASET_NAME}"
    
    def get_enabled_modalities(self):
        """返回启用的模态列表"""
        modalities = []
        if self.USE_RNA_SEQ:
            modalities.append('RNA_SEQ')
        if self.USE_RNA_STRUCT:
            modalities.append('RNA_STRUCT')
        if self.USE_RNA_DISEASE:
            modalities.append('RNA_DISEASE')
        if self.USE_DRUG_SEQ:
            modalities.append('DRUG_SEQ')
        if self.USE_DRUG_STRUCT:
            modalities.append('DRUG_STRUCT')
        if self.USE_DRUG_DISEASE:
            modalities.append('DRUG_DISEASE')
        return modalities
    
    def get_token_order(self):
        """
        🆕 根据TOKEN_ORDER_ID返回token顺序列表
        
        Returns:
            list of str: 模态名称的顺序列表
        """
        # 定义20种代表性排列
        token_orders = {
            # 0. 默认顺序 (原始)
            0: ['RNA_SEQ', 'RNA_STRUCT', 'RNA_DISEASE', 'DRUG_SEQ', 'DRUG_STRUCT', 'DRUG_DISEASE'],
            
            # 1. 完全反转
            1: ['DRUG_DISEASE', 'DRUG_STRUCT', 'DRUG_SEQ', 'RNA_DISEASE', 'RNA_STRUCT', 'RNA_SEQ'],
            
            # 2-3. RNA-Drug交替
            2: ['RNA_SEQ', 'DRUG_SEQ', 'RNA_STRUCT', 'DRUG_STRUCT', 'RNA_DISEASE', 'DRUG_DISEASE'],
            3: ['DRUG_SEQ', 'RNA_SEQ', 'DRUG_STRUCT', 'RNA_STRUCT', 'DRUG_DISEASE', 'RNA_DISEASE'],
            
            # 4-6. 特征类型优先
            4: ['RNA_SEQ', 'DRUG_SEQ', 'RNA_STRUCT', 'RNA_DISEASE', 'DRUG_STRUCT', 'DRUG_DISEASE'],  # 序列优先
            5: ['RNA_STRUCT', 'DRUG_STRUCT', 'RNA_SEQ', 'DRUG_SEQ', 'RNA_DISEASE', 'DRUG_DISEASE'],  # 结构优先
            6: ['RNA_DISEASE', 'DRUG_DISEASE', 'RNA_SEQ', 'DRUG_SEQ', 'RNA_STRUCT', 'DRUG_STRUCT'],  # 疾病优先
            
            # 7-9. Drug在前
            7: ['DRUG_SEQ', 'DRUG_STRUCT', 'DRUG_DISEASE', 'RNA_SEQ', 'RNA_STRUCT', 'RNA_DISEASE'],
            8: ['DRUG_STRUCT', 'DRUG_SEQ', 'DRUG_DISEASE', 'RNA_STRUCT', 'RNA_SEQ', 'RNA_DISEASE'],
            9: ['DRUG_DISEASE', 'DRUG_SEQ', 'DRUG_STRUCT', 'RNA_DISEASE', 'RNA_SEQ', 'RNA_STRUCT'],
            
            # 10-12. 混合交错
            10: ['RNA_SEQ', 'RNA_STRUCT', 'DRUG_SEQ', 'DRUG_STRUCT', 'RNA_DISEASE', 'DRUG_DISEASE'],
            11: ['RNA_SEQ', 'DRUG_SEQ', 'RNA_STRUCT', 'RNA_DISEASE', 'DRUG_STRUCT', 'DRUG_DISEASE'],
            12: ['RNA_DISEASE', 'RNA_SEQ', 'DRUG_DISEASE', 'DRUG_SEQ', 'RNA_STRUCT', 'DRUG_STRUCT'],
            
            # 13-14. 随机示例
            13: ['DRUG_STRUCT', 'RNA_SEQ', 'RNA_DISEASE', 'DRUG_SEQ', 'RNA_STRUCT', 'DRUG_DISEASE'],
            14: ['RNA_STRUCT', 'DRUG_DISEASE', 'RNA_SEQ', 'DRUG_STRUCT', 'DRUG_SEQ', 'RNA_DISEASE'],
            
            # 15-19. 更多变体
            15: ['RNA_SEQ', 'DRUG_STRUCT', 'RNA_DISEASE', 'DRUG_SEQ', 'RNA_STRUCT', 'DRUG_DISEASE'],
            16: ['DRUG_SEQ', 'RNA_STRUCT', 'DRUG_DISEASE', 'RNA_SEQ', 'DRUG_STRUCT', 'RNA_DISEASE'],
            17: ['RNA_DISEASE', 'DRUG_SEQ', 'RNA_STRUCT', 'DRUG_DISEASE', 'RNA_SEQ', 'DRUG_STRUCT'],
            18: ['DRUG_DISEASE', 'RNA_STRUCT', 'DRUG_SEQ', 'RNA_DISEASE', 'DRUG_STRUCT', 'RNA_SEQ'],
            19: ['RNA_STRUCT', 'DRUG_SEQ', 'RNA_DISEASE', 'DRUG_STRUCT', 'RNA_SEQ', 'DRUG_DISEASE'],
        }
        
        if self.TOKEN_ORDER_ID == -1:
            # 随机顺序
            import random
            order = token_orders[0].copy()
            random.shuffle(order)
            return order
        else:
            return token_orders.get(self.TOKEN_ORDER_ID, token_orders[0])
    
    
    def get_total_input_dim(self):
        """计算总输入维度（用于baseline）"""
        total_dim = 0
        if self.USE_RNA_SEQ:
            total_dim += self.RNA_SEQ_DIM
        if self.USE_RNA_STRUCT:
            total_dim += self.RNA_STRUCT_DIM
        if self.USE_RNA_DISEASE:
            total_dim += self.RNA_DISEASE_DIM
        if self.USE_DRUG_SEQ:
            total_dim += self.DRUG_SEQ_DIM
        if self.USE_DRUG_STRUCT:
            total_dim += self.DRUG_STRUCT_DIM
        if self.USE_DRUG_DISEASE:
            total_dim += self.DRUG_DISEASE_DIM
        return total_dim
    
    def __repr__(self):
        """打印配置信息"""
        config_str = "\n" + "="*60 + "\n"
        config_str += "🔧 配置信息\n"
        config_str += "="*60 + "\n"
        
        config_str += f"\n📊 数据集: {self.DATASET_NAME}\n"  # 🆕
        config_str += f"\n📊 模型类型: {self.MODEL_TYPE.upper()}\n"
        
        config_str += f"\n🎯 启用的模态: {', '.join(self.get_enabled_modalities())}\n"
        
        if self.MODEL_TYPE == 'llm':
            config_str += f"\n🤖 LLM配置:\n"
            config_str += f"   - 模型: LLaMA-3.1-8B\n"
            config_str += f"   - LoRA: {'启用' if self.USE_LORA else '禁用'}\n"
            if self.USE_LORA:
                config_str += f"   - LoRA Rank: {self.LORA_R}\n"
                config_str += f"   - LoRA模块: {len(self.LORA_TARGET_MODULES)}个\n"
            config_str += f"   - 分类头方案: {self.POOLING_METHOD.upper()}\n"
            config_str += f"   - Token顺序ID: {self.TOKEN_ORDER_ID}\n"
            config_str += f"   - Token顺序: {' → '.join(self.get_token_order())}\n"
        
        config_str += f"\n⚙️  训练配置:\n"
        config_str += f"   - Batch Size: {self.BATCH_SIZE}\n"
        config_str += f"   - 梯度累积: {self.ACCUMULATION_STEPS}步\n"
        config_str += f"   - 有效Batch Size: {self.BATCH_SIZE * self.ACCUMULATION_STEPS}\n"
        config_str += f"   - 学习率: {self.LEARNING_RATE}\n"
        config_str += f"   - 混合精度: {'启用' if self.USE_MIXED_PRECISION else '禁用'}\n"
        config_str += f"   - 多进程加载: {self.NUM_WORKERS}个worker\n"
        
        config_str += f"\n💾 数据配置:\n"
        config_str += f"   - 交叉验证折数: {self.N_FOLDS}\n"
        config_str += f"   - 负样本比例: {self.NEGATIVE_RATIO}:1\n"
        config_str += f"   - 总输入维度: {self.get_total_input_dim()}D\n"
        
        config_str += "\n" + "="*60 + "\n"
        return config_str

# 创建全局配置实例
config = Config()