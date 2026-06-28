import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from config import config


# ========== 基础组件 ==========

class FeatureAdapter(nn.Module):
    """
    特征Adapter:将不同维度的特征映射到LLM embedding空间
    
    三种类型:
    1. 序列特征 (640D/768D → 4096D)
    2. 结构特征 (128D/1024D → 4096D)  
    3. 疾病特征 (1690D → 512D → 4096D,压缩稀疏性)
    """
    def __init__(self, input_dim, output_dim=4096, adapter_type='normal'):
        super().__init__()
        self.adapter_type = adapter_type
        
        if adapter_type == 'disease':
            # 疾病特征:先压缩稀疏性
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            # 普通特征:渐进式升维
            if input_dim <= 256:
                hidden_dim = 1024
            elif input_dim <= 768:
                hidden_dim = 2048
            else:
                hidden_dim = 2048
            
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.adapter(x)


class WeightedPooling(nn.Module):
    """可学习权重的池化(learnable_weight方法)"""
    def __init__(self, num_features):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_features) / num_features)
    
    def forward(self, features):
        """
        Args:
            features: list of [B, D], 长度为num_features
        Returns:
            pooled: [B, D]
        """
        weights = F.softmax(self.weights, dim=0)
        stacked = torch.stack(features, dim=1)  # [B, num_features, D]
        pooled = (stacked * weights.view(1, -1, 1)).sum(dim=1)  # [B, D]
        return pooled
    
    def get_normalized_weights(self):
        """获取归一化后的权重"""
        with torch.no_grad():
            return F.softmax(self.weights, dim=0).cpu().numpy()


class TwoLayerClassifier(nn.Module):
    """2层MLP分类头"""
    def __init__(self, input_dim, hidden_dim=1024, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 2:
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)


# ========== 主模型 ==========

class MultimodalLLM(nn.Module):
    """
    多模态LLM - 修复混合精度训练版本
    
    🔧 关键修复:
    1. 支持torch.bfloat16加载LLM (更稳定的混合精度)
    2. 修复GradScaler导入
    3. 自动根据GPU能力选择精度
    """
    
    def __init__(self):
        super().__init__()
        
        print("\n" + "="*60)
        print("🔧 初始化多模态LLM (混合精度优化版)")
        print("="*60)
        
        # 1. 加载LLM
        self._load_llm()
        
        # 2. 添加特殊tokens
        self._add_special_tokens()
        
        # 3. 创建Adapters(根据启用的模态)
        self._create_adapters()
        
        # 4. 创建分类头
        self._create_classifier()
        
        self._print_model_info()
        print("="*60 + "\n")
    
    def _load_llm(self):
        """
        加载LLM并配置LoRA
        
        🔧 关键修复:
        - 使用torch.bfloat16而不是torch.float32
        - bfloat16在Ampere GPU (A100/RTX 30xx+)上更稳定
        - 如果不支持bfloat16,自动降级到float16
        """
        print("   📥 加载LLM模型...")
        
        # possible_paths = [
        #     "./llama3.1/LLM-Research/Meta-Llama-3___1-8B-Instruct",
        #     "./llama3.1/LLM-Research/Meta-Llama-3.1-8B-Instruct",
            
        #     config.LLM_MODEL_ID,
        # ]
        possible_paths = [
            "./llama3.1/LLM-Research/Llama-3___2-3B-Instruct",
            "./llama3.1/LLM-Research/Meta-Llama-3.1-3B-Instruct",
            
            config.LLM_MODEL_ID,
        ]
        # possible_paths = [
        #     "./llama3.1/LLM-Research/Llama-3___2-1B-Instruct",
            
        #     config.LLM_MODEL_ID,
        # ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"   ✅ 找到本地模型: {path}")
                break
        
        if model_path is None:
            raise RuntimeError("❌ 未找到本地模型")
        
        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 🔧 根据混合精度配置选择加载精度
        if config.USE_MIXED_PRECISION:
            # 检查GPU是否支持bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                load_dtype = torch.bfloat16
                print(f"   ✅ 使用 bfloat16 加载LLM (更稳定的混合精度)")
            else:
                load_dtype = torch.float16
                print(f"   ✅ 使用 float16 加载LLM (标准混合精度)")
        else:
            load_dtype = torch.float32
            print(f"   ℹ️  使用 float32 加载LLM (全精度训练)")
        
        # 加载模型
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None,
            torch_dtype=load_dtype,  # 🔧 修复: 根据配置动态选择精度
            trust_remote_code=True,
            local_files_only=True
        ).to(config.DEVICE)
                
        # 配置LoRA
        if config.USE_LORA:
            print(f"   🔧 配置LoRA (r={config.LORA_R}, alpha={config.LORA_ALPHA})...")
            
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=config.LORA_TARGET_MODULES,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            
            trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llm.parameters())
            print(f"   ✅ LoRA已启用:")
            print(f"      可训练参数: {trainable:,} ({100*trainable/total:.2f}%)")
            print(f"      目标模块: {len(config.LORA_TARGET_MODULES)}个")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("   🔒 LLM参数已冻结")
        
        self.llm_dtype = self.llm.dtype
        print(f"   📊 LLM运行精度: {self.llm_dtype}")
    
    def _add_special_tokens(self):
        """添加特殊tokens"""
        special_tokens = {
            'additional_special_tokens': [
                '<RNA_SEQ>', '<RNA_STRUCT>', '<RNA_DISEASE>',
                '<DRUG_SEQ>', '<DRUG_STRUCT>', '<DRUG_DISEASE>',
                '<CLS>'
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # 保存token IDs
        self.rna_seq_token_id = self.tokenizer.convert_tokens_to_ids('<RNA_SEQ>')
        self.rna_struct_token_id = self.tokenizer.convert_tokens_to_ids('<RNA_STRUCT>')
        self.rna_disease_token_id = self.tokenizer.convert_tokens_to_ids('<RNA_DISEASE>')
        self.drug_seq_token_id = self.tokenizer.convert_tokens_to_ids('<DRUG_SEQ>')
        self.drug_struct_token_id = self.tokenizer.convert_tokens_to_ids('<DRUG_STRUCT>')
        self.drug_disease_token_id = self.tokenizer.convert_tokens_to_ids('<DRUG_DISEASE>')
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids('<CLS>')
        
        print(f"   ✅ 添加了{num_added}个特殊token")
    
    def _create_adapters(self):
        """创建Adapters"""
        print(f"   🔧 创建Adapters (目标维度: {config.LLM_HIDDEN_DIM}D)...")
        
        # RNA Adapters
        if True:
            self.rna_seq_adapter = FeatureAdapter(config.RNA_SEQ_DIM, config.LLM_HIDDEN_DIM)
            print(f"      ✅ RNA序列Adapter: {config.RNA_SEQ_DIM}D → {config.LLM_HIDDEN_DIM}D")
        
        if True:
            self.rna_struct_adapter = FeatureAdapter(config.RNA_STRUCT_DIM, config.LLM_HIDDEN_DIM)
            print(f"      ✅ RNA结构Adapter: {config.RNA_STRUCT_DIM}D → {config.LLM_HIDDEN_DIM}D")
        
        if True:
            self.rna_disease_adapter = FeatureAdapter(
                config.RNA_DISEASE_DIM, config.LLM_HIDDEN_DIM, adapter_type='disease'
            )
            print(f"      ✅ RNA疾病Adapter: {config.RNA_DISEASE_DIM}D → {config.LLM_HIDDEN_DIM}D (压缩)")
        
        # Drug Adapters
        if True:
            self.drug_seq_adapter = FeatureAdapter(config.DRUG_SEQ_DIM, config.LLM_HIDDEN_DIM)
            print(f"      ✅ Drug序列Adapter: {config.DRUG_SEQ_DIM}D → {config.LLM_HIDDEN_DIM}D")
        
        if True:
            self.drug_struct_adapter = FeatureAdapter(config.DRUG_STRUCT_DIM, config.LLM_HIDDEN_DIM)
            print(f"      ✅ Drug结构Adapter: {config.DRUG_STRUCT_DIM}D → {config.LLM_HIDDEN_DIM}D")
        
        if True:
            self.drug_disease_adapter = FeatureAdapter(
                config.DRUG_DISEASE_DIM, config.LLM_HIDDEN_DIM, adapter_type='disease'
            )
            print(f"      ✅ Drug疾病Adapter: {config.DRUG_DISEASE_DIM}D → {config.LLM_HIDDEN_DIM}D (压缩)")
    
    def _create_classifier(self):
        """创建分类头"""
        print(f"   🔧 创建分类头: {config.POOLING_METHOD.upper()}...")
        
        # 创建Pooling层(如果需要)
        if config.POOLING_METHOD == 'learnable_weight':
            rna_modalities = sum([
                True,
                True,
                True
            ])
            drug_modalities = sum([
                True,
                True,
                True
            ])
            
            if rna_modalities > 1:
                self.rna_pooling = WeightedPooling(rna_modalities)
                print(f"      ✅ RNA加权池化: {rna_modalities}个模态")
            
            if drug_modalities > 1:
                self.drug_pooling = WeightedPooling(drug_modalities)
                print(f"      ✅ Drug加权池化: {drug_modalities}个模态")
            
            classifier_input_dim = config.LLM_HIDDEN_DIM * 2
        
        elif config.POOLING_METHOD == 'cls':
            classifier_input_dim = config.LLM_HIDDEN_DIM
        
        else:
            raise ValueError(f"❌ 不支持的分类头方案: {config.POOLING_METHOD}")
        
        # 创建分类器
        self.classifier = TwoLayerClassifier(
            input_dim=classifier_input_dim,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
            dropout=config.CLASSIFIER_DROPOUT
        )
        print(f"      ✅ 分类器: {classifier_input_dim}D → {config.CLASSIFIER_HIDDEN_DIM}D → 2")
    
    def _process_through_llm(self, embeds_dict, batch_size, device):
        """
        通过LLM处理embeddings
        
        Args:
            embeds_dict: {token_id: [B, D]}
            batch_size: int
            device: torch.device
        
        Returns:
            processed: {token_id: [B, D]}
        """
        # 🆕 根据配置的顺序排列token_ids
        token_order_names = config.get_token_order()
        
        # 将token名称映射到token_id
        name_to_id = {
            'RNA_SEQ': self.rna_seq_token_id,
            'RNA_STRUCT': self.rna_struct_token_id,
            'RNA_DISEASE': self.rna_disease_token_id,
            'DRUG_SEQ': self.drug_seq_token_id,
            'DRUG_STRUCT': self.drug_struct_token_id,
            'DRUG_DISEASE': self.drug_disease_token_id,
        }
        
        # 按配置的顺序构建token_ids列表（只包含存在的token）
        token_ids = []
        for name in token_order_names:
            token_id = name_to_id.get(name)
            if token_id is not None and token_id in embeds_dict:
                token_ids.append(token_id)
        
        # 如果使用CLS token，添加到最前面
        if config.POOLING_METHOD == 'cls' and self.cls_token_id in embeds_dict:
            token_ids.insert(0, self.cls_token_id)
        
        seq_len = len(token_ids)
        
        input_ids = torch.tensor([token_ids] * batch_size, dtype=torch.long, device=device)
        
        # 获取embeddings
        embeds = self.llm.get_input_embeddings()
        
        # 替换特殊token的embeddings
        inputs_embeds = embeds(input_ids)
        for idx, token_id in enumerate(token_ids):
            inputs_embeds[:, idx, :] = embeds_dict[token_id]
        
        # 通过LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 提取最后一层的hidden states
        last_hidden = outputs.hidden_states[-1]
        
        # 构建输出字典
        processed = {}
        for idx, token_id in enumerate(token_ids):
            processed[token_id] = last_hidden[:, idx, :]
        
        return processed
    
    def get_modality_features(self, rna_seq_features=None, drug_seq_features=None,
                              rna_struct_features=None, drug_struct_features=None,
                              rna_disease_features=None, drug_disease_features=None):
        """
        获取各模态的LLM处理后特征
        
        Returns:
            dict: {
                'rna_seq': [B, 4096] or None,
                'drug_seq': [B, 4096] or None,
                ...
            }
        """
        # 获取batch_size和device
        batch_size, device = None, None
        for feat in [rna_seq_features, drug_seq_features, rna_struct_features,
                     drug_struct_features, rna_disease_features, drug_disease_features]:
            if feat is not None:
                batch_size = feat.shape[0]
                device = feat.device
                break
        
        # Step 1: 准备embeddings
        embeds_dict = {}
        
        if True and rna_seq_features is not None:
            embeds_dict[self.rna_seq_token_id] = self.rna_seq_adapter(rna_seq_features)
        if True and rna_struct_features is not None:
            embeds_dict[self.rna_struct_token_id] = self.rna_struct_adapter(rna_struct_features)
        if True and rna_disease_features is not None:
            embeds_dict[self.rna_disease_token_id] = self.rna_disease_adapter(rna_disease_features)
        
        if True and drug_seq_features is not None:
            embeds_dict[self.drug_seq_token_id] = self.drug_seq_adapter(drug_seq_features)
        if True and drug_struct_features is not None:
            embeds_dict[self.drug_struct_token_id] = self.drug_struct_adapter(drug_struct_features)
        if True and drug_disease_features is not None:
            embeds_dict[self.drug_disease_token_id] = self.drug_disease_adapter(drug_disease_features)
        
        # Step 2: 通过LLM处理
        processed = self._process_through_llm(embeds_dict, batch_size, device)
        
        # Step 3: 返回各模态特征
        return {
            'rna_seq': processed.get(self.rna_seq_token_id),
            'rna_struct': processed.get(self.rna_struct_token_id),
            'rna_disease': processed.get(self.rna_disease_token_id),
            'drug_seq': processed.get(self.drug_seq_token_id),
            'drug_struct': processed.get(self.drug_struct_token_id),
            'drug_disease': processed.get(self.drug_disease_token_id)
        }
    
    def get_fused_features(self, rna_seq_features=None, drug_seq_features=None,
                          rna_struct_features=None, drug_struct_features=None,
                          rna_disease_features=None, drug_disease_features=None):
        """
        获取融合后的RNA和Drug特征
        
        Returns:
            rna_fused: [B, 4096]
            drug_fused: [B, 4096]
        """
        # 获取各模态特征
        modality_feats = self.get_modality_features(
            rna_seq_features, drug_seq_features,
            rna_struct_features, drug_struct_features,
            rna_disease_features, drug_disease_features
        )
        
        # RNA特征池化
        rna_features = []
        if modality_feats['rna_seq'] is not None:
            rna_features.append(modality_feats['rna_seq'])
        if modality_feats['rna_struct'] is not None:
            rna_features.append(modality_feats['rna_struct'])
        if modality_feats['rna_disease'] is not None:
            rna_features.append(modality_feats['rna_disease'])
        
        if len(rna_features) == 1:
            rna_fused = rna_features[0]
        else:
            rna_fused = self.rna_pooling(rna_features)
        
        # Drug特征池化
        drug_features = []
        if modality_feats['drug_seq'] is not None:
            drug_features.append(modality_feats['drug_seq'])
        if modality_feats['drug_struct'] is not None:
            drug_features.append(modality_feats['drug_struct'])
        if modality_feats['drug_disease'] is not None:
            drug_features.append(modality_feats['drug_disease'])
        
        if len(drug_features) == 1:
            drug_fused = drug_features[0]
        else:
            drug_fused = self.drug_pooling(drug_features)
        
        return rna_fused, drug_fused
    
    def forward(self, rna_seq_features=None, drug_seq_features=None,
                rna_struct_features=None, drug_struct_features=None,
                rna_disease_features=None, drug_disease_features=None):
        """
        前向传播
        
        Returns:
            logits: [B, 2]
        """
        # 获取batch_size和device
        batch_size, device = None, None
        for feat in [rna_seq_features, drug_seq_features, rna_struct_features,
                     drug_struct_features, rna_disease_features, drug_disease_features]:
            if feat is not None:
                batch_size = feat.shape[0]
                device = feat.device
                break
        
        # Step 1: 准备embeddings
        embeds_dict = {}
        
        if True and rna_seq_features is not None:
            embeds_dict[self.rna_seq_token_id] = self.rna_seq_adapter(rna_seq_features)
        if True and rna_struct_features is not None:
            embeds_dict[self.rna_struct_token_id] = self.rna_struct_adapter(rna_struct_features)
        if True and rna_disease_features is not None:
            embeds_dict[self.rna_disease_token_id] = self.rna_disease_adapter(rna_disease_features)
        
        if True and drug_seq_features is not None:
            embeds_dict[self.drug_seq_token_id] = self.drug_seq_adapter(drug_seq_features)
        if True and drug_struct_features is not None:
            embeds_dict[self.drug_struct_token_id] = self.drug_struct_adapter(drug_struct_features)
        if True and drug_disease_features is not None:
            embeds_dict[self.drug_disease_token_id] = self.drug_disease_adapter(drug_disease_features)
        
        # Step 2: 添加CLS Token的占位embedding
        if config.POOLING_METHOD == 'cls':
            embeds_dict[self.cls_token_id] = torch.zeros(
                batch_size, config.LLM_HIDDEN_DIM, device=device
            )
        
        # Step 3: 通过LLM处理
        processed = self._process_through_llm(embeds_dict, batch_size, device)
        
        # Step 4: 根据分类头方案提取特征
        if config.POOLING_METHOD == 'cls':
            cls_feat = processed[self.cls_token_id]
            logits = self.classifier(cls_feat)
        
        else:  # 'learnable_weight' or 'attention'
            rna_features = []
            if True and self.rna_seq_token_id in processed:
                rna_features.append(processed[self.rna_seq_token_id])
            if True and self.rna_struct_token_id in processed:
                rna_features.append(processed[self.rna_struct_token_id])
            if True and self.rna_disease_token_id in processed:
                rna_features.append(processed[self.rna_disease_token_id])
            
            drug_features = []
            if True and self.drug_seq_token_id in processed:
                drug_features.append(processed[self.drug_seq_token_id])
            if True and self.drug_struct_token_id in processed:
                drug_features.append(processed[self.drug_struct_token_id])
            if True and self.drug_disease_token_id in processed:
                drug_features.append(processed[self.drug_disease_token_id])
            
            # 池化
            if len(rna_features) == 1:
                rna_pooled = rna_features[0]
            else:
                rna_pooled = self.rna_pooling(rna_features)
            
            if len(drug_features) == 1:
                drug_pooled = drug_features[0]
            else:
                drug_pooled = self.drug_pooling(drug_features)
            
            # 拼接并分类
            combined = torch.cat([rna_pooled, drug_pooled], dim=-1)
            logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(self, **kwargs):
        """预测概率"""
        logits = self.forward(**kwargs)
        return torch.softmax(logits, dim=1)
    
    def predict(self, **kwargs):
        """预测类别"""
        probs = self.predict_proba(**kwargs)
        return torch.argmax(probs, dim=1)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n   📊 模型参数统计:")
        print(f"      总参数量: {total_params / 1e6:.1f}M")
        print(f"      可训练参数量: {trainable_params / 1e6:.1f}M")
        print(f"      冻结参数量: {(total_params - trainable_params) / 1e6:.1f}M")
        print(f"      可训练比例: {100*trainable_params/total_params:.2f}%")
        
        if config.USE_LORA:
            llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            print(f"\n      LLM可训练参数: {llm_trainable / 1e6:.2f}M")
        
        adapter_params = 0
        for name, module in self.named_modules():
            if 'adapter' in name.lower():
                adapter_params += sum(p.numel() for p in module.parameters())
        print(f"      Adapter参数: {adapter_params / 1e6:.2f}M")
        
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"      分类头参数: {classifier_params / 1e6:.2f}M")
        
    def get_modality_weights(self):
        """
        获取所有模态的权重
        
        Returns:
            dict: {'rna_weights': [...], 'drug_weights': [...]}
        """
        weights_dict = {}
        
        if hasattr(self, 'rna_pooling') and isinstance(self.rna_pooling, WeightedPooling):
            weights_dict['rna_weights'] = self.rna_pooling.get_normalized_weights().tolist()
        
        if hasattr(self, 'drug_pooling') and isinstance(self.drug_pooling, WeightedPooling):
            weights_dict['drug_weights'] = self.drug_pooling.get_normalized_weights().tolist()
        
        return weights_dict


# ========== 创建模型的工厂函数 ==========
def create_model():
    """根据config.MODEL_TYPE创建模型"""
    print("\n" + "="*60)
    print(f"🔧 创建模型: {config.MODEL_TYPE.upper()}")
    print("="*60)
    
    if config.MODEL_TYPE == 'llm':
        model = MultimodalLLM()
    elif config.MODEL_TYPE == 'baseline':
        from baseline import BaselineMLP
        version = getattr(config, 'BASELINE_VERSION', 'strong')
        model = BaselineMLP(version=version)
    else:
        raise ValueError(f"❌ 未知的模型类型: {config.MODEL_TYPE}")
    
    model = model.to(config.DEVICE)
    
    print(f"✅ 模型已移动到设备: {config.DEVICE}\n")
    
    return model