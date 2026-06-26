import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os


def extract_chemberta_features(excel_file, output_file="drug_prefeatures.xlsx"):
    """
    使用ChemBERTa模型提取分子SMILES特征

    Args:
        excel_file: 输入的Excel文件路径
        output_file: 输出的特征文件路径

    Returns:
        bool: 是否成功处理
    """
    print("开始使用ChemBERTa提取分子特征...")

    # 1. 检查输入文件
    if not os.path.exists(excel_file):
        print(f"✗ 输入文件不存在: {excel_file}")
        return False

    # 2. 读取Excel文件
    print("正在读取Excel文件...")
    try:
        df = pd.read_excel(excel_file)
        print(f"✓ 成功读取Excel文件，共{len(df)}条记录")
    except Exception as e:
        print(f"✗ 读取Excel文件失败: {e}")
        return False

    # 3. 检查必要的列
    if 'CID' not in df.columns or 'SMILES' not in df.columns:
        print("✗ Excel文件缺少必要的列：CID 或 SMILES")
        print(f"当前列名: {list(df.columns)}")
        return False

    # 4. 提取数据
    cid_list = df['CID'].tolist()
    smiles_list = df['SMILES'].tolist()
    print(f"✓ 提取到{len(cid_list)}个CID和{len(smiles_list)}个SMILES")

    # 5. 加载ChemBERTa模型
    print("正在加载ChemBERTa模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", from_flax=True)
        model.eval()  # 设置为评估模式
        print("✓ 成功加载ChemBERTa模型")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        print("请确保网络连接正常，或模型已下载")
        return False

    # 6. 批量处理SMILES获取特征
    print("开始提取特征向量...")
    features = []
    failed_indices = []

    for i, smiles in enumerate(tqdm(smiles_list, desc="提取SMILES特征")):
        try:
            # 检查SMILES是否有效
            if pd.isna(smiles) or smiles == "":
                print(f"⚠️ CID {cid_list[i]}: SMILES为空")
                features.append([0.0] * 768)  # 使用零向量填充
                failed_indices.append(i)
                continue

            # Tokenize SMILES
            inputs = tokenizer(str(smiles), return_tensors="pt", truncation=True, max_length=512)

            # 获取模型输出
            with torch.no_grad():
                outputs = model(**inputs)

            # 提取特征向量 (取[CLS] token的hidden state)
            feature_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # shape: (768,)
            features.append(feature_vector.tolist())

        except Exception as e:
            print(f"⚠️ CID {cid_list[i]} (SMILES: {smiles}): 处理失败 - {e}")
            features.append([0.0] * 768)  # 使用零向量填充
            failed_indices.append(i)

    # 7. 构建特征矩阵DataFrame
    print("正在构建特征矩阵...")

    # 创建列名：CID + feature0 到 feature767
    feature_columns = [f"feature{i}" for i in range(768)]
    all_columns = ["CID"] + feature_columns

    # 构建数据矩阵
    feature_matrix = []
    for i, feature_vector in enumerate(features):
        row = [cid_list[i]] + feature_vector
        feature_matrix.append(row)

    # 创建DataFrame
    feature_df = pd.DataFrame(feature_matrix, columns=all_columns)

    # 8. 保存为Excel文件
    print(f"正在保存特征矩阵到: {output_file}")
    try:
        feature_df.to_excel(output_file, index=False)
        print(f"✓ 特征矩阵已成功保存为: {output_file}")
    except Exception as e:
        print(f"✗ 保存Excel文件失败: {e}")
        # 尝试保存为CSV作为备选
        csv_file = output_file.replace('.xlsx', '.csv')
        try:
            feature_df.to_csv(csv_file, index=False)
            print(f"✓ 已保存为CSV格式: {csv_file}")
        except Exception as e2:
            print(f"✗ CSV保存也失败: {e2}")
            return False

    # 9. 打印处理结果统计
    print("\n" + "=" * 60)
    print("ChemBERTa特征提取结果统计")
    print("=" * 60)
    print(f"总分子数: {len(cid_list)}")
    print(f"成功提取: {len(cid_list) - len(failed_indices)}")
    print(f"提取失败: {len(failed_indices)}")
    print(f"成功率: {(len(cid_list) - len(failed_indices)) / len(cid_list) * 100:.1f}%")
    print(f"特征维度: 768维")
    print(f"输出文件: {output_file}")

    if failed_indices:
        print(f"\n失败的CID (前10个):")
        for idx in failed_indices[:10]:
            print(f"  CID {cid_list[idx]}: {smiles_list[idx]}")
        if len(failed_indices) > 10:
            print(f"  ... 还有 {len(failed_indices) - 10} 个失败")

    return True


def verify_feature_file(feature_file):
    """
    验证生成的特征文件
    """
    print(f"\n验证特征文件: {feature_file}")

    if not os.path.exists(feature_file):
        print("✗ 特征文件不存在")
        return

    try:
        df = pd.read_excel(feature_file)
        print(f"✓ 文件读取成功")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print(f"  CID列: {'CID' in df.columns}")

        # 检查特征列
        feature_cols = [col for col in df.columns if col.startswith('feature')]
        print(f"  特征列数: {len(feature_cols)}")

        # 显示前几行
        print(f"\n前3行数据:")
        print(df[['CID'] + feature_cols[:5]].head(3))

        # 检查是否有空值
        null_count = df.isnull().sum().sum()
        print(f"\n空值总数: {null_count}")

        # 检查特征值范围
        feature_data = df[feature_cols].values
        print(f"特征值范围: [{feature_data.min():.4f}, {feature_data.max():.4f}]")
        print(f"特征值平均: {feature_data.mean():.4f}")

    except Exception as e:
        print(f"✗ 文件验证失败: {e}")


def create_sample_visualization(feature_file, n_samples=5):
    """
    创建特征的简单可视化样本
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_excel(feature_file)
        feature_cols = [col for col in df.columns if col.startswith('feature')]

        # 选择前n_samples个分子的前100个特征进行可视化
        sample_data = df[feature_cols[:100]].head(n_samples)
        sample_cids = df['CID'].head(n_samples).tolist()

        plt.figure(figsize=(12, 6))
        sns.heatmap(sample_data,
                    yticklabels=[f"CID_{cid}" for cid in sample_cids],
                    cmap='viridis',
                    cbar=True)
        plt.title(f"ChemBERTa特征热力图 (前{n_samples}个分子, 前100维特征)")
        plt.xlabel("特征维度")
        plt.ylabel("分子")
        plt.tight_layout()
        plt.savefig("chemberta_features_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 特征可视化已保存: chemberta_features_heatmap.png")

    except ImportError:
        print("⚠️ 需要安装matplotlib和seaborn来生成可视化")
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    excel_file = r"D:\Desktop\CDLLM\ing\row\SDF\ALLdrug-smiles.xlsx"

    # 输出文件路径
    output_file = "drug_prefeatures.xlsx"

    # 提取特征
    success = extract_chemberta_features(excel_file, output_file)

    if success:
        # 验证生成的文件
        verify_feature_file(output_file)

        # 创建样本可视化
        create_sample_visualization(output_file, n_samples=5)

        print(f"\n🎉 ChemBERTa特征提取完成！")
        print(f"特征文件: {output_file}")
        print(f"格式: CID + feature0~feature767 (共769列)")
        print(f"现在可以用于后续的机器学习任务了！")
    else:
        print(f"\n❌ 特征提取失败，请检查错误信息")