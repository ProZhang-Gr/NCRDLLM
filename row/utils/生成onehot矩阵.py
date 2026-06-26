import pandas as pd
import numpy as np
from collections import defaultdict


def process_disease_associations():
    """
    处理四个疾病关联文件，生成ncRNA和Drug的one-hot编码矩阵
    """

    print("=== 步骤1: 读取疾病关联文件 ===")

    # 读取疾病关联文件
    circRNA_disease = pd.read_excel('circRNA_Disease.xlsx')
    miRNA_disease = pd.read_excel('miRNA_Disease.xlsx')
    lncRNA_disease = pd.read_excel('lncRNA_Disease.xlsx')
    drug_disease = pd.read_excel('Drug_Disease.xlsx')

    print(f"circRNA-Disease关联数: {len(circRNA_disease)}")
    print(f"miRNA-Disease关联数: {len(miRNA_disease)}")
    print(f"lncRNA-Disease关联数: {len(lncRNA_disease)}")
    print(f"Drug-Disease关联数: {len(drug_disease)}")

    print("\n=== 步骤2: 读取序列/结构文件获取完整ID列表 ===")

    # 读取所有ncRNA和Drug的完整列表
    all_circRNA = pd.read_excel('ALLcircRNA-seq.xlsx')
    all_miRNA = pd.read_excel('ALLmiRNA-seq.xlsx')
    all_lncRNA = pd.read_excel('ALLlncRNA-seq.xlsx')
    all_drugs = pd.read_excel('ALLdrug-smiles.xlsx')

    print(f"总circRNA数量: {len(all_circRNA)}")
    print(f"总miRNA数量: {len(all_miRNA)}")
    print(f"总lncRNA数量: {len(all_lncRNA)}")
    print(f"总Drug数量: {len(all_drugs)}")

    print("\n=== 步骤3: 提取所有Disease的并集并排序 ===")

    # 收集所有Disease ID
    all_diseases = set()
    all_diseases.update(circRNA_disease['DOID'].tolist())
    all_diseases.update(miRNA_disease['DOID'].tolist())
    all_diseases.update(lncRNA_disease['DOID'].tolist())
    all_diseases.update(drug_disease['DOID'].tolist())

    # 按字母顺序排序
    sorted_diseases = sorted(list(all_diseases))
    print(f"总Disease数量: {len(sorted_diseases)}")
    print(f"Disease示例: {sorted_diseases[:5]}...")

    print("\n=== 步骤4: 构建关联字典 ===")

    # 构建关联字典
    def build_association_dict(df, id_col):
        assoc_dict = defaultdict(set)
        for _, row in df.iterrows():
            assoc_dict[row[id_col]].add(row['DOID'])
        return assoc_dict

    circRNA_assoc = build_association_dict(circRNA_disease, 'circBase_ID')
    miRNA_assoc = build_association_dict(miRNA_disease, 'miRBase_ID')
    lncRNA_assoc = build_association_dict(lncRNA_disease, 'ENSEMBL_ID')
    drug_assoc = build_association_dict(drug_disease, 'CID')

    print("\n=== 步骤5: 生成one-hot编码矩阵 ===")

    def create_onehot_matrix(all_ids_df, id_col, assoc_dict, sorted_diseases, output_filename):
        """创建one-hot编码矩阵"""

        # 获取所有ID
        all_ids = all_ids_df[id_col].tolist()

        print(f"正在处理 {output_filename}...")
        print(f"  ID数量: {len(all_ids)}")
        print(f"  Disease数量: {len(sorted_diseases)}")

        # 初始化矩阵
        matrix = np.zeros((len(all_ids), len(sorted_diseases)), dtype=int)

        # 填充矩阵
        for i, item_id in enumerate(all_ids):
            if item_id in assoc_dict:
                associated_diseases = assoc_dict[item_id]
                for j, disease in enumerate(sorted_diseases):
                    if disease in associated_diseases:
                        matrix[i, j] = 1

        # 转换为DataFrame
        df_matrix = pd.DataFrame(
            matrix,
            index=all_ids,
            columns=sorted_diseases
        )

        # 保存到Excel文件
        df_matrix.to_excel(output_filename, index=True)

        # 统计信息
        total_ones = np.sum(matrix)
        total_elements = matrix.size
        sparsity = (total_elements - total_ones) / total_elements * 100

        print(f"  关联数量: {total_ones}")
        print(f"  稀疏度: {sparsity:.2f}%")

        return df_matrix

    # 生成四个one-hot矩阵文件
    circRNA_matrix = create_onehot_matrix(
        all_circRNA, 'circBase_ID', circRNA_assoc, sorted_diseases,
        'circRNA_onehot_matrix.xlsx'
    )

    miRNA_matrix = create_onehot_matrix(
        all_miRNA, 'miRBase_ID', miRNA_assoc, sorted_diseases,
        'miRNA_onehot_matrix.xlsx'
    )

    lncRNA_matrix = create_onehot_matrix(
        all_lncRNA, 'ENSEMBL_ID', lncRNA_assoc, sorted_diseases,
        'lncRNA_onehot_matrix.xlsx'
    )

    drug_matrix = create_onehot_matrix(
        all_drugs, 'CID', drug_assoc, sorted_diseases,
        'Drug_onehot_matrix.xlsx'
    )

    print("\n=== 处理完成! ===")
    print("生成的文件:")
    print("- circRNA_onehot_matrix.xlsx")
    print("- miRNA_onehot_matrix.xlsx")
    print("- lncRNA_onehot_matrix.xlsx")
    print("- Drug_onehot_matrix.xlsx")

    # 返回摘要信息
    summary = {
        'total_diseases': len(sorted_diseases),
        'circRNA_count': len(all_circRNA),
        'miRNA_count': len(all_miRNA),
        'lncRNA_count': len(all_lncRNA),
        'drug_count': len(all_drugs),
        'diseases': sorted_diseases
    }

    return summary


# 运行处理函数
if __name__ == "__main__":
    try:
        summary = process_disease_associations()
        print(f"\n处理摘要:")
        print(f"总Disease数量: {summary['total_diseases']}")
        print(f"circRNA数量: {summary['circRNA_count']}")
        print(f"miRNA数量: {summary['miRNA_count']}")
        print(f"lncRNA数量: {summary['lncRNA_count']}")
        print(f"Drug数量: {summary['drug_count']}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        print("请确保所有输入文件都存在且格式正确")