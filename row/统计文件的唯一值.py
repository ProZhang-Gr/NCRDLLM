import pandas as pd


def count_unique_identifiers(file_path):
    """
    统计xlsx文件中唯一的RNA标识符(miRBase_ID)和CID数量

    Args:
        file_path (str): xlsx文件路径

    Returns:
        dict: 包含统计结果的字典
    """
    # 读取xlsx文件
    df = pd.read_excel(file_path)

    # 统计唯一的RNA标识符数量
    unique_mirbase_ids = df['miRBase_ID'].nunique()

    # 统计唯一的CID数量
    unique_cids = df['CID'].nunique()

    # 获取唯一值列表（可选）
    unique_mirbase_list = df['miRBase_ID'].unique().tolist()
    unique_cid_list = df['CID'].unique().tolist()

    # 统计结果
    results = {
        'unique_mirbase_count': unique_mirbase_ids,
        'unique_cid_count': unique_cids,
        'total_rows': len(df),
        'unique_mirbase_ids': unique_mirbase_list,
        'unique_cids': unique_cid_list
    }

    return results


def print_statistics(results):
    """
    打印统计结果

    Args:
        results (dict): 统计结果字典
    """
    print("=" * 50)
    print("RNA标识符和CID统计结果")
    print("=" * 50)
    print(f"总行数: {results['total_rows']}")
    print(f"唯一RNA标识符(miRBase_ID)数量: {results['unique_mirbase_count']}")
    print(f"唯一CID数量: {results['unique_cid_count']}")
    print("\n" + "-" * 30)

    print(f"\n唯一的RNA标识符列表 ({len(results['unique_mirbase_ids'])}个):")
    for i, mirbase_id in enumerate(sorted(results['unique_mirbase_ids']), 1):
        print(f"{i:2d}. {mirbase_id}")

    print(f"\n唯一的CID列表 ({len(results['unique_cids'])}个):")
    for i, cid in enumerate(sorted(results['unique_cids']), 1):
        print(f"{i:2d}. {cid}")


# 如果你有原始数据，也可以直接从字典创建DataFrame进行统计
def count_from_sample_data():
    """
    使用你提供的样本数据进行统计
    """
    # 根据你提供的数据创建DataFrame
    sample_data = {
        'miRBase_ID': [
            'hsa-let-7a-1', 'hsa-let-7a-1', 'hsa-let-7a-2', 'hsa-let-7a-2',
            'hsa-let-7a-2-3p', 'hsa-let-7a-3', 'hsa-let-7a-3p', 'hsa-let-7a-3p',
            'hsa-let-7a-5p'
        ],
        'CID': [
            57379345, 44462760, 2733525, 44462760,
            6857599, 44462760, 60750, 148124, 31703
        ]
    }

    df = pd.DataFrame(sample_data)

    # 统计唯一值
    unique_mirbase_count = df['miRBase_ID'].nunique()
    unique_cid_count = df['CID'].nunique()

    print("样本数据统计结果:")
    print(f"唯一RNA标识符数量: {unique_mirbase_count}")
    print(f"唯一CID数量: {unique_cid_count}")
    print(f"总记录数: {len(df)}")

    return df


# 主函数 - 使用示例
if __name__ == "__main__":
    # 方法1: 从xlsx文件读取并统计
    file_path = r"D:\Desktop\CDLLM\项目进行时\row\miRNA-drug处理过程\RNAInter第一步收集.xlsx"  # 替换为你的文件路径
    results = count_unique_identifiers(file_path)
    print_statistics(results)

