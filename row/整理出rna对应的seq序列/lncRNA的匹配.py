import pandas as pd
import re


def read_fasta_sequences(fasta_file):
    """
    读取FASTA文件，返回{ENSG_ID: sequence}的字典

    参数:
    fasta_file: FASTA文件路径

    返回:
    sequences_dict: {ENSG_ID: sequence}字典
    """
    sequences = {}
    current_id = None
    current_sequence = ""

    print(f"正在读取FASTA文件: {fasta_file}")

    with open(fasta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line.startswith('>'):
                # 保存前一个序列
                if current_id and current_sequence:
                    sequences[current_id] = current_sequence

                # 提取ENSG ID
                current_id = line[1:]  # 去掉'>'符号
                current_sequence = ""

            else:
                # 累积序列
                current_sequence += line

        # 保存最后一个序列
        if current_id and current_sequence:
            sequences[current_id] = current_sequence

    print(f"共读取到 {len(sequences)} 个序列")
    return sequences


def extract_sequences_to_excel(excel_file, fasta_file, output_file):
    """
    根据Excel中的ENSG ID列表，从FASTA文件中提取对应序列，生成新的Excel文件

    参数:
    excel_file: 包含ENSEMBL_ID的Excel文件路径
    fasta_file: 格式化后的FASTA文件路径
    output_file: 输出Excel文件路径
    """

    # 读取Excel文件中的ENSG ID列表
    print(f"正在读取Excel文件: {excel_file}")
    df = pd.read_excel(excel_file)

    # 获取ENSEMBL_ID列
    if 'ENSEMBL_ID' not in df.columns:
        print("错误: Excel文件中未找到'ENSEMBL_ID'列")
        print(f"可用列: {list(df.columns)}")
        return

    target_ids = df['ENSEMBL_ID'].tolist()
    print(f"需要查找 {len(target_ids)} 个ENSG ID")

    # 读取FASTA文件中的序列
    sequences_dict = read_fasta_sequences(fasta_file)

    # 提取对应的序列
    results = []
    found_count = 0
    missing_ids = []

    for ensg_id in target_ids:
        if ensg_id in sequences_dict:
            results.append({
                'ID': ensg_id,
                'Sequence': sequences_dict[ensg_id]
            })
            found_count += 1
        else:
            results.append({
                'ID': ensg_id,
                'Sequence': 'NOT_FOUND'
            })
            missing_ids.append(ensg_id)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 保存到Excel文件
    result_df.to_excel(output_file, index=False)

    # 打印统计信息
    print(f"\n=== 处理结果 ===")
    print(f"总共查找: {len(target_ids)} 个ID")
    print(f"成功找到: {found_count} 个序列")
    print(f"未找到: {len(missing_ids)} 个ID")

    if missing_ids:
        print(f"\n未找到的ID:")
        for missing_id in missing_ids[:10]:  # 只显示前10个
            print(f"  {missing_id}")
        if len(missing_ids) > 10:
            print(f"  ... 还有 {len(missing_ids) - 10} 个")

    print(f"\n结果已保存到: {output_file}")

    return result_df


def analyze_sequences(result_df):
    """分析提取的序列"""
    print(f"\n=== 序列分析 ===")

    # 过滤掉未找到的序列
    valid_sequences = result_df[result_df['Sequence'] != 'NOT_FOUND']

    if len(valid_sequences) > 0:
        # 序列长度统计
        sequence_lengths = valid_sequences['Sequence'].apply(len)

        print(f"有效序列数量: {len(valid_sequences)}")
        print(f"序列长度统计:")
        print(f"  最长: {sequence_lengths.max()} bp")
        print(f"  最短: {sequence_lengths.min()} bp")
        print(f"  平均: {sequence_lengths.mean():.1f} bp")
        print(f"  中位数: {sequence_lengths.median():.1f} bp")

        # 显示前几个序列的长度
        print(f"\n前5个序列的长度:")
        for i, (idx, row) in enumerate(valid_sequences.head().iterrows()):
            seq_len = len(row['Sequence'])
            print(f"  {row['ID']}: {seq_len} bp")
    else:
        print("没有找到有效的序列")


# 主函数
def main():
    """主处理函数"""
    # 文件路径设置
    excel_file = r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\ALLlncRNA-seq.xlsx"  # 输入的Excel文件
    fasta_file = "merged_sequences_by_ENSG.fa"  # 之前生成的格式化FASTA文件
    output_file = "sequences_with_ids.xlsx"  # 输出的Excel文件

    try:
        # 执行序列提取
        result_df = extract_sequences_to_excel(excel_file, fasta_file, output_file)

        # 分析结果
        analyze_sequences(result_df)

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确认以下文件存在:")
        print(f"  1. Excel文件: {excel_file}")
        print(f"  2. FASTA文件: {fasta_file}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 辅助函数：检查文件内容
def preview_files(excel_file, fasta_file):
    """预览文件内容，用于调试"""
    print("=== 文件预览 ===")

    # 预览Excel文件
    try:
        df = pd.read_excel(excel_file)
        print(f"\nExcel文件预览 ({excel_file}):")
        print(f"  行数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        print(f"  前5个ID:")
        for i, id_val in enumerate(df['ENSEMBL_ID'].head()):
            print(f"    {i + 1}. {id_val}")
    except Exception as e:
        print(f"无法读取Excel文件: {e}")

    # 预览FASTA文件
    try:
        print(f"\nFASTA文件预览 ({fasta_file}):")
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
            print(f"  总行数: {len(lines)}")

            # 统计序列数量
            seq_count = sum(1 for line in lines if line.startswith('>'))
            print(f"  序列数量: {seq_count}")

            # 显示前几个序列标题
            print("  前5个序列ID:")
            count = 0
            for line in lines:
                if line.startswith('>') and count < 5:
                    print(f"    {count + 1}. {line.strip()}")
                    count += 1

    except Exception as e:
        print(f"无法读取FASTA文件: {e}")


# 使用示例
if __name__ == "__main__":
    # 可以先预览文件内容
    # preview_files("input_ensembl_ids.xlsx", "merged_sequences_by_ENSG.fa")

    # 执行主处理
    main()


# 如果你的Excel文件列名不是'ENSEMBL_ID'，可以使用这个函数
def extract_sequences_custom_column(excel_file, fasta_file, output_file, id_column_name):
    """自定义列名版本"""
    df = pd.read_excel(excel_file)
    target_ids = df[id_column_name].tolist()

    # 其余处理逻辑相同...
    sequences_dict = read_fasta_sequences(fasta_file)

    results = []
    for ensg_id in target_ids:
        if ensg_id in sequences_dict:
            results.append({
                'ID': ensg_id,
                'Sequence': sequences_dict[ensg_id]
            })
        else:
            results.append({
                'ID': ensg_id,
                'Sequence': 'NOT_FOUND'
            })

    result_df = pd.DataFrame(results)
    result_df.to_excel(output_file, index=False)

    return result_df