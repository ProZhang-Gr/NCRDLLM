import pandas as pd
from collections import defaultdict
import os


def read_fasta_file(fasta_file):
    """
    读取FASTA文件，返回ID到序列的字典
    """
    sequences = {}
    current_id = None

    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 提取ID，去掉'>'符号
                current_id = line[1:]
                if current_id not in sequences:
                    sequences[current_id] = []
            elif line and current_id:
                # 将序列添加到对应ID的列表中
                sequences[current_id].append(line)

    # 将序列列表合并为单个字符串
    for id_name in sequences:
        sequences[id_name] = ''.join(sequences[id_name])

    return sequences


def read_xlsx_ids(xlsx_file, column_name='miRBase_ID'):
    """
    读取xlsx文件中指定列的ID
    """
    df = pd.read_excel(xlsx_file)
    return df[column_name].tolist()


def match_sequences(fasta_sequences, target_ids):
    """
    匹配序列，返回匹配结果和未匹配的ID
    """
    matched_results = []
    unmatched_ids = []

    for target_id in target_ids:
        if target_id in fasta_sequences:
            matched_results.append({
                'miRBase_ID': target_id,
                'Sequence': fasta_sequences[target_id],
                'Status': 'Matched'
            })
        else:
            matched_results.append({
                'miRBase_ID': target_id,
                'Sequence': '',
                'Status': 'Not Found'
            })
            unmatched_ids.append(target_id)

    return matched_results, unmatched_ids


def create_statistics_report(matched_results, unmatched_ids, fasta_sequences):
    """
    创建详细的统计报告
    """
    total_target_ids = len(matched_results)
    matched_count = sum(1 for result in matched_results if result['Status'] == 'Matched')
    unmatched_count = len(unmatched_ids)

    # 统计FASTA文件中的总序列数
    total_fasta_sequences = len(fasta_sequences)

    # 创建统计数据
    stats_data = {
        'Metric': [
            'Total Target IDs',
            'Successfully Matched',
            'Not Found',
            'Match Rate (%)',
            'Total Sequences in FASTA'
        ],
        'Count': [
            total_target_ids,
            matched_count,
            unmatched_count,
            round((matched_count / total_target_ids * 100), 2) if total_target_ids > 0 else 0,
            total_fasta_sequences
        ]
    }

    return stats_data


def main():
    # 文件路径设置
    fasta_file = r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\output.fa'  # 你的fa文件路径
    xlsx_file = r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\ALLmiRNA-seq.xlsx'  # 你的xlsx文件路径

    # 输出文件路径
    output_matched = 'matched_sequences.xlsx'
    output_unmatched = 'unmatched_ids.xlsx'
    output_statistics = 'matching_statistics.xlsx'

    try:
        # 读取FASTA文件
        print("读取FASTA文件...")
        fasta_sequences = read_fasta_file(fasta_file)
        print(f"FASTA文件中共找到 {len(fasta_sequences)} 个序列")

        # 读取xlsx文件中的目标ID
        print("读取xlsx文件...")
        target_ids = read_xlsx_ids(xlsx_file)
        print(f"目标ID列表中共有 {len(target_ids)} 个ID")

        # 执行匹配
        print("执行序列匹配...")
        matched_results, unmatched_ids = match_sequences(fasta_sequences, target_ids)

        # 创建匹配结果DataFrame
        matched_df = pd.DataFrame(matched_results)

        # 保存匹配结果
        matched_df.to_excel(output_matched, index=False)
        print(f"匹配结果已保存到: {output_matched}")

        # 创建未匹配ID的DataFrame
        if unmatched_ids:
            unmatched_df = pd.DataFrame({
                'Unmatched_miRBase_ID': unmatched_ids,
                'Reason': ['Not found in FASTA file'] * len(unmatched_ids)
            })
            unmatched_df.to_excel(output_unmatched, index=False)
            print(f"未匹配ID列表已保存到: {output_unmatched}")

        # 创建统计报告
        stats_data = create_statistics_report(matched_results, unmatched_ids, fasta_sequences)
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(output_statistics, index=False)
        print(f"统计报告已保存到: {output_statistics}")

        # 打印摘要统计
        print("\n=== 匹配摘要 ===")
        print(f"目标ID总数: {len(target_ids)}")
        print(f"成功匹配: {len(target_ids) - len(unmatched_ids)}")
        print(f"未匹配: {len(unmatched_ids)}")
        print(f"匹配率: {((len(target_ids) - len(unmatched_ids)) / len(target_ids) * 100):.2f}%")

        if unmatched_ids:
            print(f"\n未匹配的ID (前10个):")
            for i, unmatched_id in enumerate(unmatched_ids[:10]):
                print(f"  {i + 1}. {unmatched_id}")
            if len(unmatched_ids) > 10:
                print(f"  ... 还有 {len(unmatched_ids) - 10} 个未匹配的ID")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确保以下文件存在:")
        print(f"  - FASTA文件: {fasta_file}")
        print(f"  - Excel文件: {xlsx_file}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main()