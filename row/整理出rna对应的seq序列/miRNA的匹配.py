import pandas as pd
from collections import defaultdict


def read_fasta_file(fasta_file_path):
    """
    读取FASTA文件，返回ID到序列的映射字典

    参数:
    fasta_file_path: FASTA文件路径

    返回:
    字典，键为miRNA ID，值为序列列表（因为可能有多个序列对应同一个ID）
    """
    sequences = defaultdict(list)
    current_id = None

    with open(fasta_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith('>'):
                # 提取ID，去掉>符号
                current_id = line[1:]  # 去掉开头的>
            elif line and current_id:
                # 添加序列到对应ID
                sequences[current_id].append(line)

    return dict(sequences)


def match_sequences(excel_file_path, fasta_file_path, output_file_path=None):
    """
    根据Excel文件中的miRNA ID匹配FASTA文件中的序列

    参数:
    excel_file_path: Excel文件路径
    fasta_file_path: FASTA文件路径
    output_file_path: 输出文件路径（可选）

    返回:
    包含匹配结果的DataFrame
    """

    # 读取Excel文件
    df = pd.read_excel(excel_file_path)

    # 假设第一列包含miRNA ID
    mirna_ids = df.iloc[:, 0].tolist()  # 获取第一列所有值

    # 读取FASTA文件
    fasta_sequences = read_fasta_file(fasta_file_path)

    # 准备结果数据
    results = []

    for mirna_id in mirna_ids:
        if pd.isna(mirna_id):  # 跳过空值
            continue

        mirna_id = str(mirna_id).strip()  # 转换为字符串并去除空格

        # 查找匹配的序列
        matched_sequences = []

        # 直接匹配
        if mirna_id in fasta_sequences:
            matched_sequences.extend(fasta_sequences[mirna_id])

        # 模糊匹配：查找包含该ID的序列
        for fasta_id, seqs in fasta_sequences.items():
            if mirna_id in fasta_id or fasta_id in mirna_id:
                if mirna_id not in fasta_sequences:  # 避免重复添加
                    matched_sequences.extend(seqs)

        # 添加结果
        if matched_sequences:
            for i, seq in enumerate(matched_sequences):
                results.append({
                    'miRBase_ID': mirna_id,
                    'Sequence_Index': i + 1,
                    'Sequence': seq,
                    'Match_Type': 'Direct' if mirna_id in fasta_sequences else 'Fuzzy'
                })
        else:
            results.append({
                'miRBase_ID': mirna_id,
                'Sequence_Index': 0,
                'Sequence': 'NOT_FOUND',
                'Match_Type': 'No_Match'
            })

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 输出到文件（如果指定了输出路径）
    if output_file_path:
        result_df.to_excel(output_file_path, index=False)
        print(f"结果已保存到: {output_file_path}")

    return result_df


def advanced_match_sequences(excel_file_path, fasta_file_path, output_file_path=None):
    """
    高级匹配功能，支持更复杂的ID匹配规则

    参数同上
    """

    # 读取Excel文件
    df = pd.read_excel(excel_file_path)
    mirna_ids = df.iloc[:, 0].tolist()

    # 读取FASTA文件
    fasta_sequences = read_fasta_file(fasta_file_path)

    results = []

    for mirna_id in mirna_ids:
        if pd.isna(mirna_id):
            continue

        mirna_id = str(mirna_id).strip()
        matched_sequences = []
        match_details = []

        # 1. 精确匹配
        if mirna_id in fasta_sequences:
            matched_sequences.extend(fasta_sequences[mirna_id])
            match_details.extend(['Exact'] * len(fasta_sequences[mirna_id]))

        # 2. 前缀匹配：Excel中的ID是FASTA中ID的前缀
        for fasta_id, seqs in fasta_sequences.items():
            if fasta_id.startswith(mirna_id) and mirna_id not in fasta_sequences:
                matched_sequences.extend(seqs)
                match_details.extend([f'Prefix_of_{fasta_id}'] * len(seqs))

        # 3. 包含匹配：FASTA中的ID包含Excel中的ID
        for fasta_id, seqs in fasta_sequences.items():
            if (mirna_id in fasta_id and
                    not fasta_id.startswith(mirna_id) and
                    mirna_id not in fasta_sequences):
                matched_sequences.extend(seqs)
                match_details.extend([f'Contains_in_{fasta_id}'] * len(seqs))

        # 添加结果
        if matched_sequences:
            for i, (seq, detail) in enumerate(zip(matched_sequences, match_details)):
                results.append({
                    'miRBase_ID': mirna_id,
                    'Sequence_Index': i + 1,
                    'Sequence': seq,
                    'Match_Detail': detail,
                    'Sequence_Length': len(seq)
                })
        else:
            results.append({
                'miRBase_ID': mirna_id,
                'Sequence_Index': 0,
                'Sequence': 'NOT_FOUND',
                'Match_Detail': 'No_Match',
                'Sequence_Length': 0
            })

    result_df = pd.DataFrame(results)

    if output_file_path:
        result_df.to_excel(output_file_path, index=False)
        print(f"高级匹配结果已保存到: {output_file_path}")

    return result_df


def display_statistics(result_df):
    """
    显示匹配统计信息
    """
    total_ids = len(result_df['miRBase_ID'].unique())
    found_ids = len(result_df[result_df['Sequence'] != 'NOT_FOUND']['miRBase_ID'].unique())
    not_found_ids = total_ids - found_ids
    total_sequences = len(result_df[result_df['Sequence'] != 'NOT_FOUND'])

    print("\n=== 匹配统计 ===")
    print(f"总miRNA ID数量: {total_ids}")
    print(f"找到序列的ID数量: {found_ids}")
    print(f"未找到序列的ID数量: {not_found_ids}")
    print(f"匹配成功率: {found_ids / total_ids * 100:.1f}%")
    print(f"总匹配序列数量: {total_sequences}")

    if 'Match_Detail' in result_df.columns:
        print("\n=== 匹配类型统计 ===")
        match_counts = result_df['Match_Detail'].value_counts()
        for match_type, count in match_counts.items():
            print(f"{match_type}: {count}")


# 使用示例
if __name__ == "__main__":
    # 基础匹配
    print("执行基础匹配...")
    basic_result = match_sequences(r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\ALLmiRNA-seq.xlsx', r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\output.fa', 'basic_match_result.xlsx')

    print("\n基础匹配结果预览:")
    print(basic_result.head(10))
    display_statistics(basic_result)

    print("\n" + "=" * 50)

    # 高级匹配
    print("执行高级匹配...")
    advanced_result = advanced_match_sequences(r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\ALLmiRNA-seq.xlsx', r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\output.fa', 'advanced_match_result.xlsx')

    print("\n高级匹配结果预览:")
    print(advanced_result.head(10))
    display_statistics(advanced_result)

    # 显示未找到的ID
    not_found = advanced_result[advanced_result['Sequence'] == 'NOT_FOUND']['miRBase_ID'].unique()
    if len(not_found) > 0:
        print(f"\n未找到序列的ID ({len(not_found)}个):")
        for nf_id in not_found[:10]:  # 只显示前10个
            print(f"  {nf_id}")
        if len(not_found) > 10:
            print(f"  ... 还有{len(not_found) - 10}个")