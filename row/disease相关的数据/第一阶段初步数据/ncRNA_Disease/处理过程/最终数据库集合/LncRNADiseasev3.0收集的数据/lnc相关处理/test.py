import pandas as pd
import re
import numpy as np
from tqdm import tqdm


def parse_gff3_attributes(attributes):
    """解析GFF3文件的attributes列"""
    attr_dict = {}
    for attr in attributes.split(';'):
        if '=' in attr:
            key, value = attr.split('=', 1)
            attr_dict[key] = value
    return attr_dict


def create_id_mapping(gff3_file):
    """从GFF3文件创建ID到ENSG的映射字典"""
    print("正在读取GFF3文件...")

    # 存储映射关系的字典
    gene_name_to_ensg = {}
    id_to_ensg = {}

    with open(gff3_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 跳过注释行
            if line.startswith('#'):
                continue

            line = line.strip()
            if not line:
                continue

            # 解析GFF3行
            parts = line.split('\t')
            if len(parts) != 9:
                continue

            attributes = parts[8]
            attr_dict = parse_gff3_attributes(attributes)

            # 提取需要的字段
            gene_id = attr_dict.get('gene_id', '')
            gene_name = attr_dict.get('gene_name', '')
            record_id = attr_dict.get('ID', '')

            # 确保gene_id是ENSG格式
            if gene_id and gene_id.startswith('ENSG'):
                # 添加gene_name映射
                if gene_name:
                    gene_name_to_ensg[gene_name] = gene_id

                # 添加ID映射 (包括ENSG, ENST, exon等所有ID)
                if record_id:
                    id_to_ensg[record_id] = gene_id

            # 每处理100000行显示进度
            if line_num % 100000 == 0:
                print(f"已处理 {line_num} 行...")

    print(f"映射创建完成！")
    print(f"gene_name映射: {len(gene_name_to_ensg)} 条")
    print(f"ID映射: {len(id_to_ensg)} 条")

    return gene_name_to_ensg, id_to_ensg


def map_ensembl_ids(xlsx_file, gff3_file, output_file=None):
    """将Excel文件中的ENSEMBL_ID列映射为ENSG标识符"""

    # 读取Excel文件
    print("正在读取Excel文件...")
    df = pd.read_excel(xlsx_file)

    # 检查是否有ENSEMBL_ID列
    if 'ENSEMBL_ID' not in df.columns:
        raise ValueError("Excel文件中未找到'ENSEMBL_ID'列")

    print(f"Excel文件包含 {len(df)} 行数据")

    # 创建映射字典
    gene_name_to_ensg, id_to_ensg = create_id_mapping(gff3_file)

    # 创建新列存储ENSG结果
    ensg_results = []
    match_types = []  # 记录匹配类型

    print("正在进行ID映射...")

    for idx, ensembl_id in enumerate(tqdm(df['ENSEMBL_ID'], desc="映射进度")):
        ensembl_id_str = str(ensembl_id).strip()

        # 跳过空值
        if pd.isna(ensembl_id) or ensembl_id_str == '' or ensembl_id_str == 'nan':
            ensg_results.append(np.nan)
            match_types.append('Empty')
            continue

        # 优先按gene_name匹配
        if ensembl_id_str in gene_name_to_ensg:
            ensg_results.append(gene_name_to_ensg[ensembl_id_str])
            match_types.append('gene_name')
        # 其次按ID匹配
        elif ensembl_id_str in id_to_ensg:
            ensg_results.append(id_to_ensg[ensembl_id_str])
            match_types.append('ID')
        # 如果已经是ENSG格式，直接使用
        elif ensembl_id_str.startswith('ENSG'):
            ensg_results.append(ensembl_id_str)
            match_types.append('Already_ENSG')
        # 未找到匹配
        else:
            ensg_results.append(np.nan)
            match_types.append('Not_found')

    # 添加结果到DataFrame
    df['ENSG_ID'] = ensg_results
    df['Match_Type'] = match_types

    # 统计结果
    match_stats = pd.Series(match_types).value_counts()
    print("\n映射结果统计:")
    print(match_stats)

    total_mapped = len(df) - match_stats.get('Not_found', 0) - match_stats.get('Empty', 0)
    print(f"\n成功映射: {total_mapped}/{len(df)} ({total_mapped / len(df) * 100:.1f}%)")

    # 保存结果
    if output_file is None:
        output_file = xlsx_file.replace('.xlsx', '_with_ensg.xlsx')

    df.to_excel(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    # 显示一些示例结果
    print("\n映射示例:")
    sample_df = df[['ENSEMBL_ID', 'ENSG_ID', 'Match_Type']].head(10)
    print(sample_df.to_string(index=False))

    return df


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    xlsx_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\处理过程\最终数据库集合\LncRNADiseasev3.0收集的数据\lnc相关处理\lncRNA.xlsx"  # 替换为你的Excel文件路径
    gff3_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\处理过程\最终数据库集合\LncRNADiseasev3.0收集的数据\lnc相关处理\gencode.v49.annotation.gff3"  # 替换为你的GFF3文件路径

    # 执行映射
    try:
        result_df = map_ensembl_ids(xlsx_file, gff3_file)
        print("映射完成！")
    except Exception as e:
        print(f"错误: {e}")