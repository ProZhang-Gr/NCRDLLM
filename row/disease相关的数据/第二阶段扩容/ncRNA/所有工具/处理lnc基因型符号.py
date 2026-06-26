import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    logger.info("正在读取GFF3文件...")

    # 存储映射关系的字典
    gene_name_to_ensg = {}
    id_to_ensg = {}
    alias_to_ensg = {}  # 新增：存储基因别名映射

    try:
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

                # 只处理gene行以避免重复
                feature_type = parts[2]
                if feature_type != 'gene':
                    continue

                attributes = parts[8]
                attr_dict = parse_gff3_attributes(attributes)

                # 提取需要的字段
                gene_id = attr_dict.get('gene_id', '').split('.')[0]  # 移除版本号
                gene_name = attr_dict.get('gene_name', '')
                gene_type = attr_dict.get('gene_type', '')

                # 确保gene_id是ENSG格式
                if gene_id and gene_id.startswith('ENSG'):
                    # 添加gene_name映射
                    if gene_name:
                        gene_name_to_ensg[gene_name] = gene_id

                        # 处理可能的别名格式
                        # 如果gene_name包含连字符，也尝试不含连字符的版本
                        if '-' in gene_name:
                            alias_name = gene_name.replace('-', '')
                            alias_to_ensg[alias_name] = gene_id

                # 每处理100000行显示进度
                if line_num % 100000 == 0:
                    logger.info(f"已处理 {line_num} 行...")

    except FileNotFoundError:
        logger.error(f"未找到GFF3文件: {gff3_file}")
        raise
    except Exception as e:
        logger.error(f"读取GFF3文件时出错: {e}")
        raise

    logger.info(f"映射创建完成！")
    logger.info(f"gene_name映射: {len(gene_name_to_ensg)} 条")
    logger.info(f"alias映射: {len(alias_to_ensg)} 条")

    return gene_name_to_ensg, alias_to_ensg


def map_ensembl_ids(xlsx_file, gff3_file, output_file=None):
    """将Excel文件中的ENSEMBL_ID列映射为ENSG标识符"""

    # 读取Excel文件
    logger.info("正在读取Excel文件...")
    try:
        df = pd.read_excel(xlsx_file)
    except FileNotFoundError:
        logger.error(f"未找到Excel文件: {xlsx_file}")
        raise
    except Exception as e:
        logger.error(f"读取Excel文件时出错: {e}")
        raise

    # 检查是否有ENSEMBL_ID列
    if 'ENSEMBL_ID' not in df.columns:
        raise ValueError("Excel文件中未找到'ENSEMBL_ID'列")

    logger.info(f"Excel文件包含 {len(df)} 行数据")

    # 创建映射字典
    gene_name_to_ensg, alias_to_ensg = create_id_mapping(gff3_file)

    # 创建新列存储ENSG结果
    ensg_results = []
    match_types = []  # 记录匹配类型

    logger.info("正在进行ID映射...")

    for idx, ensembl_id in enumerate(tqdm(df['ENSEMBL_ID'], desc="映射进度")):
        ensembl_id_str = str(ensembl_id).strip()

        # 跳过空值
        if pd.isna(ensembl_id) or ensembl_id_str == '' or ensembl_id_str == 'nan':
            ensg_results.append(np.nan)
            match_types.append('Empty')
            continue

        # 如果已经是ENSG格式，直接使用（优先检查）
        if ensembl_id_str.startswith('ENSG'):
            # 移除版本号（如果有）
            ensg_id = ensembl_id_str.split('.')[0]
            ensg_results.append(ensg_id)
            match_types.append('Already_ENSG')
        # 按gene_name精确匹配
        elif ensembl_id_str in gene_name_to_ensg:
            ensg_results.append(gene_name_to_ensg[ensembl_id_str])
            match_types.append('gene_name_exact')
        # 按别名匹配
        elif ensembl_id_str in alias_to_ensg:
            ensg_results.append(alias_to_ensg[ensembl_id_str])
            match_types.append('gene_name_alias')
        # 尝试大小写不敏感匹配
        else:
            found = False
            ensembl_id_upper = ensembl_id_str.upper()
            for gene_name, ensg_id in gene_name_to_ensg.items():
                if gene_name.upper() == ensembl_id_upper:
                    ensg_results.append(ensg_id)
                    match_types.append('gene_name_case_insensitive')
                    found = True
                    break

            if not found:
                ensg_results.append(np.nan)
                match_types.append('Not_found')

    # 添加结果到DataFrame
    df['ENSG_ID'] = ensg_results
    df['Match_Type'] = match_types

    # 统计结果
    match_stats = pd.Series(match_types).value_counts()
    logger.info("\n映射结果统计:")
    for match_type, count in match_stats.items():
        logger.info(f"{match_type}: {count}")

    total_mapped = len(df) - match_stats.get('Not_found', 0) - match_stats.get('Empty', 0)
    success_rate = total_mapped / len(df) * 100
    logger.info(f"\n成功映射: {total_mapped}/{len(df)} ({success_rate:.1f}%)")

    # 保存结果
    if output_file is None:
        output_file = xlsx_file.replace('.xlsx', '_with_ensg.xlsx')

    try:
        df.to_excel(output_file, index=False)
        logger.info(f"\n结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存文件时出错: {e}")
        raise

    # 显示一些示例结果
    logger.info("\n映射示例:")
    sample_df = df[['ENSEMBL_ID', 'ENSG_ID', 'Match_Type']].head(10)
    print(sample_df.to_string(index=False))

    # 显示未找到的条目（前10个）
    not_found = df[df['Match_Type'] == 'Not_found']['ENSEMBL_ID'].head(10)
    if len(not_found) > 0:
        logger.info(f"\n未找到映射的示例 (前10个):")
        for item in not_found:
            print(f"  - {item}")

    return df


def analyze_unmapped_ids(xlsx_file, gff3_file):
    """分析未映射的ID，帮助调试"""
    logger.info("分析未映射的ID...")

    # 执行映射
    df = map_ensembl_ids(xlsx_file, gff3_file)

    # 获取未映射的ID
    unmapped = df[df['Match_Type'] == 'Not_found']['ENSEMBL_ID'].unique()

    if len(unmapped) > 0:
        logger.info(f"\n共有 {len(unmapped)} 个唯一的未映射ID:")

        # 分析ID模式
        patterns = {}
        for uid in unmapped[:20]:  # 只显示前20个
            uid_str = str(uid)
            if uid_str.startswith('LINC'):
                patterns['LINC开头'] = patterns.get('LINC开头', 0) + 1
            elif uid_str.startswith('EN'):
                patterns['EN开头'] = patterns.get('EN开头', 0) + 1
            elif re.match(r'^\d+', uid_str):
                patterns['数字开头'] = patterns.get('数字开头', 0) + 1
            elif len(uid_str) <= 10:
                patterns['短ID'] = patterns.get('短ID', 0) + 1
            else:
                patterns['其他'] = patterns.get('其他', 0) + 1

        logger.info("未映射ID的模式分析:")
        for pattern, count in patterns.items():
            logger.info(f"  {pattern}: {count}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    xlsx_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\第一阶段所接触的数据\LncRNADiseasev3.0收集的数据\lncRNA.xlsx"
    gff3_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\所有工具\gencode.v49.annotation.gff3"

    # 执行映射
    try:
        result_df = map_ensembl_ids(xlsx_file, gff3_file)
        logger.info("映射完成！")

        # 可选：分析未映射的ID
        # analyze_unmapped_ids(xlsx_file, gff3_file)

    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise