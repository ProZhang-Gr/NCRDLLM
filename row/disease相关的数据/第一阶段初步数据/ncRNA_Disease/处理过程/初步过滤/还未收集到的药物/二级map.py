import pandas as pd
import numpy as np
import re


def extract_first_doid(doid_string):
    """
    从DOID字符串中提取第一个DOID
    处理多种情况：
    1. DO:DOID:172|DO:DOID:174|DO:DOID:4322 -> DO:DOID:172
    2. DO:DOID:0090106|OMIM:261640 -> DO:DOID:0090106
    3. OMIM:264300 -> None (没有DOID)
    4. DO:DOID:0060395 -> DO:DOID:0060395
    """
    if pd.isna(doid_string):
        return None

    doid_str = str(doid_string).strip()

    # 按|分割
    parts = doid_str.split('|')

    # 寻找第一个包含DOID的部分
    for part in parts:
        part = part.strip()
        # 检查是否包含DOID（忽略大小写）
        if 'DOID:' in part.upper():
            return part

    # 如果没有找到DOID，返回None
    return None


def normalize_disease_name(name):
    """
    标准化疾病名称用于匹配
    """
    if pd.isna(name):
        return ""

    # 转换为字符串，转小写，去除首尾空格
    name = str(name).lower().strip()

    # 去除多余的空格
    import re
    name = re.sub(r'\s+', ' ', name)

    return name


def create_disease_doid_mapping(map_df):
    """
    创建从疾病名称到DOID的映射字典
    """
    mapping_dict = {}

    for idx, row in map_df.iterrows():
        disease_name = row['Disease_Name']
        doid_string = row['DOID']

        if pd.notna(disease_name):
            # 提取第一个DOID
            first_doid = extract_first_doid(doid_string)

            if first_doid is not None:
                normalized_name = normalize_disease_name(disease_name)
                if normalized_name:
                    mapping_dict[normalized_name] = first_doid

    return mapping_dict


def map_disease_to_doid(xlsx_file, csv_map_file, output_file):
    """
    将xlsx文件中的Disease_Name映射为DOID
    """
    try:
        # 读取文件
        print("正在读取xlsx文件...")
        xlsx_df = pd.read_excel(xlsx_file)

        print("正在读取csv映射文件...")
        csv_df = pd.read_csv(csv_map_file)

        # 检查必要的列
        if 'Disease_Name' not in xlsx_df.columns:
            raise ValueError("xlsx文件中未找到'Disease_Name'列")

        if 'Disease_Name' not in csv_df.columns or 'DOID' not in csv_df.columns:
            raise ValueError("csv文件中未找到'Disease_Name'或'DOID'列")

        print(f"xlsx文件包含 {len(xlsx_df)} 行数据")
        print(f"csv映射文件包含 {len(csv_df)} 行数据")

        # 创建映射字典
        print("正在创建映射字典...")
        mapping_dict = create_disease_doid_mapping(csv_df)
        print(f"成功创建 {len(mapping_dict)} 个映射条目")

        # 显示映射字典示例
        print("\n映射字典示例（前10个）:")
        for i, (disease, doid) in enumerate(list(mapping_dict.items())[:10]):
            print(f"  '{disease}' -> '{doid}'")
        if len(mapping_dict) > 10:
            print("  ...")

        # 执行映射
        print("\n正在执行映射...")
        result_df = xlsx_df.copy()

        mapped_count = 0
        unmapped_count = 0
        unmapped_diseases = []
        mapping_details = []

        new_disease_column = []

        for idx, row in result_df.iterrows():
            original_disease = row['Disease_Name']
            normalized_disease = normalize_disease_name(original_disease)

            if normalized_disease in mapping_dict:
                # 找到映射
                mapped_doid = mapping_dict[normalized_disease]
                new_disease_column.append(mapped_doid)
                mapped_count += 1
                mapping_details.append((original_disease, mapped_doid))
                print(f"  ✓ '{original_disease}' -> '{mapped_doid}'")
            else:
                # 未找到映射
                new_disease_column.append(np.nan)
                unmapped_count += 1
                unmapped_diseases.append(str(original_disease))
                print(f"  ✗ '{original_disease}' -> NaN")

        # 替换Disease_Name列为DOID
        result_df['Disease_Name'] = new_disease_column

        # 保存结果
        print(f"\n正在保存结果到 {output_file}...")
        result_df.to_excel(output_file, index=False)

        # 输出统计信息
        print(f"\n=== 映射统计 ===")
        print(f"总行数: {len(result_df)}")
        print(f"成功映射: {mapped_count}")
        print(f"未找到映射: {unmapped_count}")
        print(f"映射成功率: {mapped_count / len(result_df) * 100:.1f}%")

        # 显示一些映射示例
        if mapping_details:
            print(f"\n成功映射示例（前10个）:")
            for i, (original, mapped) in enumerate(mapping_details[:10]):
                print(f"  {i + 1}: '{original}' -> '{mapped}'")

        # 显示未映射的疾病
        if unmapped_diseases:
            unique_unmapped = sorted(list(set(unmapped_diseases)))
            print(f"\n未找到映射的疾病（前10个）:")
            for i, disease in enumerate(unique_unmapped[:10]):
                print(f"  {i + 1}: '{disease}'")
            if len(unique_unmapped) > 10:
                print(f"  ... 还有 {len(unique_unmapped) - 10} 个")

        print(f"\n结果已保存到: {output_file}")

        return result_df

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


def analyze_csv_doid_patterns(csv_file):
    """
    分析csv文件中DOID列的模式
    """
    try:
        df = pd.read_csv(csv_file)

        if 'DOID' not in df.columns:
            print("csv文件中未找到DOID列")
            return

        print("=== DOID模式分析 ===")

        patterns = {
            'single_doid': [],  # 单个DOID
            'multiple_doid': [],  # 多个DOID
            'mixed_doid_omim': [],  # DOID和OMIM混合
            'only_omim': [],  # 只有OMIM
            'other': []  # 其他格式
        }

        for doid_string in df['DOID'].dropna():
            doid_str = str(doid_string).strip()

            # 统计DOID和OMIM的数量
            doid_count = doid_str.upper().count('DOID:')
            omim_count = doid_str.upper().count('OMIM:')

            if doid_count == 1 and omim_count == 0:
                patterns['single_doid'].append(doid_str)
            elif doid_count > 1 and omim_count == 0:
                patterns['multiple_doid'].append(doid_str)
            elif doid_count >= 1 and omim_count >= 1:
                patterns['mixed_doid_omim'].append(doid_str)
            elif doid_count == 0 and omim_count >= 1:
                patterns['only_omim'].append(doid_str)
            else:
                patterns['other'].append(doid_str)

        # 显示统计
        for pattern_name, examples in patterns.items():
            print(f"\n{pattern_name.replace('_', ' ').title()} ({len(examples)} 个):")
            for example in examples[:3]:
                extracted = extract_first_doid(example)
                if extracted:
                    print(f"  '{example}' -> 提取: '{extracted}'")
                else:
                    print(f"  '{example}' -> 提取: None")
            if len(examples) > 3:
                print(f"  ... 还有 {len(examples) - 3} 个")

    except Exception as e:
        print(f"DOID模式分析错误: {str(e)}")


def preview_mapping(xlsx_file, csv_map_file):
    """
    预览映射结果
    """
    try:
        xlsx_df = pd.read_excel(xlsx_file)
        csv_df = pd.read_csv(csv_map_file)

        mapping_dict = create_disease_doid_mapping(csv_df)

        print("=== 映射预览 ===")
        print(f"映射字典包含 {len(mapping_dict)} 个条目")

        # 预览前20行的映射结果
        preview_count = min(20, len(xlsx_df))
        mapped_preview = 0
        unmapped_preview = 0

        print(f"\n前{preview_count}行映射预览:")
        for idx, row in xlsx_df.head(preview_count).iterrows():
            disease = row['Disease_Name']
            normalized = normalize_disease_name(disease)

            if normalized in mapping_dict:
                doid = mapping_dict[normalized]
                print(f"  {idx + 1}: '{disease}' -> '{doid}' ✓")
                mapped_preview += 1
            else:
                print(f"  {idx + 1}: '{disease}' -> NaN ✗")
                unmapped_preview += 1

        # 统计全部数据
        total_mapped = 0
        total_unmapped = 0

        for _, row in xlsx_df.iterrows():
            normalized = normalize_disease_name(row['Disease_Name'])
            if normalized in mapping_dict:
                total_mapped += 1
            else:
                total_unmapped += 1

        print(f"\n全文件映射统计:")
        print(f"  总行数: {len(xlsx_df)}")
        print(f"  预计映射成功: {total_mapped}")
        print(f"  预计映射失败: {total_unmapped}")
        print(f"  预计成功率: {total_mapped / len(xlsx_df) * 100:.1f}%")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    xlsx_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\还未收集到的药物\merged_diseases.xlsx"  # xlsx文件路径
    csv_map_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\还未收集到的药物\CTD_diseases.csv"  # csv映射文件路径
    output_file = "mapped_result.xlsx"  # 输出文件路径

    try:
        # 分析csv文件中的DOID模式
        analyze_csv_doid_patterns(csv_map_file)

        print("\n" + "=" * 60)

        # 预览映射结果
        preview_mapping(xlsx_file, csv_map_file)

        print("\n" + "=" * 60)

        # 执行实际映射
        result_df = map_disease_to_doid(xlsx_file, csv_map_file, output_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 测试extract_first_doid函数
def test_doid_extraction():
    """
    测试DOID提取功能
    """
    test_cases = [
        "DO:DOID:0060395",
        "OMIM:264300",
        "OMIM:203400|OMIM:610600",
        "OMIM:616034",
        "DO:DOID:0050573|OMIM:236792|OMIM:600721|OMIM:613657|OMIM:615182",
        "OMIM:610006",
        "DO:DOID:172|DO:DOID:174|DO:DOID:4322",
        "DO:DOID:0090106|OMIM:261640"
    ]

    print("=== DOID提取测试 ===")
    for test_case in test_cases:
        result = extract_first_doid(test_case)
        print(f"'{test_case}' -> '{result}'")

# 如果需要测试，取消下面的注释
# test_doid_extraction()