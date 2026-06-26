import pandas as pd
import numpy as np
import re


def normalize_disease_name(name):
    """
    标准化疾病名称用于匹配
    转换为小写，去除多余空格
    """
    if pd.isna(name):
        return ""

    # 转换为字符串，转小写，去除首尾空格
    name = str(name).lower().strip()

    # 去除多余的空格（将多个连续空格替换为单个空格）
    name = re.sub(r'\s+', ' ', name)

    return name


def has_doid(doid_value):
    """
    检查DOID列是否已有值
    """
    if pd.isna(doid_value):
        return False

    doid_str = str(doid_value).strip()
    return bool(doid_str and 'DOID:' in doid_str.upper())


def create_disease_mapping(map_df):
    """
    从map文件创建疾病名称到DOID的映射字典
    """
    mapping_dict = {}

    for idx, row in map_df.iterrows():
        disease_name = row['Disease_Name']
        doid = row['DOID']

        if pd.notna(disease_name) and pd.notna(doid):
            normalized_name = normalize_disease_name(disease_name)
            if normalized_name and str(doid).strip():
                # 如果同一个疾病名称有多个DOID，保留第一个
                if normalized_name not in mapping_dict:
                    mapping_dict[normalized_name] = str(doid).strip()

    return mapping_dict


def fill_missing_doids(main_file, map_file, output_file):
    """
    补全缺失的DOID
    """
    try:
        # 读取文件
        print("正在读取主文件...")
        main_df = pd.read_excel(main_file)

        print("正在读取映射文件...")
        map_df = pd.read_excel(map_file)

        # 检查必要的列
        required_cols_main = ['ENSEMBL_ID', 'Disease_Name', 'DOID']
        required_cols_map = ['Disease_Name', 'DOID']

        for col in required_cols_main:
            if col not in main_df.columns:
                raise ValueError(f"主文件中未找到'{col}'列")

        for col in required_cols_map:
            if col not in map_df.columns:
                raise ValueError(f"映射文件中未找到'{col}'列")

        print(f"主文件包含 {len(main_df)} 行数据")
        print(f"映射文件包含 {len(map_df)} 行数据")

        # 创建映射字典
        print("正在创建映射字典...")
        mapping_dict = create_disease_mapping(map_df)
        print(f"成功创建 {len(mapping_dict)} 个映射条目")

        # 显示映射字典示例
        print("\n映射字典示例（前10个）:")
        for i, (disease, doid) in enumerate(list(mapping_dict.items())[:10]):
            print(f"  '{disease}' -> '{doid}'")
        if len(mapping_dict) > 10:
            print("  ...")

        # 统计当前状态
        has_doid_mask = main_df['DOID'].apply(has_doid)
        already_have_doid = has_doid_mask.sum()
        need_doid = len(main_df) - already_have_doid

        print(f"\n=== 当前状态 ===")
        print(f"总行数: {len(main_df)}")
        print(f"已有DOID: {already_have_doid}")
        print(f"需要补全DOID: {need_doid}")

        # 执行DOID补全
        print("\n正在补全DOID...")
        result_df = main_df.copy()

        filled_count = 0
        not_found_count = 0
        fill_details = []
        not_found_diseases = []

        for idx, row in result_df.iterrows():
            if not has_doid(row['DOID']):
                # 需要补全DOID
                disease_name = row['Disease_Name']
                normalized_name = normalize_disease_name(disease_name)

                if normalized_name in mapping_dict:
                    # 找到映射
                    mapped_doid = mapping_dict[normalized_name]
                    result_df.at[idx, 'DOID'] = mapped_doid
                    filled_count += 1
                    fill_details.append((disease_name, mapped_doid))
                    print(f"  ✓ '{disease_name}' -> '{mapped_doid}'")
                else:
                    # 未找到映射
                    not_found_count += 1
                    not_found_diseases.append(disease_name)
                    print(f"  ✗ '{disease_name}' -> 未找到映射")
            else:
                # 已经有DOID，跳过
                print(f"  → '{row['Disease_Name']}' 已有DOID: {row['DOID']}")

        # 保存结果
        print(f"\n正在保存结果到 {output_file}...")
        result_df.to_excel(output_file, index=False)

        # 最终统计
        final_has_doid = result_df['DOID'].apply(has_doid).sum()
        final_missing_doid = len(result_df) - final_has_doid

        print(f"\n=== 最终统计 ===")
        print(f"原本已有DOID: {already_have_doid}")
        print(f"新补全DOID: {filled_count}")
        print(f"未找到映射: {not_found_count}")
        print(f"最终DOID覆盖率: {final_has_doid}/{len(result_df)} ({final_has_doid / len(result_df) * 100:.1f}%)")

        # 显示补全详情
        if fill_details:
            print(f"\n新补全的DOID（前10个）:")
            for i, (disease, doid) in enumerate(fill_details[:10]):
                print(f"  {i + 1}: '{disease}' -> '{doid}'")
            if len(fill_details) > 10:
                print(f"  ... 还有 {len(fill_details) - 10} 个")

        # 显示未找到映射的疾病
        if not_found_diseases:
            unique_not_found = sorted(list(set(not_found_diseases)))
            print(f"\n未找到映射的疾病（前10个）:")
            for i, disease in enumerate(unique_not_found[:10]):
                print(f"  {i + 1}: '{disease}'")
            if len(unique_not_found) > 10:
                print(f"  ... 还有 {len(unique_not_found) - 10} 个")

        print(f"\n结果已保存到: {output_file}")

        return result_df

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


def preview_mapping(main_file, map_file):
    """
    预览映射情况，不保存文件
    """
    try:
        main_df = pd.read_excel(main_file)
        map_df = pd.read_excel(map_file)

        mapping_dict = create_disease_mapping(map_df)

        print("=== 映射预览 ===")
        print(f"映射字典包含 {len(mapping_dict)} 个条目")

        # 分析主文件中的疾病
        has_doid_count = 0
        need_mapping_count = 0
        can_map_count = 0
        cannot_map_count = 0

        preview_details = []

        for idx, row in main_df.iterrows():
            disease_name = row['Disease_Name']
            current_doid = row['DOID']

            if has_doid(current_doid):
                has_doid_count += 1
                status = f"已有DOID: {current_doid}"
            else:
                need_mapping_count += 1
                normalized_name = normalize_disease_name(disease_name)

                if normalized_name in mapping_dict:
                    can_map_count += 1
                    mapped_doid = mapping_dict[normalized_name]
                    status = f"可映射到: {mapped_doid}"
                else:
                    cannot_map_count += 1
                    status = "无法映射"

            preview_details.append((disease_name, status))

        print(f"\n预览统计:")
        print(f"  总行数: {len(main_df)}")
        print(f"  已有DOID: {has_doid_count}")
        print(f"  需要映射: {need_mapping_count}")
        print(f"  可以映射: {can_map_count}")
        print(f"  无法映射: {cannot_map_count}")
        print(f"  预计最终覆盖率: {(has_doid_count + can_map_count) / len(main_df) * 100:.1f}%")

        print(f"\n详细预览（前15行）:")
        for i, (disease, status) in enumerate(preview_details[:15]):
            print(f"  {i + 1}: '{disease}' -> {status}")

        if len(preview_details) > 15:
            print(f"  ... 还有 {len(preview_details) - 15} 行")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


def analyze_map_file(map_file):
    """
    分析映射文件的内容
    """
    try:
        df = pd.read_excel(map_file)

        print("=== 映射文件分析 ===")
        print(f"总行数: {len(df)}")
        print(f"唯一疾病名称: {df['Disease_Name'].nunique()}")
        print(f"唯一DOID: {df['DOID'].nunique()}")

        # 检查重复
        duplicate_diseases = df[df.duplicated(['Disease_Name'], keep=False)]
        if not duplicate_diseases.empty:
            print(f"\n发现重复的疾病名称 ({len(duplicate_diseases)} 行):")
            for disease in duplicate_diseases['Disease_Name'].unique()[:5]:
                matching_rows = df[df['Disease_Name'] == disease]
                doids = matching_rows['DOID'].tolist()
                print(f"  '{disease}': {doids}")

        # 显示示例
        print(f"\n映射文件示例（前10行）:")
        for idx, row in df.head(10).iterrows():
            print(f"  '{row['Disease_Name']}' -> '{row['DOID']}'")

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")


def find_partial_matches(target_disease, mapping_dict, threshold=0.7):
    """
    寻找部分匹配的疾病名称
    """
    try:
        from difflib import SequenceMatcher

        target_normalized = normalize_disease_name(target_disease)
        matches = []

        for disease_name, doid in mapping_dict.items():
            similarity = SequenceMatcher(None, target_normalized, disease_name).ratio()
            if similarity >= threshold:
                matches.append((disease_name, doid, similarity))

        return sorted(matches, key=lambda x: x[2], reverse=True)[:3]

    except ImportError:
        return []


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    main_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\LncRNADiseasev3.0收集的数据\lncRNA.xlsx"  # 主数据文件
    map_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\二次map.xlsx"  # 映射文件
    output_file = "completed_data.xlsx"  # 输出文件

    try:
        # 分析映射文件
        analyze_map_file(map_file)

        print("\n" + "=" * 60)

        # 预览映射情况
        preview_mapping(main_file, map_file)

        print("\n" + "=" * 60)

        # 执行DOID补全
        result_df = fill_missing_doids(main_file, map_file, output_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 简化版本函数
def simple_fill_doids(main_file, map_file, output_file):
    """
    简化版DOID补全函数
    """
    # 读取文件
    main_df = pd.read_excel(main_file)
    map_df = pd.read_excel(map_file)

    # 创建映射字典
    mapping_dict = create_disease_mapping(map_df)

    # 补全DOID
    filled_count = 0
    for idx, row in main_df.iterrows():
        if not has_doid(row['DOID']):
            normalized_name = normalize_disease_name(row['Disease_Name'])
            if normalized_name in mapping_dict:
                main_df.at[idx, 'DOID'] = mapping_dict[normalized_name]
                filled_count += 1

    # 保存结果
    main_df.to_excel(output_file, index=False)

    print(f"补全了 {filled_count} 个DOID")
    return main_df