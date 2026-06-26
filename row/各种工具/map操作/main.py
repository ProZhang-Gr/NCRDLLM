import pandas as pd
import numpy as np


def normalize_disease_name(name):
    """
    标准化疾病名称用于匹配
    转换为小写，去除多余空格，统一格式
    """
    if pd.isna(name):
        return ""

    # 转换为字符串，转小写，去除首尾空格
    name = str(name).lower().strip()

    # 去除多余的空格（将多个连续空格替换为单个空格）
    import re
    name = re.sub(r'\s+', ' ', name)

    return name


def create_disease_mapping(map_df):
    """
    创建从标准化疾病名称到Disease_Doid的映射字典
    """
    mapping_dict = {}

    for idx, row in map_df.iterrows():
        disease_name = row['Disease_Name']
        disease_doid = row['Disease_Doid']

        if pd.notna(disease_name) and pd.notna(disease_doid):
            normalized_name = normalize_disease_name(disease_name)
            if normalized_name:  # 确保不是空字符串
                mapping_dict[normalized_name] = disease_doid

    return mapping_dict


def map_disease_names_to_doids(original_file, map_file, output_file):
    """
    将原始文件中的Disease_Name映射为Disease_Doid
    """
    try:
        # 读取文件
        print("正在读取原始文件...")
        original_df = pd.read_excel(original_file)

        print("正在读取映射文件...")
        map_df = pd.read_excel(map_file)

        # 检查必要的列是否存在
        if 'Disease_Name' not in original_df.columns:
            raise ValueError("原始文件中未找到'Disease_Name'列")

        if 'Disease_Name' not in map_df.columns or 'Disease_Doid' not in map_df.columns:
            raise ValueError("映射文件中未找到'Disease_Name'或'Disease_Doid'列")

        # 创建映射字典
        print("正在创建映射字典...")
        mapping_dict = create_disease_mapping(map_df)
        print(f"映射字典中包含 {len(mapping_dict)} 个条目")

        # 显示映射字典的一些示例
        print("\n映射字典示例:")
        for i, (key, value) in enumerate(list(mapping_dict.items())[:5]):
            print(f"  '{key}' -> '{value}'")
        if len(mapping_dict) > 5:
            print("  ...")

        # 执行映射
        print("\n正在执行映射...")
        original_df_copy = original_df.copy()

        mapped_count = 0
        unmapped_count = 0
        unmapped_diseases = set()

        new_disease_column = []

        for idx, row in original_df_copy.iterrows():
            original_disease = row['Disease_Name']
            normalized_disease = normalize_disease_name(original_disease)

            if normalized_disease in mapping_dict:
                # 找到映射
                mapped_doid = mapping_dict[normalized_disease]
                new_disease_column.append(mapped_doid)
                mapped_count += 1
                print(f"  映射成功: '{original_disease}' -> '{mapped_doid}'")
            else:
                # 未找到映射，设置为NaN
                new_disease_column.append(np.nan)
                unmapped_count += 1
                unmapped_diseases.add(str(original_disease))
                print(f"  未找到映射: '{original_disease}' -> NaN")

        # 替换Disease_Name列
        original_df_copy['Disease_Name'] = new_disease_column

        # 保存结果
        print(f"\n正在保存结果到 {output_file}...")
        original_df_copy.to_excel(output_file, index=False)

        # 输出统计信息
        print(f"\n=== 映射统计 ===")
        print(f"总行数: {len(original_df_copy)}")
        print(f"成功映射: {mapped_count}")
        print(f"未找到映射: {unmapped_count}")

        if unmapped_diseases:
            print(f"\n未映射的疾病名称:")
            for disease in sorted(unmapped_diseases):
                print(f"  - {disease}")

        print(f"\n结果已保存到: {output_file}")

        return original_df_copy

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


def preview_mapping(original_file, map_file):
    """
    预览映射过程，不保存文件
    """
    try:
        original_df = pd.read_excel(original_file)
        map_df = pd.read_excel(map_file)

        mapping_dict = create_disease_mapping(map_df)

        print("=== 原始文件前10行的映射预览 ===")
        for idx, row in original_df.head(10).iterrows():
            original_disease = row['Disease_Name']
            normalized_disease = normalize_disease_name(original_disease)

            if normalized_disease in mapping_dict:
                mapped_doid = mapping_dict[normalized_disease]
                status = "✓ 找到映射"
            else:
                mapped_doid = "NaN"
                status = "✗ 未找到映射"

            print(f"行{idx + 1}: '{original_disease}' -> '{mapped_doid}' ({status})")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


def find_similar_diseases(original_file, map_file, threshold=0.99):
    """
    寻找相似的疾病名称，帮助调试未匹配的情况
    """
    try:
        from difflib import SequenceMatcher

        original_df = pd.read_excel(original_file)
        map_df = pd.read_excel(map_file)

        # 获取所有疾病名称
        original_diseases = [normalize_disease_name(name) for name in original_df['Disease_Name'].unique()]
        map_diseases = [normalize_disease_name(name) for name in map_df['Disease_Name'].unique()]

        print("=== 相似疾病名称分析 ===")
        for orig_disease in original_diseases:
            if orig_disease:  # 跳过空字符串
                best_match = ""
                best_score = 0

                for map_disease in map_diseases:
                    if map_disease:
                        score = SequenceMatcher(None, orig_disease, map_disease).ratio()
                        if score > best_score:
                            best_score = score
                            best_match = map_disease

                if best_score >= threshold:
                    print(f"'{orig_disease}' 与 '{best_match}' 相似度: {best_score:.3f}")
                elif best_score > 0.5:  # 显示中等相似度的匹配
                    print(f"'{orig_disease}' 与 '{best_match}' 相似度: {best_score:.3f} (低于阈值)")

    except ImportError:
        print("需要安装difflib库来进行相似度分析")
    except Exception as e:
        print(f"相似度分析过程中出现错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    original_file = r"D:\Desktop\CDLLM\ing\row\各种工具\map操作\原始数据\miR2Disease收集的信息\cleaned_file.xlsx"  # 原始数据文件路径
    map_file = r"D:\Desktop\CDLLM\ing\row\各种工具\map操作\原始数据\DiseaseName2DOID.xlsx"  # 映射文件路径
    output_file = "mapped_result.xlsx"  # 输出文件路径

    try:
        # 首先预览映射情况
        print("=== 预览映射情况 ===")
        preview_mapping(original_file, map_file)

        print("\n" + "=" * 50)

        # 执行映射
        result_df = map_disease_names_to_doids(original_file, map_file, output_file)

        print("\n" + "=" * 50)

        # 分析相似疾病名称（可选）
        print("\n=== 寻找相似疾病名称（帮助调试）===")
        find_similar_diseases(original_file, map_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 单独的简化版本（如果你只需要基本功能）
def simple_mapping(original_file, map_file, output_file):
    """
    简化版映射函数
    """
    # 读取文件
    original_df = pd.read_excel(original_file)
    map_df = pd.read_excel(map_file)

    # 创建映射字典（忽略大小写）
    mapping_dict = {}
    for _, row in map_df.iterrows():
        if pd.notna(row['Disease_Name']) and pd.notna(row['Disease_Doid']):
            key = str(row['Disease_Name']).lower().strip()
            mapping_dict[key] = row['Disease_Doid']

    # 执行映射
    def map_disease(disease_name):
        if pd.isna(disease_name):
            return np.nan
        key = str(disease_name).lower().strip()
        return mapping_dict.get(key, np.nan)

    # 添加新的Disease_Doid列，保留原始Disease_Name列
    original_df['Disease_Doid'] = original_df['Disease_Name'].apply(map_disease)

    # 保存结果
    original_df.to_excel(output_file, index=False)

    return original_df