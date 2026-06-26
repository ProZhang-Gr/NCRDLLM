import pandas as pd
import numpy as np


def merge_and_deduplicate_diseases(input_file, output_file):
    """
    合并5列Disease_Name，转换为小写，去除重复值
    """
    try:
        # 读取文件
        print("正在读取文件...")
        df = pd.read_excel(input_file)

        print(f"原始文件形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 检查是否有Disease_Name列
        disease_columns = [col for col in df.columns if 'Disease_Name' in col]

        if len(disease_columns) == 0:
            raise ValueError("未找到包含'Disease_Name'的列")

        print(f"找到 {len(disease_columns)} 个Disease_Name列: {disease_columns}")

        # 显示原始数据示例
        print("\n原始数据示例:")
        print(df.head())

        # 合并所有Disease_Name列的数据
        print("\n正在合并列...")
        all_diseases = []

        for col in disease_columns:
            # 获取该列的所有非空值
            column_diseases = df[col].dropna().tolist()
            all_diseases.extend(column_diseases)

        print(f"合并前总数据量: {len(all_diseases)}")

        # 转换为小写
        print("正在转换为小写...")
        lowercase_diseases = []
        original_to_lowercase = {}  # 记录原始值到小写值的映射

        for disease in all_diseases:
            if pd.notna(disease):
                original_disease = str(disease).strip()
                lowercase_disease = original_disease.lower().strip()
                lowercase_diseases.append(lowercase_disease)

                # 记录映射关系（保留第一次出现的原始形式）
                if lowercase_disease not in original_to_lowercase:
                    original_to_lowercase[lowercase_disease] = original_disease

        print(f"转换为小写后数据量: {len(lowercase_diseases)}")

        # 去除重复值
        print("正在去除重复值...")
        unique_diseases = list(set(lowercase_diseases))
        unique_diseases.sort()  # 排序便于查看

        print(f"去重后数据量: {len(unique_diseases)}")

        # 创建结果DataFrame
        result_df = pd.DataFrame({
            'Disease_Name': unique_diseases
        })

        # 保存结果
        print(f"\n正在保存结果到 {output_file}...")
        result_df.to_excel(output_file, index=False)

        # 输出统计信息
        print(f"\n=== 处理统计 ===")
        print(f"原始文件列数: {len(disease_columns)}")
        print(f"合并前总条目: {len(all_diseases)}")
        print(f"转换小写后: {len(lowercase_diseases)}")
        print(f"去重后唯一值: {len(unique_diseases)}")
        print(f"重复条目数: {len(lowercase_diseases) - len(unique_diseases)}")

        # 显示一些示例
        print(f"\n去重后的前10个疾病名称:")
        for i, disease in enumerate(unique_diseases[:10]):
            original_form = original_to_lowercase.get(disease, disease)
            if disease != original_form.lower():
                print(f"  {i + 1}: '{disease}' (原始: '{original_form}')")
            else:
                print(f"  {i + 1}: '{disease}'")

        if len(unique_diseases) > 10:
            print(f"  ... 还有 {len(unique_diseases) - 10} 个")

        print(f"\n结果已保存到: {output_file}")

        return result_df

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


def analyze_duplicates(input_file):
    """
    分析重复情况，显示哪些疾病名称有重复
    """
    try:
        df = pd.read_excel(input_file)

        disease_columns = [col for col in df.columns if 'Disease_Name' in col]

        if len(disease_columns) == 0:
            print("未找到Disease_Name列")
            return

        # 收集所有疾病名称
        all_diseases = []
        disease_sources = {}  # 记录每个疾病来自哪些列

        for col in disease_columns:
            for idx, disease in enumerate(df[col].dropna()):
                if pd.notna(disease):
                    original = str(disease).strip()
                    lowercase = original.lower().strip()
                    all_diseases.append(lowercase)

                    if lowercase not in disease_sources:
                        disease_sources[lowercase] = {
                            'original_forms': set(),
                            'columns': set(),
                            'count': 0
                        }

                    disease_sources[lowercase]['original_forms'].add(original)
                    disease_sources[lowercase]['columns'].add(col)
                    disease_sources[lowercase]['count'] += 1

        # 找出重复的疾病
        print("=== 重复分析 ===")
        duplicates = {k: v for k, v in disease_sources.items() if v['count'] > 1}

        if duplicates:
            print(f"发现 {len(duplicates)} 个重复的疾病名称:")

            for disease, info in sorted(duplicates.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"\n'{disease}' (出现 {info['count']} 次):")
                print(f"  原始形式: {', '.join(sorted(info['original_forms']))}")
                print(f"  出现在列: {', '.join(sorted(info['columns']))}")
        else:
            print("没有发现重复的疾病名称")

        # 显示大小写变化的情况
        case_changes = {}
        for disease, info in disease_sources.items():
            if len(info['original_forms']) > 1:
                case_changes[disease] = info['original_forms']

        if case_changes:
            print(f"\n发现 {len(case_changes)} 个有大小写变化的疾病:")
            for disease, original_forms in sorted(case_changes.items()):
                print(f"  '{disease}' <- {', '.join(sorted(original_forms))}")

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")


def preview_merge(input_file):
    """
    预览合并结果，不保存文件
    """
    try:
        df = pd.read_excel(input_file)

        disease_columns = [col for col in df.columns if 'Disease_Name' in col]

        print("=== 合并预览 ===")
        print(f"找到 {len(disease_columns)} 个Disease_Name列")

        # 统计每列的数据
        for col in disease_columns:
            non_null_count = df[col].notna().sum()
            unique_count = df[col].nunique()
            print(f"  {col}: {non_null_count} 个非空值, {unique_count} 个唯一值")

        # 收集并预览数据
        all_diseases = []
        for col in disease_columns:
            column_data = df[col].dropna().tolist()
            all_diseases.extend(column_data)

        # 转小写并去重
        lowercase_diseases = [str(d).lower().strip() for d in all_diseases if pd.notna(d)]
        unique_diseases = sorted(list(set(lowercase_diseases)))

        print(f"\n合并统计:")
        print(f"  总条目数: {len(all_diseases)}")
        print(f"  转小写后: {len(lowercase_diseases)}")
        print(f"  去重后: {len(unique_diseases)}")
        print(f"  重复条目: {len(lowercase_diseases) - len(unique_diseases)}")

        print(f"\n预览去重后的前15个结果:")
        for i, disease in enumerate(unique_diseases[:15]):
            print(f"  {i + 1}: {disease}")

        if len(unique_diseases) > 15:
            print(f"  ... 还有 {len(unique_diseases) - 15} 个")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\还未收集到的药物\NoDoID.xlsx"  # 输入文件路径
    output_file = "merged_diseases.xlsx"  # 输出文件路径

    try:
        # 预览合并结果
        preview_merge(input_file)

        print("\n" + "=" * 60)

        # 分析重复情况
        analyze_duplicates(input_file)

        print("\n" + "=" * 60)

        # 执行实际合并
        result_df = merge_and_deduplicate_diseases(input_file, output_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 简化版本函数
def simple_merge_diseases(input_file, output_file):
    """
    简化版合并函数
    """
    # 读取文件
    df = pd.read_excel(input_file)

    # 找到所有Disease_Name列
    disease_columns = [col for col in df.columns if 'Disease_Name' in col]

    # 合并所有数据
    all_diseases = []
    for col in disease_columns:
        all_diseases.extend(df[col].dropna().tolist())

    # 转小写并去重
    unique_diseases = list(set([str(d).lower().strip() for d in all_diseases if pd.notna(d)]))
    unique_diseases.sort()

    # 创建结果DataFrame
    result_df = pd.DataFrame({'Disease_Name': unique_diseases})

    # 保存结果
    result_df.to_excel(output_file, index=False)

    print(f"合并完成: {len(all_diseases)} -> {len(unique_diseases)} (去重后)")

    return result_df