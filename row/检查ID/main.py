"""
check_id_consistency.py
用于检查Excel文件中的ID一致性问题
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def check_excel_ids(file_path):
    """
    全面检查Excel文件中的ID问题
    """
    print("=" * 80)
    print(f"📋 检查文件: {file_path}")
    print("=" * 80)

    # 读取Excel文件
    df = pd.read_excel(file_path)
    print(f"\n📊 文件基本信息:")
    print(f"   - 总行数: {len(df)}")
    print(f"   - 列名: {df.columns.tolist()}")

    # 识别ID列
    id_columns = []
    if 'miRBase_ID' in df.columns:
        id_columns.append('miRBase_ID')
    if 'RNA_ID' in df.columns:
        id_columns.append('RNA_ID')
    if 'CID' in df.columns:
        id_columns.append('CID')

    print(f"   - 检测到的ID列: {id_columns}")

    # 对每个ID列进行详细检查
    for col in id_columns:
        print(f"\n{'='*60}")
        print(f"🔍 检查列: {col}")
        print('='*60)

        # 1. 数据类型分析
        print(f"\n1️⃣ 数据类型分析:")
        type_counts = defaultdict(int)
        problematic_values = []

        for idx, value in enumerate(df[col]):
            original_type = type(value).__name__
            type_counts[original_type] += 1

            # 检查问题值
            if pd.isna(value):
                problematic_values.append((idx, value, "空值"))
            elif isinstance(value, float) and not value.is_integer():
                problematic_values.append((idx, value, "浮点数(非整数)"))
            elif isinstance(value, str):
                # 检查字符串问题
                if value != value.strip():
                    problematic_values.append((idx, value, f"含有空白字符: '{value}'"))
                if '\t' in value or '\n' in value or '\r' in value:
                    problematic_values.append((idx, value, "含有制表符或换行符"))

        print(f"   类型分布:")
        for dtype, count in type_counts.items():
            print(f"      - {dtype}: {count} 个 ({count/len(df)*100:.2f}%)")

        # 2. 问题值报告
        if problematic_values:
            print(f"\n2️⃣ 发现问题值: {len(problematic_values)} 个")
            for idx, value, problem in problematic_values[:10]:  # 只显示前10个
                print(f"   行 {idx}: {problem}")
                if not pd.isna(value):
                    print(f"      原始值: '{value}' (type: {type(value).__name__})")
        else:
            print(f"\n2️⃣ 未发现明显问题值 ✅")

        # 3. ID转换测试
        print(f"\n3️⃣ ID标准化转换测试:")
        conversion_issues = []
        unique_before = df[col].nunique()

        # 模拟标准化过程
        standardized = []
        for value in df[col]:
            if pd.isna(value):
                standardized.append(None)
            else:
                # 尝试标准化
                std_value = str(value).strip()
                try:
                    # 如果是数字，转为整数字符串
                    std_value = str(int(float(std_value)))
                except:
                    pass
                standardized.append(std_value)

        unique_after = len(set(filter(lambda x: x is not None, standardized)))

        print(f"   标准化前唯一值: {unique_before}")
        print(f"   标准化后唯一值: {unique_after}")
        if unique_before != unique_after:
            print(f"   ⚠️ 警告: 标准化可能导致ID合并!")

        # 4. 重复值检查
        print(f"\n4️⃣ 重复值检查:")
        duplicates = df[col].value_counts()
        duplicates = duplicates[duplicates > 1]
        if len(duplicates) > 0:
            print(f"   发现 {len(duplicates)} 个重复的ID值:")
            for value, count in duplicates.head(5).items():
                print(f"      '{value}': 出现 {count} 次")
        else:
            print(f"   无重复值 ✅")

        # 5. 特殊字符检查
        print(f"\n5️⃣ 特殊字符检查:")
        special_char_count = 0
        special_examples = []

        for value in df[col].dropna().unique():
            str_value = str(value)
            # 检查是否包含特殊字符
            if any(ord(c) < 32 or ord(c) > 126 for c in str_value if c not in ['-', '_']):
                special_char_count += 1
                special_examples.append(str_value)
                if len(special_examples) >= 5:
                    break

        if special_char_count > 0:
            print(f"   ⚠️ 发现 {special_char_count} 个含特殊字符的ID:")
            for example in special_examples:
                print(f"      '{example}' -> ASCII: {[ord(c) for c in example]}")
        else:
            print(f"   未发现特殊字符 ✅")

    # 如果有两列ID，检查配对情况
    if len(id_columns) >= 2:
        print(f"\n{'='*60}")
        print(f"🔗 检查ID配对情况")
        print('='*60)

        col1, col2 = id_columns[0], id_columns[1]

        # 创建配对
        pairs = []
        for _, row in df.iterrows():
            if not pd.isna(row[col1]) and not pd.isna(row[col2]):
                # 标准化ID
                id1 = str(row[col1]).strip()
                id2 = str(row[col2]).strip()
                try:
                    id2 = str(int(float(id2)))
                except:
                    pass
                pairs.append((id1, id2))

        print(f"\n   总配对数: {len(pairs)}")
        print(f"   唯一配对数: {len(set(pairs))}")

        # 检查重复配对
        from collections import Counter
        pair_counts = Counter(pairs)
        duplicated_pairs = {k: v for k, v in pair_counts.items() if v > 1}

        if duplicated_pairs:
            print(f"\n   ⚠️ 发现重复配对: {len(duplicated_pairs)} 个")
            for pair, count in list(duplicated_pairs.items())[:5]:
                print(f"      {pair}: 重复 {count} 次")
        else:
            print(f"\n   无重复配对 ✅")

    print("\n" + "="*80)
    print("检查完成！")
    print("="*80)

    # 返回诊断结果
    return {
        'total_rows': len(df),
        'id_columns': id_columns,
        'type_counts': type_counts,
        'problematic_values': len(problematic_values),
        'special_char_count': special_char_count
    }


def check_multiple_files(positive_file, feature_files=None):
    """
    检查多个相关文件的ID一致性
    """
    print("\n" + "🔍 开始多文件一致性检查 ".center(80, "="))

    # 检查正样本文件
    print("\n📂 正样本文件:")
    pos_result = check_excel_ids(positive_file)

    # 如果提供了特征文件，也检查它们
    if feature_files:
        for file_path in feature_files:
            print(f"\n📂 特征文件: {file_path}")
            check_excel_ids(file_path)

    print("\n" + "="*80)
    print("✅ 所有文件检查完成!")


if __name__ == "__main__":
    # ========== 配置要检查的文件路径 ==========

    # 正样本文件路径
    POSITIVE_FILE = r"D:\Desktop\CDLLM\ing\data\responsed_miRNA-drug.xlsx"  # 修改为你的文件路径

    # 特征文件路径（可选）
    FEATURE_FILES = [
        # "rna_features.xlsx",
        # "drug_features.xlsx",
    ]

    # ========== 执行检查 ==========

    # 选项1: 只检查单个文件
    check_excel_ids(POSITIVE_FILE)

    # 选项2: 检查多个文件的一致性
    # check_multiple_files(POSITIVE_FILE, FEATURE_FILES)

    # ========== 额外的测试 ==========

    # 测试ID标准化函数
    print("\n" + "="*80)
    print("测试ID标准化逻辑:")
    print("="*80)

    test_values = [
        441203,        # 整数
        441203.0,      # 浮点数
        "441203",      # 字符串
        " 441203 ",    # 带空格的字符串
        "441203.0",    # 字符串形式的浮点数
        "hsa-let-7a",  # RNA ID
    ]

    for value in test_values:
        # 标准化处理
        std_value = str(value).strip()
        try:
            std_value = str(int(float(std_value)))
        except:
            pass

        print(f"原始: {repr(value):20} -> 标准化: {repr(std_value):20}")