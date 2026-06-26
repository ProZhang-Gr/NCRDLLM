import pandas as pd
import re


def clean_disease_id_column(input_file, output_file, disease_id_column='Disease ID'):
    """
    清理Disease ID列，只保留DOID开头的标识符

    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    disease_id_column: Disease ID列的列名
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)

        print(f"原始数据预览:")
        print(df[[disease_id_column]].head(10))
        print("\n" + "=" * 50 + "\n")

        def extract_doid(disease_id_str):
            """提取DOID标识符"""
            if pd.isna(disease_id_str):
                return None

            disease_id_str = str(disease_id_str)

            # 使用正则表达式提取DOID:数字格式
            doid_match = re.search(r'DOID:\d+', disease_id_str)

            if doid_match:
                return doid_match.group()
            else:
                # 如果没有DOID，返回None或空字符串
                return None

        # 应用清理函数
        df[disease_id_column] = df[disease_id_column].apply(extract_doid)

        # 显示清理后的结果
        print(f"清理后数据预览:")
        print(df[[disease_id_column]].head(10))

        # 统计结果
        total_rows = len(df)
        doid_rows = df[disease_id_column].notna().sum()
        removed_rows = total_rows - doid_rows

        print(f"\n清理结果统计:")
        print(f"总行数: {total_rows}")
        print(f"保留DOID的行数: {doid_rows}")
        print(f"清除/置空的行数: {removed_rows}")

        # 保存到新文件
        df.to_excel(output_file, index=False)

        print(f"\n✅ 成功保存清理后的文件: {output_file}")

        # 显示唯一的DOID值
        unique_doids = df[disease_id_column].dropna().unique()
        print(f"\n发现的唯一DOID值 ({len(unique_doids)}个):")
        for doid in sorted(unique_doids):
            print(f"  {doid}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


def preview_changes(input_file, disease_id_column='Disease ID'):
    """预览将要进行的更改"""
    try:
        df = pd.read_excel(input_file)

        print("=== 更改预览 ===")
        print(f"原始 -> 清理后")
        print("-" * 50)

        for i, disease_id in enumerate(df[disease_id_column].head(15)):
            if pd.isna(disease_id):
                continue

            original = str(disease_id)
            doid_match = re.search(r'DOID:\d+', original)
            cleaned = doid_match.group() if doid_match else "删除"

            print(f"{original} -> {cleaned}")

    except Exception as e:
        print(f"❌ 预览错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\zyc收集的数据\lncRNA-disease.xlsx"  # 替换为你的文件名
    output_file = "cleaned_file.xlsx"  # 输出文件名

    # 先预览更改
    print("1. 预览将要进行的更改:")
    preview_changes(input_file)

    print("\n" + "=" * 60 + "\n")

    # 执行清理
    print("2. 执行清理:")
    clean_disease_id_column(input_file, output_file)


# 测试数据示例
def test_cleaning():
    """测试清理函数"""
    test_cases = [
        "DOID:4362//D002583",
        "D009298",
        "DOID:10652//D000544",
        "DOID:1324//D008175",
        "DOID:1612//D001943",
        "D001943",
        "DOID:123"
    ]

    print("=== 清理规则测试 ===")
    for case in test_cases:
        doid_match = re.search(r'DOID:\d+', case)
        result = doid_match.group() if doid_match else "删除"
        print(f"{case} -> {result}")


# 运行测试
print("=== 清理规则演示 ===")
test_cleaning()