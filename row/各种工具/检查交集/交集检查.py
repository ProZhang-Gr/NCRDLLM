import pandas as pd
import os


def compare_first_columns(file1_path, file2_path):
    """比较两个Excel文件第一列的交集"""

    try:
        # 读取两个文件
        df1 = pd.read_excel(file1_path)
        df2 = pd.read_excel(file2_path)

        # 获取第一列数据
        col1_name = df1.columns[0]
        col2_name = df2.columns[0]

        # 提取第一列的唯一值，转换为字符串集合
        set1 = set(str(val).strip() for val in df1.iloc[:, 0].dropna())
        set2 = set(str(val).strip() for val in df2.iloc[:, 0].dropna())

        # 计算交集
        intersection = set1.intersection(set2)

        # 输出结果
        print(f"文件1: {os.path.basename(file1_path)}")
        print(f"  第一列名: {col1_name}")
        print(f"  唯一值数量: {len(set1)}")

        print(f"文件2: {os.path.basename(file2_path)}")
        print(f"  第一列名: {col2_name}")
        print(f"  唯一值数量: {len(set2)}")

        print(f"交集数量: {len(intersection)}")
        print(f"交集比例: {len(intersection) / min(len(set1), len(set2)) * 100:.1f}% (相对较小集合)")

        # 如果交集不为空且数量适中，显示具体值
        if 0 < len(intersection) <= 20:
            print(f"交集内容: {sorted(list(intersection))}")
        elif len(intersection) > 20:
            sorted_intersection = sorted(list(intersection))
            print(f"交集前10个: {sorted_intersection[:10]}")

        print("-" * 60)
        return len(intersection)

    except Exception as e:
        print(f"错误: {e}")
        print("-" * 60)
        return -1


def main():
    # =================================================================
    # 在这里直接修改要比较的文件路径
    # =================================================================

    file_pairs = [
        # 示例2: 比较circRNA数据
        (
            r"D:\Desktop\CDLLM\ing\official\ncRNA-Disease\miRNA_Disease.xlsx",
            r"D:\Desktop\CDLLM\ing\official\miRNA-drug.xlsx"
        ),

        # # 示例3: 比较lncRNA数据
        # (
        #     r"D:\Desktop\CDLLM\ing\row\各种工具\检查交集\ALLlncRNA-seq.xlsx",
        #     r"D:\Desktop\CDLLM\ing\row\各种工具\检查交集\lncRNA_Disease.xlsx"
        # ),
        #
        # # 示例4: 比较miRNA数据
        # (
        #     r"D:\Desktop\CDLLM\ing\row\各种工具\检查交集\ALLmiRNA-seq.xlsx",
        #     r"D:\Desktop\CDLLM\ing\row\各种工具\检查交集\miRNA_Disease.xlsx"
        # ),
        # # 示例4: 比较miRNA数据
        # (
        #     r"D:\Desktop\CDLLM\ing\row\各种工具\检查交集\ALLdrug-smiles.xlsx",
        #     r"D:\Desktop\CDLLM\ing\row\各种工具\检查交集\Drug_Disease.xlsx"
        # )
        # 添加更多文件对比较...
        # (
        #     r"路径1",
        #     r"路径2"
        # ),
    ]

    # =================================================================
    # 批量比较所有文件对
    # =================================================================

    print("=" * 60)
    print("批量文件第一列交集检查")
    print("=" * 60)

    results = []

    for i, (file1, file2) in enumerate(file_pairs, 1):
        print(f"\n检查对 {i}:")

        # 检查文件是否存在
        if not os.path.exists(file1):
            print(f"错误: 文件不存在 {file1}")
            continue

        if not os.path.exists(file2):
            print(f"错误: 文件不存在 {file2}")
            continue

        # 比较文件
        intersection_count = compare_first_columns(file1, file2)

        if intersection_count >= 0:
            results.append({
                'pair': i,
                'file1': os.path.basename(file1),
                'file2': os.path.basename(file2),
                'intersection': intersection_count
            })

    # 汇总结果
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)

    for result in results:
        print(f"对{result['pair']}: {result['file1']} vs {result['file2']} -> 交集: {result['intersection']}")

    print(f"\n总计检查了 {len(results)} 对文件")


if __name__ == "__main__":
    main()