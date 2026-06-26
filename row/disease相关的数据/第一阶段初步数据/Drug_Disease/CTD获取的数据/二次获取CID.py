import pandas as pd
import os


def load_failed_chemicals(failed_file):
    """加载没有找到CID的化合物名称列表"""

    print(f"加载失败化合物列表: {failed_file}")

    try:
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_chemicals = [line.strip() for line in f.readlines() if line.strip()]

        print(f"成功加载 {len(failed_chemicals)} 个失败化合物")

        # 显示前10个失败化合物
        print(f"失败化合物示例:")
        for i, chemical in enumerate(failed_chemicals[:10]):
            print(f"  {i + 1}. {chemical}")

        if len(failed_chemicals) > 10:
            print(f"  ... 还有 {len(failed_chemicals) - 10} 个")

        return set(failed_chemicals)  # 转换为集合以提高查找效率

    except Exception as e:
        print(f"加载失败化合物列表出错: {e}")
        return set()


def extract_failed_chemical_records(ctd_file, failed_chemicals, output_file):
    """从CTD数据中提取失败化合物的所有记录"""

    print(f"\n开始提取CTD记录: {ctd_file}")

    try:
        # 读取CTD数据
        df_ctd = pd.read_csv(ctd_file, encoding='utf-8')
        print(f"CTD文件行数: {len(df_ctd)}")
        print(f"CTD文件列名: {list(df_ctd.columns)}")

        # 检查ChemicalName列是否存在
        if 'ChemicalName' not in df_ctd.columns:
            print("错误: CTD文件中没有找到'ChemicalName'列")
            return None

        # 显示原始数据预览
        print(f"\nCTD原始数据预览:")
        print(df_ctd.head(3).to_string(index=False))

        # 统计原始数据
        total_records = len(df_ctd)
        unique_chemicals = df_ctd['ChemicalName'].nunique()

        print(f"\nCTD数据统计:")
        print(f"总记录数: {total_records}")
        print(f"唯一化合物数: {unique_chemicals}")

        # 筛选失败化合物的记录
        print(f"\n开始筛选失败化合物的记录...")

        # 使用isin方法进行筛选
        mask = df_ctd['ChemicalName'].isin(failed_chemicals)
        failed_records = df_ctd[mask].copy()

        print(f"筛选完成!")

        # 统计筛选结果
        extracted_records = len(failed_records)
        found_chemicals = failed_records['ChemicalName'].nunique()

        print(f"\n筛选结果:")
        print(f"提取的记录数: {extracted_records}")
        print(f"找到的失败化合物数: {found_chemicals}")
        print(f"提取率: {extracted_records / total_records * 100:.2f}% (从总记录中)")
        print(f"化合物覆盖率: {found_chemicals / len(failed_chemicals) * 100:.1f}% (失败化合物中)")

        # 检查哪些失败化合物在CTD中没有找到
        found_chemical_names = set(failed_records['ChemicalName'].unique())
        not_found_in_ctd = failed_chemicals - found_chemical_names

        if not_found_in_ctd:
            print(f"\n在CTD中未找到的失败化合物 ({len(not_found_in_ctd)} 个):")
            not_found_list = sorted(list(not_found_in_ctd))
            for i, chemical in enumerate(not_found_list[:10]):
                print(f"  {i + 1}. {chemical}")
            if len(not_found_list) > 10:
                print(f"  ... 还有 {len(not_found_list) - 10} 个")

        # 保存提取的记录
        failed_records.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n提取的记录已保存: {output_file}")

        # 同时保存为Excel格式
        excel_output = output_file.replace('.csv', '.xlsx')
        failed_records.to_excel(excel_output, index=False)
        print(f"Excel格式已保存: {excel_output}")

        # 显示提取数据的预览
        print(f"\n提取数据预览:")
        print(failed_records.head(5).to_string(index=False))

        # 统计每个失败化合物的记录数
        chemical_counts = failed_records['ChemicalName'].value_counts().head(10)
        print(f"\n记录数最多的失败化合物 (前10个):")
        for chemical, count in chemical_counts.items():
            print(f"  {chemical}: {count} 条记录")

        # 统计涉及的疾病
        if 'DiseaseName' in failed_records.columns:
            unique_diseases = failed_records['DiseaseName'].nunique()
            print(f"\n涉及的唯一疾病数: {unique_diseases}")

        return failed_records

    except Exception as e:
        print(f"提取CTD记录时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_failed_chemicals(failed_records):
    """分析失败化合物的特征"""

    if failed_records is None or len(failed_records) == 0:
        print("没有数据可分析")
        return

    print(f"\n失败化合物分析:")
    print("=" * 50)

    # 分析化合物名称特征
    chemical_names = failed_records['ChemicalName'].unique()

    # 统计名称长度
    name_lengths = [len(name) for name in chemical_names]
    avg_length = sum(name_lengths) / len(name_lengths)

    print(f"化合物名称特征:")
    print(f"  平均名称长度: {avg_length:.1f} 字符")
    print(f"  最长名称: {max(name_lengths)} 字符")
    print(f"  最短名称: {min(name_lengths)} 字符")

    # 分析是否包含特殊字符
    special_chars = ['-', '(', ')', ',', '[', ']', "'", '"']
    for char in special_chars:
        count = sum(1 for name in chemical_names if char in name)
        if count > 0:
            print(f"  包含'{char}'的化合物: {count} 个 ({count / len(chemical_names) * 100:.1f}%)")

    # 分析ChemicalID模式
    if 'ChemicalID' in failed_records.columns:
        chemical_ids = failed_records['ChemicalID'].unique()
        c_ids = [cid for cid in chemical_ids if str(cid).startswith('C')]
        print(f"\nChemicalID统计:")
        print(f"  总计: {len(chemical_ids)} 个唯一ID")
        print(f"  C开头的ID: {len(c_ids)} 个")


def main():
    # 文件配置
    failed_file = 'failed_chemicals.txt'  # 失败化合物列表
    ctd_file = 'CTD_chemicals_diseases.csv'  # 原始CTD数据
    output_file = 'failed_chemicals_records.csv'  # 输出的CTD记录

    print("=" * 70)
    print("提取失败化合物的CTD记录")
    print("=" * 70)

    # 检查文件存在性
    if not os.path.exists(failed_file):
        print(f"错误: 找不到失败化合物文件 {failed_file}")
        return

    if not os.path.exists(ctd_file):
        print(f"错误: 找不到CTD文件 {ctd_file}")
        return

    # 加载失败化合物列表
    failed_chemicals = load_failed_chemicals(failed_file)

    if not failed_chemicals:
        print("无法加载失败化合物列表，程序终止")
        return

    # 提取CTD记录
    failed_records = extract_failed_chemical_records(ctd_file, failed_chemicals, output_file)

    if failed_records is not None:
        # 分析失败化合物特征
        analyze_failed_chemicals(failed_records)

        print(f"\n" + "=" * 70)
        print("提取完成!")
        print("=" * 70)
        print(f"失败化合物数: {len(failed_chemicals)}")
        print(f"提取的记录数: {len(failed_records)}")
        print(f"输出文件: {output_file}")
        print(f"Excel文件: {output_file.replace('.csv', '.xlsx')}")
        print("\n现在你可以对这些记录进行二次处理，尝试其他方法获取CID")
        print("=" * 70)
    else:
        print("提取失败")


if __name__ == "__main__":
    main()