import pandas as pd
import os
from pathlib import Path


def merge_rna_drug_data(base_path):
    """
    合并RNA-drug数据集，按类型分别处理circRNA、miRNA、lncRNA

    Args:
        base_path (str): 包含数据文件夹的根目录路径
    """

    # 定义RNA类型及其对应的标识符列名
    rna_types = {
        'circRNA-drug处理过程': 'circBase_ID',
        'miRNA-drug处理过程': 'miRBase_ID',
        'lncRNA-drug处理过程': 'ENSEMBL_ID'
    }

    # 定义要合并的文件名模式
    file_patterns = [
        'ncRNAdrug第一步收集.xlsx',
        'NoncoRNA第一步收集.xlsx',
        'RNAInter第一步收集.xlsx'
    ]

    # 存储合并结果
    merged_results = {}

    for folder_name, id_column in rna_types.items():
        print(f"\n处理 {folder_name}...")
        folder_path = Path(base_path) / folder_name

        if not folder_path.exists():
            print(f"警告: 文件夹 {folder_path} 不存在，跳过...")
            continue

        # 存储当前RNA类型的所有数据
        all_data = []

        for file_pattern in file_patterns:
            file_path = folder_path / file_pattern

            if file_path.exists():
                try:
                    # 读取Excel文件
                    df = pd.read_excel(file_path)

                    # 检查数据格式
                    if len(df.columns) >= 2:
                        # 重命名列
                        df.columns = [id_column, 'CID'] + list(df.columns[2:])

                        # 只保留前两列
                        df_clean = df[[id_column, 'CID']].copy()

                        # 移除空值
                        df_clean = df_clean.dropna()

                        print(f"  从 {file_pattern} 读取 {len(df_clean)} 条记录")
                        all_data.append(df_clean)
                    else:
                        print(f"  警告: {file_pattern} 格式不正确，列数少于2列")

                except Exception as e:
                    print(f"  错误: 无法读取 {file_pattern}: {str(e)}")
            else:
                print(f"  警告: {file_pattern} 不存在")

        # 合并所有数据
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)

            # 去重 - 基于RNA ID和CID的组合去重
            print(f"  合并前总记录数: {len(merged_df)}")
            merged_df = merged_df.drop_duplicates(subset=[id_column, 'CID'])
            print(f"  去重后记录数: {len(merged_df)}")

            # 按RNA ID排序
            merged_df = merged_df.sort_values(by=id_column).reset_index(drop=True)

            # 存储结果
            rna_type_name = folder_name.replace('-drug处理过程', '')
            merged_results[rna_type_name] = merged_df

        else:
            print(f"  没有找到有效数据")

    return merged_results


def save_merged_data(merged_results, output_dir='merged_data'):
    """
    保存合并后的数据到文件

    Args:
        merged_results (dict): 合并后的数据字典
        output_dir (str): 输出目录
    """

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n保存合并结果到 {output_path}...")

    for rna_type, df in merged_results.items():
        # 保存为Excel文件
        excel_file = output_path / f"{rna_type}_merged.xlsx"
        df.to_excel(excel_file, index=False)
        print(f"  {rna_type}: {len(df)} 条记录 -> {excel_file}")

        # 同时保存为CSV文件（可选）
        csv_file = output_path / f"{rna_type}_merged.csv"
        df.to_csv(csv_file, index=False)
        print(f"  {rna_type}: CSV格式 -> {csv_file}")


def print_summary(merged_results):
    """
    打印数据汇总信息
    """
    print("\n" + "=" * 50)
    print("数据合并汇总:")
    print("=" * 50)

    for rna_type, df in merged_results.items():
        print(f"\n{rna_type}:")
        print(f"  总记录数: {len(df)}")
        print(f"  唯一RNA数量: {df.iloc[:, 0].nunique()}")
        print(f"  唯一CID数量: {df['CID'].nunique()}")

        # 显示前5条记录
        print(f"  前5条记录:")
        for i in range(min(5, len(df))):
            print(f"    {df.iloc[i, 0]} -> {df.iloc[i, 1]}")


def main():
    """
    主函数
    """
    print("RNA-Drug数据合并工具")
    print("=" * 30)

    # 设置数据路径 - 请修改为您的实际路径
    base_path = "."  # 当前目录，请根据实际情况修改

    try:
        # 合并数据
        merged_results = merge_rna_drug_data(base_path)

        if merged_results:
            # 打印汇总
            print_summary(merged_results)

            # 保存结果
            save_merged_data(merged_results)

            print(f"\n✅ 数据合并完成！共处理 {len(merged_results)} 种RNA类型的数据")
        else:
            print("\n❌ 没有找到可处理的数据")

    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()