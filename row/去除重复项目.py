import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def process_two_column_file(file_path, col1_name=None, col2_name=None, save_results=True):
    """
    处理包含两列的文件：去重并统计

    Args:
        file_path (str): 文件路径
        col1_name (str): 第一列的列名，如果为None则使用默认名称
        col2_name (str): 第二列的列名，如果为None则使用默认名称
        save_results (bool): 是否保存结果到文件

    Returns:
        dict: 包含处理结果和统计信息的字典
    """

    print(f"正在处理文件: {file_path}")

    # 读取文件
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t')
        else:
            # 尝试以不同分隔符读取
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}: {str(e)}")
        return None

    # 检查数据格式
    if len(df.columns) < 2:
        print(f"错误：文件至少需要包含两列数据，当前只有 {len(df.columns)} 列")
        return None

    # 只保留前两列
    df = df.iloc[:, :2].copy()

    # 设置列名
    if col1_name and col2_name:
        df.columns = [col1_name, col2_name]
    elif col1_name:
        df.columns = [col1_name, df.columns[1]]
    elif col2_name:
        df.columns = [df.columns[0], col2_name]
    else:
        # 使用默认列名
        df.columns = ['Column1', 'Column2']

    col1, col2 = df.columns[0], df.columns[1]

    print(f"列名: {col1}, {col2}")
    print(f"原始数据行数: {len(df)}")

    # 移除空值
    df_clean = df.dropna()
    print(f"移除空值后行数: {len(df_clean)}")

    # 去重前的统计
    original_stats = {
        'total_rows': len(df_clean),
        'unique_col1': df_clean[col1].nunique(),
        'unique_col2': df_clean[col2].nunique(),
        'unique_pairs': len(df_clean.drop_duplicates())
    }

    # 去重 - 基于两列组合去重
    df_dedup = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"去重后行数: {len(df_dedup)}")
    print(f"去除的重复行数: {len(df_clean) - len(df_dedup)}")

    # 去重后的统计
    deduplicated_stats = {
        'total_rows': len(df_dedup),
        'unique_col1': df_dedup[col1].nunique(),
        'unique_col2': df_dedup[col2].nunique(),
        'unique_pairs': len(df_dedup)
    }

    # 详细统计分析
    col1_counts = df_dedup[col1].value_counts()
    col2_counts = df_dedup[col2].value_counts()

    # 构建结果字典
    results = {
        'original_data': df_clean,
        'deduplicated_data': df_dedup,
        'original_stats': original_stats,
        'deduplicated_stats': deduplicated_stats,
        'col1_counts': col1_counts,
        'col2_counts': col2_counts,
        'column_names': [col1, col2]
    }

    # 打印详细统计
    print_detailed_statistics(results)

    # 保存结果
    if save_results:
        save_analysis_results(results, file_path)

    return results


def print_detailed_statistics(results):
    """
    打印详细的统计信息
    """
    col1, col2 = results['column_names']

    print("\n" + "=" * 60)
    print("详细统计分析")
    print("=" * 60)

    # 基本统计对比
    print(f"\n📊 基本统计对比:")
    print(f"{'项目':<20} {'原始数据':<15} {'去重后':<15} {'差异':<10}")
    print("-" * 60)

    orig = results['original_stats']
    deduplicated = results['deduplicated_stats']

    print(
        f"{'总行数':<20} {orig['total_rows']:<15} {deduplicated['total_rows']:<15} {orig['total_rows'] - deduplicated['total_rows']:<10}")
    print(
        f"{f'唯一{col1}数量':<20} {orig['unique_col1']:<15} {deduplicated['unique_col1']:<15} {orig['unique_col1'] - deduplicated['unique_col1']:<10}")
    print(
        f"{f'唯一{col2}数量':<20} {orig['unique_col2']:<15} {deduplicated['unique_col2']:<15} {orig['unique_col2'] - deduplicated['unique_col2']:<10}")

    # 重复率统计
    duplicate_rate = (orig['total_rows'] - deduplicated['total_rows']) / orig['total_rows'] * 100
    print(f"\n🔄 重复率: {duplicate_rate:.2f}%")

    # Top统计
    print(f"\n🔝 {col1} 出现频次Top 10:")
    for i, (value, count) in enumerate(results['col1_counts'].head(10).items(), 1):
        print(f"  {i:2d}. {value}: {count} 次")

    print(f"\n🔝 {col2} 出现频次Top 10:")
    for i, (value, count) in enumerate(results['col2_counts'].head(10).items(), 1):
        print(f"  {i:2d}. {value}: {count} 次")

    # 分布统计
    col1_dist = results['col1_counts'].describe()
    col2_dist = results['col2_counts'].describe()

    print(f"\n📈 {col1} 频次分布统计:")
    print(f"  平均频次: {col1_dist['mean']:.2f}")
    print(f"  中位数: {col1_dist['50%']:.2f}")
    print(f"  最大频次: {int(col1_dist['max'])}")
    print(f"  最小频次: {int(col1_dist['min'])}")

    print(f"\n📈 {col2} 频次分布统计:")
    print(f"  平均频次: {col2_dist['mean']:.2f}")
    print(f"  中位数: {col2_dist['50%']:.2f}")
    print(f"  最大频次: {int(col2_dist['max'])}")
    print(f"  最小频次: {int(col2_dist['min'])}")


def save_analysis_results(results, original_file_path):
    """
    保存分析结果到文件
    """
    file_path = Path(original_file_path)
    output_dir = file_path.parent / f"{file_path.stem}_analysis"
    output_dir.mkdir(exist_ok=True)

    col1, col2 = results['column_names']

    print(f"\n💾 保存结果到: {output_dir}")

    # 1. 保存去重后的数据
    dedup_file = output_dir / f"{file_path.stem}_deduplicated.xlsx"
    results['deduplicated_data'].to_excel(dedup_file, index=False)
    print(f"  ✅ 去重数据: {dedup_file}")

    # 2. 保存统计报告
    stats_file = output_dir / f"{file_path.stem}_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据去重统计报告\n")
        f.write("=" * 50 + "\n\n")

        # 基本统计
        f.write("基本统计:\n")
        f.write(f"原始行数: {results['original_stats']['total_rows']}\n")
        f.write(f"去重后行数: {results['deduplicated_stats']['total_rows']}\n")
        f.write(
            f"去除重复行数: {results['original_stats']['total_rows'] - results['deduplicated_stats']['total_rows']}\n")
        f.write(
            f"重复率: {(results['original_stats']['total_rows'] - results['deduplicated_stats']['total_rows']) / results['original_stats']['total_rows'] * 100:.2f}%\n\n")

        # 详细统计
        f.write(f"{col1} 统计 (Top 20):\n")
        for value, count in results['col1_counts'].head(20).items():
            f.write(f"  {value}: {count}\n")

        f.write(f"\n{col2} 统计 (Top 20):\n")
        for value, count in results['col2_counts'].head(20).items():
            f.write(f"  {value}: {count}\n")

    print(f"  ✅ 统计报告: {stats_file}")

    # 3. 保存频次统计表
    col1_stats_file = output_dir / f"{col1}_frequency.xlsx"
    col1_df = results['col1_counts'].reset_index()
    col1_df.columns = [col1, 'Count']
    col1_df.to_excel(col1_stats_file, index=False)
    print(f"  ✅ {col1}频次统计: {col1_stats_file}")

    col2_stats_file = output_dir / f"{col2}_frequency.xlsx"
    col2_df = results['col2_counts'].reset_index()
    col2_df.columns = [col2, 'Count']
    col2_df.to_excel(col2_stats_file, index=False)
    print(f"  ✅ {col2}频次统计: {col2_stats_file}")


def process_multiple_files(file_list, col1_name=None, col2_name=None):
    """
    批量处理多个文件
    """
    all_results = {}

    print("批量处理文件模式")
    print("=" * 30)

    for file_path in file_list:
        print(f"\n处理文件: {file_path}")
        result = process_two_column_file(file_path, col1_name, col2_name)
        if result:
            all_results[file_path] = result

    # 汇总统计
    if all_results:
        print("\n" + "=" * 60)
        print("批量处理汇总")
        print("=" * 60)

        for file_path, result in all_results.items():
            file_name = Path(file_path).name
            stats = result['deduplicated_stats']
            print(f"{file_name}:")
            print(f"  去重后行数: {stats['total_rows']}")
            print(f"  唯一值1: {stats['unique_col1']}")
            print(f"  唯一值2: {stats['unique_col2']}")

    return all_results


def main():
    """
    主函数 - 提供不同的使用方式
    """
    print("两列数据去重统计工具")
    print("=" * 30)

    # 方式1: 处理单个文件
    file_path = r"D:\Desktop\CDLLM\项目进行时\row\miRNA-drug处理过程\RNAInter第一步收集.xlsx"  # 修改为您的文件路径
    results = process_two_column_file(file_path,
                                      col1_name="miRBase_ID",  # 可选：指定第一列名称
                                      col2_name="CID")  # 可选：指定第二列名称

    # 方式2: 批量处理多个文件
    # file_list = [
    #     "file1.xlsx",
    #     "file2.csv",
    #     "file3.txt"
    # ]
    # results = process_multiple_files(file_list, col1_name="ID", col2_name="Value")

    print("\n处理完成！")


if __name__ == "__main__":
    main()