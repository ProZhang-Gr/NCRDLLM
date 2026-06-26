import pandas as pd


def find_missing_cids(drugs_file, other_files):
    """
    找到drugs.xlsx文件中没有出现在其他三个文件并集中的CID

    参数:
    drugs_file: drugs.xlsx文件路径
    other_files: 其他三个文件路径的列表

    返回:
    missing_cids: 不在其他文件并集中的CID列表
    """

    # 读取drugs.xlsx文件的第一列CID
    print(f"正在读取 {drugs_file}...")
    drugs_df = pd.read_excel(drugs_file)
    drugs_cids = set(drugs_df.iloc[:, 0].dropna())  # 第一列，去除NaN值
    print(f"drugs.xlsx中共有 {len(drugs_cids)} 个CID")

    # 读取其他三个文件的第二列CID，并合并为一个集合
    other_cids = set()
    for i, file_path in enumerate(other_files, 1):
        print(f"正在读取文件 {i}: {file_path}...")
        df = pd.read_excel(file_path)
        file_cids = set(df.iloc[:, 1].dropna())  # 第二列，去除NaN值
        other_cids.update(file_cids)
        print(f"文件 {i} 中有 {len(file_cids)} 个CID")

    print(f"其他三个文件的CID并集共有 {len(other_cids)} 个唯一CID")

    # 找到在drugs.xlsx中但不在其他文件并集中的CID
    missing_cids = drugs_cids - other_cids
    print(f"找到 {len(missing_cids)} 个在drugs.xlsx中但不在其他文件中的CID")

    return list(missing_cids)


def save_results(missing_cids, output_file="missing_cids.xlsx"):
    """
    将结果保存到Excel文件

    参数:
    missing_cids: 缺失的CID列表
    output_file: 输出文件名
    """
    if missing_cids:
        result_df = pd.DataFrame({"Missing_CID": missing_cids})
        result_df.to_excel(output_file, index=False)
        print(f"结果已保存到 {output_file}")
    else:
        print("没有找到缺失的CID")


# 使用示例
if __name__ == "__main__":
    # 指定文件路径
    drugs_file = r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\ChemBERTa_Drug_Features_768D.xlsx"  # drugs文件路径
    other_files = [
        r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\circRNA-drug.xlsx",
     r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\lncRNA-drug.xlsx",
     r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\miRNA-drug.xlsx"  # 第三个其他文件
    ]
    try:
        # 执行查找
        missing_cids = find_missing_cids(drugs_file, other_files)

        # 显示结果
        if missing_cids:
            print("\n缺失的CID:")
            for cid in sorted(missing_cids):
                print(cid)

            # 保存结果
            save_results(missing_cids)
        else:
            print("所有drugs.xlsx中的CID都存在于其他文件中")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 如果你只想要一个简化版本的函数：
def simple_find_missing_cids(drugs_file, file1, file2, file3):
    """简化版本：直接指定四个文件路径"""
    drugs_df = pd.read_excel(drugs_file)
    drugs_cids = set(drugs_df.iloc[:, 0].dropna())

    other_cids = set()
    for file in [file1, file2, file3]:
        df = pd.read_excel(file)
        other_cids.update(df.iloc[:, 1].dropna())

    missing_cids = drugs_cids - other_cids
    return list(missing_cids)


missing = simple_find_missing_cids(r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\ChemBERTa_Drug_Features_768D.xlsx",
                                   r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\circRNA-drug.xlsx",
                                   r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\lncRNA-drug.xlsx",
                                   r"D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\miRNA-drug.xlsx")