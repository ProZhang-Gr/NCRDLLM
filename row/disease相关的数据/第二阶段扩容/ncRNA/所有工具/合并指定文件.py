import pandas as pd
import os


def merge_xlsx_files():
    # 在这里添加所有需要合并的文件路径
    file_paths = [
        r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\lnc2CancerV3收集的数据\lncRNA.xlsx",
        r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\RNADiseaseV4.0收集的数据\lncRNA_Disease\lncRNA.xlsx",
        r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\zyc收集的数据\lncRNA.xlsx",
        r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\第一阶段所接触的数据\LncRNADiseasev3.0收集的数据\lncRNA.xlsx"

        # 添加更多文件路径...
    ]

    # 检查文件是否存在
    existing_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"找到文件: {file_path}")
        else:
            print(f"文件不存在: {file_path}")

    if not existing_files:
        print("没有找到任何有效文件！")
        return

    # 读取所有文件并合并
    all_dataframes = []

    for file_path in existing_files:
        try:
            df = pd.read_excel(file_path)
            print(f"读取 {os.path.basename(file_path)}: {len(df)} 行")
            all_dataframes.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

    if not all_dataframes:
        print("没有成功读取任何文件！")
        return

    # 合并所有数据
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"合并后总行数: {len(merged_df)}")

    # 去重（基于所有列）
    before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    after_dedup = len(merged_df)

    print(f"去重前: {before_dedup} 行")
    print(f"去重后: {after_dedup} 行")
    print(f"删除了 {before_dedup - after_dedup} 行重复数据")

    # 保存合并后的文件
    output_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\所有工具\result.xlsx"  # 修改输出路径
    merged_df.to_excel(output_file, index=False)

    print(f"合并完成！文件保存为: {output_file}")
    print("\n前10行预览:")
    print(merged_df.head(10))


if __name__ == "__main__":
    merge_xlsx_files()