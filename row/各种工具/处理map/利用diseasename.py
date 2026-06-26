import pandas as pd
import numpy as np


def process_disease_files(xlsx_file_path, csv_file_path, output_csv_path):
    """
    处理疾病文件，从CSV中删除在XLSX中出现的疾病名称行

    参数:
    xlsx_file_path: xlsx文件路径
    csv_file_path: csv文件路径  
    output_csv_path: 输出的csv文件路径
    """

    # 读取xlsx文件
    try:
        xlsx_df = pd.read_excel(xlsx_file_path)
        print(f"XLSX文件读取成功，共{len(xlsx_df)}行")
    except Exception as e:
        print(f"读取XLSX文件出错: {e}")
        return

    # 读取csv文件
    try:
        csv_df = pd.read_csv(csv_file_path)
        print(f"CSV文件读取成功，共{len(csv_df)}行")
    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return

    # 检查列名是否存在
    if 'Disease_Name' not in xlsx_df.columns:
        print("XLSX文件中未找到Disease_Name列")
        return

    if 'Disease_Name' not in csv_df.columns:
        print("CSV文件中未找到Disease_Name列")
        return

    # 获取xlsx中的所有疾病名称，转换为小写进行比较
    xlsx_diseases = set()
    for disease in xlsx_df['Disease_Name'].dropna():
        # 处理可能的字符串，去除首尾空格并转换为小写
        disease_clean = str(disease).strip().lower()
        if disease_clean:  # 确保不是空字符串
            xlsx_diseases.add(disease_clean)

    print(f"XLSX中共有{len(xlsx_diseases)}个唯一疾病名称")

    # 标记要保留的行
    rows_to_keep = []
    deleted_count = 0

    for index, row in csv_df.iterrows():
        disease_name = str(row['Disease_Name']).strip().lower()

        # 如果疾病名称不在xlsx的疾病列表中，则保留这一行
        if disease_name not in xlsx_diseases:
            rows_to_keep.append(index)
        else:
            deleted_count += 1
            print(f"删除行: {row['Disease_Name']}")

    # 创建新的DataFrame，只包含要保留的行
    filtered_csv_df = csv_df.loc[rows_to_keep]

    # 保存处理后的CSV文件
    try:
        filtered_csv_df.to_csv(output_csv_path, index=False)
        print(f"\n处理完成！")
        print(f"原CSV文件行数: {len(csv_df)}")
        print(f"删除行数: {deleted_count}")
        print(f"保留行数: {len(filtered_csv_df)}")
        print(f"结果已保存到: {output_csv_path}")
    except Exception as e:
        print(f"保存文件出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    xlsx_file = r"D:\Desktop\CDLLM\ing\row\各种工具\处理map\drugmap\从DO数据库获取的name2OID.xlsx"  # 替换为你的xlsx文件路径
    csv_file = r"D:\Desktop\CDLLM\ing\row\各种工具\处理map\drugmap\从CTD获取的name_diseases.csv"  # 替换为你的csv文件路径
    output_file = "filtered_diseases.csv"  # 输出文件路径

    process_disease_files(xlsx_file, csv_file, output_file)

    # 可选：显示处理后文件的前几行预览
    try:
        result_df = pd.read_csv(output_file)
        print(f"\n处理后文件预览（前5行）:")
        print(result_df.head())
    except:
        print("无法预览结果文件")