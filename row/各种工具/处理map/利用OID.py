import pandas as pd
import re


def clean_doid(doid_str):
    """
    清理和标准化DOID字符串
    移除前缀如"DO:"，统一大小写
    """
    if pd.isna(doid_str):
        return ""

    # 转换为字符串并去除空格
    doid_str = str(doid_str).strip()

    # 移除"DO:"前缀（如果存在）
    doid_str = re.sub(r'^DO:', '', doid_str, flags=re.IGNORECASE)

    # 转换为大写以便比较
    return doid_str.upper()


def extract_doids_from_cell(cell_value):
    """
    从单元格中提取所有DOID
    处理用"|"分隔的多个ID，只提取DOID部分
    """
    if pd.isna(cell_value):
        return []

    cell_str = str(cell_value).strip()

    # 按"|"分割
    parts = cell_str.split('|')

    doids = []
    for part in parts:
        part = part.strip()
        # 只处理包含"DOID:"的部分
        if 'DOID:' in part.upper():
            cleaned = clean_doid(part)
            if cleaned:
                doids.append(cleaned)

    return doids


def process_disease_files(xlsx_file_path, csv_file_path, output_csv_path):
    """
    处理疾病文件，删除csv中在xlsx中出现的Disease_Doid行
    """
    try:
        # 读取xlsx文件
        print("正在读取xlsx文件...")
        xlsx_df = pd.read_excel(xlsx_file_path)

        # 读取csv文件
        print("正在读取csv文件...")
        csv_df = pd.read_csv(csv_file_path)

        # 检查列名是否存在
        if 'Disease_Doid' not in xlsx_df.columns:
            raise ValueError("xlsx文件中未找到'Disease_Doid'列")
        if 'Disease_Doid' not in csv_df.columns:
            raise ValueError("csv文件中未找到'Disease_Doid'列")

        # 提取xlsx文件中的所有Disease_Doid（清理后）
        print("正在处理xlsx文件中的Disease_Doid...")
        xlsx_doids = set()
        for doid in xlsx_df['Disease_Doid']:
            cleaned = clean_doid(doid)
            if cleaned:
                xlsx_doids.add(cleaned)

        print(f"xlsx文件中找到 {len(xlsx_doids)} 个唯一的Disease_Doid")

        # 标记要删除的行
        print("正在检查csv文件中的Disease_Doid...")
        rows_to_keep = []
        deleted_count = 0

        for idx, row in csv_df.iterrows():
            cell_doids = extract_doids_from_cell(row['Disease_Doid'])

            # 检查是否有任何DOID在xlsx中出现
            should_delete = False
            for doid in cell_doids:
                if doid in xlsx_doids:
                    should_delete = True
                    break

            if should_delete:
                deleted_count += 1
                print(f"删除行 {idx}: {row['Disease_Doid']}")
            else:
                rows_to_keep.append(idx)

        # 创建新的DataFrame，只包含要保留的行
        filtered_df = csv_df.iloc[rows_to_keep].copy()

        # 保存结果
        print(f"正在保存结果到 {output_csv_path}...")
        filtered_df.to_csv(output_csv_path, index=False)

        print(f"处理完成！")
        print(f"原始csv文件行数: {len(csv_df)}")
        print(f"删除的行数: {deleted_count}")
        print(f"剩余行数: {len(filtered_df)}")
        print(f"结果已保存到: {output_csv_path}")

        return filtered_df

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    xlsx_file = r"D:\Desktop\CDLLM\ing\row\各种工具\处理map\drugmap\从DO数据库获取的name2OID.xlsx"  # 替换为你的xlsx文件路径
    csv_file = r"D:\Desktop\CDLLM\ing\row\各种工具\处理map\filtered_diseases.csv"  # 替换为你的csv文件路径
    output_file = "filtered_output1.csv"  # 输出文件路径

    # 执行处理
    try:
        result_df = process_disease_files(xlsx_file, csv_file, output_file)
        print("任务成功完成！")
    except Exception as e:
        print(f"执行失败: {str(e)}")


# 如果你想要查看处理详情，可以使用以下调试函数
def debug_processing(xlsx_file_path, csv_file_path):
    """
    调试函数，显示详细的处理过程
    """
    xlsx_df = pd.read_excel(xlsx_file_path)
    csv_df = pd.read_csv(csv_file_path)

    print("=== XLSX文件中的Disease_Doid示例 ===")
    for i, doid in enumerate(xlsx_df['Disease_Doid'].head(10)):
        print(f"{i + 1}: 原始='{doid}' -> 清理后='{clean_doid(doid)}'")

    print("\n=== CSV文件中的Disease_Doid示例 ===")
    for i, doid in enumerate(csv_df['Disease_Doid'].head(10)):
        extracted = extract_doids_from_cell(doid)
        print(f"{i + 1}: 原始='{doid}' -> 提取的DOIDs={extracted}")

# 使用调试函数查看处理过程
# debug_processing("your_xlsx_file.xlsx", "your_csv_file.csv")