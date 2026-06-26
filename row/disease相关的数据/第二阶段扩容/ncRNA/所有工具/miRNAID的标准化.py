import pandas as pd
import re
import os


def format_mirna_id(mirna_id):
    """
    格式化miRNA ID
    规则：
    1. 只保留前两个-之前的内容
    2. 删除*号
    3. mir改为miR
    4. let保持小写
    """

    mirna_id = str(mirna_id).strip()

    if pd.isna(mirna_id) or mirna_id == '' or mirna_id == 'nan':
        return mirna_id

    # 删除*号
    formatted_id = mirna_id.replace('*', '')

    # 只保留前两个-的内容
    parts = formatted_id.split('-')
    if len(parts) > 2:
        formatted_id = '-'.join(parts[:3])  # hsa-mir/miR-数字

    # mir改为miR，但let保持小写
    if 'mir-' in formatted_id.lower() and 'let-7' not in formatted_id.lower():
        formatted_id = re.sub(r'mir-', 'miR-', formatted_id, flags=re.IGNORECASE)

    return formatted_id


def process_file():
    # 直接在这里修改文件路径
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\第一阶段所接触的数据\miR2Disease收集的信息\miRNA.xlsx"  # 修改这里的路径

    # 读取文件
    df = pd.read_excel(input_file)

    # 格式化miRBase_ID列
    df['miRBase_ID'] = df['miRBase_ID'].apply(format_mirna_id)

    # 生成输出文件名
    input_dir = os.path.dirname(input_file)
    input_name = os.path.basename(input_file)
    name_without_ext = os.path.splitext(input_name)[0]
    output_file = os.path.join(input_dir, f"{name_without_ext}_formatted.xlsx")

    # 保存文件
    df.to_excel(output_file, index=False)

    print(f"处理完成，文件保存为: {output_file}")
    print("前10行结果:")
    print(df[['miRBase_ID', 'DOID']].head(10))


if __name__ == "__main__":
    process_file()