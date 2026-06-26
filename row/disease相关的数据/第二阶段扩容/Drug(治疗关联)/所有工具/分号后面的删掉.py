import pandas as pd
import re


def clean_semicolon_content(file_path, output_path=None):
    """
    读取Excel文件，删除分号及其后面的内容

    Parameters:
    file_path (str): 输入Excel文件路径
    output_path (str, optional): 输出文件路径，如果不提供则覆盖原文件
    """

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 处理每一列的每个单元格
    for column in df.columns:
        if df[column].dtype == 'object':  # 只处理文本列
            # 使用正则表达式删除分号及其后面的所有内容
            df[column] = df[column].astype(str).str.replace(r';.*$', '', regex=True)
            # 去除可能产生的前后空格
            df[column] = df[column].str.strip()

    # 保存处理后的数据
    if output_path is None:
        output_path = file_path

    df.to_excel(output_path, index=False)
    print(f"处理完成！文件已保存至: {output_path}")

    return df


def clean_specific_column(file_path, column_name='Disease name', output_path=None):
    """
    只处理指定列的分号内容

    Parameters:
    file_path (str): 输入Excel文件路径
    column_name (str): 要处理的列名
    output_path (str, optional): 输出文件路径
    """

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 检查列是否存在
    if column_name not in df.columns:
        print(f"错误：列 '{column_name}' 不存在")
        print(f"可用列名: {list(df.columns)}")
        return None

    # 处理指定列
    df[column_name] = df[column_name].astype(str).str.replace(r';.*$', '', regex=True)
    df[column_name] = df[column_name].str.strip()

    # 保存处理后的数据
    if output_path is None:
        output_path = file_path.replace('.xlsx', '_cleaned.xlsx')

    df.to_excel(output_path, index=False)
    print(f"处理完成！文件已保存至: {output_path}")

    return df


# 使用示例
if __name__ == "__main__":
    # 方法1: 处理所有文本列
    # df = clean_semicolon_content('your_file.xlsx', 'output_file.xlsx')


    df = clean_specific_column(r'D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\Drug(治疗关联)\所有工具\复制文件.xlsx', 'Disease name', 'cleaned_file.xlsx')

    # 示例数据处理
    sample_data = {
        'CID': [71158, 41774, 41774, 41774, 41774],
        'Disease name': [
            'Alcohol Dependence',
            'Diabetes Mellitus, Noninsulin-Dependent; Niddm',
            'Maturity-Onset Diabetes Of The Young, Type 1; Mody1',
            'Maturity-Onset Diabetes Of The Young, Type 2; Mody2',
            'Maturity-Onset Diabetes Of The Young, Type 3; Mody3'
        ]
    }

    # 创建示例DataFrame并处理
    df = pd.DataFrame(df)
    print("处理前:")
    print(df)

    # 处理Disease name列
    df['Disease name'] = df['Disease name'].str.replace(r';.*$', '', regex=True)
    df['Disease name'] = df['Disease name'].str.strip()

    print("\n处理后:")
    print(df)