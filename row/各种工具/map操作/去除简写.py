import pandas as pd
import re


def clean_disease_names(input_file, output_file=None):
    """
    清理Excel文件中Disease_Name列的括号内容

    参数:
    input_file: 输入的Excel文件路径
    output_file: 输出文件路径，如果为None则覆盖原文件
    """

    # 读取Excel文件
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取文件: {input_file}")
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 检查是否存在Disease_Name列
    if 'Disease_Name' not in df.columns:
        print("错误: 未找到'Disease_Name'列")
        print(f"可用列: {list(df.columns)}")
        return

    # 显示清理前的数据示例
    print("\n清理前的数据示例:")
    print(df[['miRBASE_ID', 'Disease_Name']].head(10))

    # 清理Disease_Name列，去除括号及其内容
    def remove_brackets(text):
        if pd.isna(text):
            return text
        # 使用正则表达式去除括号及其内容，并清理多余空格
        cleaned = re.sub(r'\s*\([^)]*\)\s*', '', str(text))
        # 清理多余的空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    # 应用清理函数
    df['Disease_Name'] = df['Disease_Name'].apply(remove_brackets)

    # 显示清理后的数据示例
    print("\n清理后的数据示例:")
    print(df[['miRBASE_ID', 'Disease_Name']].head(10))

    # 统计清理效果
    print(f"\n处理完成:")
    print(f"总共处理了 {len(df)} 行数据")

    # 保存结果
    if output_file is None:
        output_file = input_file

    try:
        df.to_excel(output_file, index=False)
        print(f"结果已保存到: {output_file}")

    except Exception as e:
        print(f"保存文件失败: {e}")
        return

    return df


# 使用示例
if __name__ == "__main__":
    # 替换为你的文件路径
    input_file = r"D:\Desktop\CDLLM\ing\row\各种工具\map操作\原始数据\miR2Disease收集的信息\新建 文本文档.xlsx"  # 输入文件路径
    output_file = "cleaned_file.xlsx"  # 输出文件路径，可选

    # 清理数据
    cleaned_df = clean_disease_names(input_file, output_file)

    # 如果想要查看更多清理结果
    if cleaned_df is not None:
        print("\n所有疾病名称(去重后):")
        unique_diseases = cleaned_df['Disease_Name'].unique()
        for disease in sorted(unique_diseases):
            print(f"- {disease}")


# 或者简化版本，直接处理
def quick_clean(input_file):
    """简化版本的快速清理"""
    df = pd.read_excel(input_file)
    df['Disease_Name'] = df['Disease_Name'].str.replace(r'\s*\([^)]*\)\s*', '', regex=True)
    df['Disease_Name'] = df['Disease_Name'].str.strip()
    df.to_excel(input_file, index=False)
    print("清理完成!")
    return df

# 使用简化版本:
# clean_df = quick_clean("your_file.xlsx")