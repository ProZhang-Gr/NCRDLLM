import pandas as pd
import re


def filter_seven_digit_circ(input_file, output_file=None):
    """
    筛选第一列中circ_后有七位数字的行

    参数:
    input_file: 输入的xlsx文件路径
    output_file: 输出文件路径（可选）
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)

        # 获取第一列的列名
        first_column = df.columns[0]

        # 定义正则表达式：匹配circ_后面恰好7位数字的模式
        pattern = r'circ_\d{7}$'

        # 筛选符合条件的行
        mask = df[first_column].astype(str).str.contains(pattern, regex=True, na=False)
        filtered_df = df[mask]

        print(f"原始数据: {len(df)} 行")
        print(f"筛选后: {len(filtered_df)} 行")
        print(f"筛选条件: circ_后恰好7位数字")

        # 显示筛选结果示例
        if len(filtered_df) > 0:
            print(f"\n筛选后的数据示例:")
            print(filtered_df.head())

            # 如果指定了输出文件，保存结果
            if output_file:
                filtered_df.to_excel(output_file, index=False)
                print(f"\n结果已保存到: {output_file}")
        else:
            print("\n未找到符合条件的数据")

        return filtered_df

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\ncRNA-Drug\所有工具\复制文件.xlsx"  # 替换为你的文件路径
    output_file = "filtered_circ_data.xlsx"  # 输出文件路径

    # 执行筛选
    result = filter_seven_digit_circ(input_file, output_file)