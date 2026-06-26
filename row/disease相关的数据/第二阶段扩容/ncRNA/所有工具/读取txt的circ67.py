import pandas as pd
import os


def convert_circRNA_data(input_file, output_file=None):
    """
    专门用于转换circRNA数据的函数
    """
    if output_file is None:
        # 自动生成输出文件名
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.xlsx"

    try:
        # 尝试多种分隔符
        separators = ['\t', ' ', ',', ';']
        df = None

        for sep in separators:
            try:
                df = pd.read_csv(input_file, sep=sep, header=0)
                if df.shape[1] >= 2:  # 确保至少有两列
                    break
            except:
                continue

        if df is None or df.shape[1] < 2:
            print("无法正确解析文件，请检查文件格式")
            return None

        # 确保列名正确
        if df.shape[1] == 2:
            df.columns = ['circID', 'name']

        # 保存为Excel
        df.to_excel(output_file, index=False)

        print(f"✅ 转换完成!")
        print(f"📁 输入: {input_file}")
        print(f"📁 输出: {output_file}")
        print(f"📊 数据: {df.shape[0]} 行 × {df.shape[1]} 列")

        # 显示统计信息
        print(f"\n📋 数据概览:")
        print(f"- 总计 {len(df)} 个circRNA记录")
        print(f"- circID 示例: {df['circID'].iloc[0] if len(df) > 0 else 'N/A'}")
        print(f"- name 示例: {df['name'].iloc[0] if len(df) > 0 else 'N/A'}")

        return df

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None


# 直接使用
input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\map文件\新建 文本文档.txt"  # 替换为你的文件路径
df = convert_circRNA_data(input_file)