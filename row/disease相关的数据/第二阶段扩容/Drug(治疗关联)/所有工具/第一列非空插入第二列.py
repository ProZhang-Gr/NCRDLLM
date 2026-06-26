import pandas as pd


def insert_col1_to_col2(input_file, output_file=None):
    """
    将第一列的非空内容插入到第二列中

    参数:
    input_file: 输入的Excel文件路径
    output_file: 输出文件路径（可选）
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(input_file)

        # 获取列名
        col1_name = df.columns[0]  # 第一列
        col2_name = df.columns[1]  # 第二列

        print(f"处理列: {col1_name} -> {col2_name}")

        # 创建副本以避免修改原数据
        df_modified = df.copy()

        # 找到第一列非空的行
        mask = df_modified[col1_name].notna() & (df_modified[col1_name] != '')

        # 将第一列的非空值插入到第二列
        df_modified.loc[mask, col2_name] = df_modified.loc[mask, col1_name]

        # 统计信息
        modified_count = mask.sum()
        total_count = len(df_modified)

        print(f"总行数: {total_count}")
        print(f"第一列非空行数: {modified_count}")
        print(f"已将第一列的 {modified_count} 个非空值插入到第二列")

        # 显示修改前后对比示例
        print("\n修改示例:")
        comparison_df = pd.DataFrame({
            '原第一列': df[col1_name],
            '原第二列': df[col2_name],
            '新第二列': df_modified[col2_name]
        })
        print(comparison_df[mask].head())

        # 保存结果
        if output_file:
            df_modified.to_excel(output_file, index=False)
            print(f"\n结果已保存到: {output_file}")
        else:
            # 如果没有指定输出文件，覆盖原文件
            df_modified.to_excel(input_file, index=False)
            print(f"\n原文件已更新: {input_file}")

        return df_modified

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\Drug(治疗关联)\所有工具\复制文件.xlsx"  # 替换为你的文件路径
    output_file = "modified_file.xlsx"  # 输出文件路径（可选）

    # 执行处理
    result = insert_col1_to_col2(input_file, output_file)