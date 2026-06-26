import pandas as pd
import re


def format_circbase_id(circbase_id):
    """
    格式化circBASE_ID
    返回格式化后的ID，如果不符合要求则返回None
    """
    if pd.isna(circbase_id):
        return None

    circbase_str = str(circbase_id).strip()

    # 检查是否以hsa_circ_开头
    if not circbase_str.startswith('hsa_circ_'):
        return None

    # 移除hsa_circ_前缀，获取后缀部分
    suffix = circbase_str[9:]  # 'hsa_circ_' 长度为9

    # 情况1: 纯7位数字 (如hsa_circ_0000003)
    if re.match(r'^\d{7}$', suffix):
        return circbase_str  # 保持原样

    # 情况2: 数字+下划线+英文 (如hsa_circ_0032131_CBC1)
    match = re.match(r'^(\d+)_[A-Za-z]+.*$', suffix)
    if match:
        number_part = match.group(1)
        return f'hsa_circ_{number_part}'

    # 其他情况都不保留
    return None


def process_circbase_file(input_file, output_file):
    """
    处理circBASE文件，格式化circBASE_ID列
    """
    try:
        # 读取文件
        print("正在读取文件...")
        df = pd.read_excel(input_file)

        # 检查circBASE_ID列是否存在
        if 'circBASE_ID' not in df.columns:
            raise ValueError("文件中未找到'circBASE_ID'列")

        print(f"原始文件包含 {len(df)} 行数据")

        # 显示原始数据示例
        print("\n原始数据示例:")
        for i, circbase_id in enumerate(df['circBASE_ID'].head(10)):
            print(f"  {i + 1}: {circbase_id}")

        # 处理每一行
        print("\n正在处理circBASE_ID...")
        processed_data = []
        kept_count = 0
        removed_count = 0
        format_changes = []

        for idx, row in df.iterrows():
            original_id = row['circBASE_ID']
            formatted_id = format_circbase_id(original_id)

            if formatted_id is not None:
                # 保留这一行
                new_row = row.copy()
                new_row['circBASE_ID'] = formatted_id
                processed_data.append(new_row)
                kept_count += 1

                # 记录格式变化
                if str(original_id) != str(formatted_id):
                    format_changes.append((original_id, formatted_id))
                    print(f"  格式化: '{original_id}' -> '{formatted_id}'")
                else:
                    print(f"  保持: '{original_id}'")
            else:
                # 删除这一行
                removed_count += 1
                print(f"  删除: '{original_id}'")

        # 创建新的DataFrame
        if processed_data:
            result_df = pd.DataFrame(processed_data)
        else:
            # 如果没有数据保留，创建空的DataFrame但保持列结构
            result_df = df.iloc[0:0].copy()

        # 保存结果
        print(f"\n正在保存结果到 {output_file}...")
        result_df.to_excel(output_file, index=False)

        # 输出统计信息
        print(f"\n=== 处理统计 ===")
        print(f"原始行数: {len(df)}")
        print(f"保留行数: {kept_count}")
        print(f"删除行数: {removed_count}")
        print(f"格式变化数量: {len(format_changes)}")

        if format_changes:
            print(f"\n格式变化详情:")
            for original, formatted in format_changes[:10]:  # 只显示前10个
                print(f"  '{original}' -> '{formatted}'")
            if len(format_changes) > 10:
                print(f"  ... 还有 {len(format_changes) - 10} 个格式变化")

        print(f"\n结果已保存到: {output_file}")

        return result_df

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


def preview_processing(input_file):
    """
    预览处理结果，不保存文件
    """
    try:
        df = pd.read_excel(input_file)

        if 'circBASE_ID' not in df.columns:
            raise ValueError("文件中未找到'circBASE_ID'列")

        print("=== 处理预览 ===")
        keep_count = 0
        remove_count = 0
        format_count = 0

        print("前20行处理结果:")
        for idx, circbase_id in enumerate(df['circBASE_ID'].head(20)):
            formatted_id = format_circbase_id(circbase_id)

            if formatted_id is not None:
                if str(circbase_id) != str(formatted_id):
                    status = f"✓ 格式化: '{circbase_id}' -> '{formatted_id}'"
                    format_count += 1
                else:
                    status = f"✓ 保持: '{circbase_id}'"
                keep_count += 1
            else:
                status = f"✗ 删除: '{circbase_id}'"
                remove_count += 1

            print(f"  行{idx + 1}: {status}")

        # 统计所有数据
        total_keep = 0
        total_remove = 0
        total_format = 0

        for circbase_id in df['circBASE_ID']:
            formatted_id = format_circbase_id(circbase_id)
            if formatted_id is not None:
                total_keep += 1
                if str(circbase_id) != str(formatted_id):
                    total_format += 1
            else:
                total_remove += 1

        print(f"\n=== 全文件统计预览 ===")
        print(f"总行数: {len(df)}")
        print(f"将保留: {total_keep} 行")
        print(f"将删除: {total_remove} 行")
        print(f"将格式化: {total_format} 行")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


def validate_format_rules():
    """
    验证格式化规则是否正确
    """
    test_cases = [
        "hsa_circ_0000002",  # 7位数字，应该保留
        "hsa_circ_0000003",  # 7位数字，应该保留
        "hsa_circ_0032131_CBC1",  # 数字+下划线+英文，应该格式化为hsa_circ_0032131
        "hsa_circ_000011",  # 6位数字，应该删除
        "hsa_circ_12345_ABC",  # 5位数字+下划线+英文，应该格式化为hsa_circ_12345
        "other_circ_0000003",  # 不是hsa_circ_开头，应该删除
        "hsa_circ_abc",  # 非数字，应该删除
        "",  # 空字符串，应该删除
    ]

    print("=== 格式化规则验证 ===")
    for test_case in test_cases:
        result = format_circbase_id(test_case)
        if result is not None:
            print(f"'{test_case}' -> '{result}' (保留)")
        else:
            print(f"'{test_case}' -> 删除")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    input_file = r"D:\Desktop\CDLLM\ing\row\各种工具\map操作\原始数据\LncRNADiseasev3.0收集的数据\circ第二步.xlsx"  # 输入文件路径
    output_file = "formatted_circbase.xlsx"  # 输出文件路径

    try:
        # 验证格式化规则
        validate_format_rules()

        print("\n" + "=" * 50)

        # 预览处理结果
        preview_processing(input_file)

        print("\n" + "=" * 50)

        # 执行实际处理
        result_df = process_circbase_file(input_file, output_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 简化版本函数
def simple_format_circbase(input_file, output_file):
    """
    简化版格式化函数
    """
    df = pd.read_excel(input_file)

    # 应用格式化函数
    df['formatted_id'] = df['circBASE_ID'].apply(format_circbase_id)

    # 只保留格式化成功的行
    result_df = df[df['formatted_id'].notna()].copy()
    result_df['circBASE_ID'] = result_df['formatted_id']
    result_df = result_df.drop('formatted_id', axis=1)

    # 保存结果
    result_df.to_excel(output_file, index=False)

    return result_df