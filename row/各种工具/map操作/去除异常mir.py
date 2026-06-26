import pandas as pd
import re


def format_mirbase_id(mirbase_id):
    """
    格式化miRBase_ID
    返回格式化后的ID，如果需要删除则返回None
    """
    if pd.isna(mirbase_id):
        return None

    mirbase_str = str(mirbase_id).strip()

    # 规则1: 如果包含*号，则删除该行
    if '*' in mirbase_str:
        return None

    # 规则2: 如果是hsa-miR-开头，且有第三个"-"，则去除第三个"-"及之后的内容
    if mirbase_str.startswith('hsa-miR-'):
        # 按"-"分割
        parts = mirbase_str.split('-')

        # 如果有4个或更多部分 (hsa, miR, 124a, 1, ...)，则只保留前3部分
        if len(parts) >= 4:
            formatted_id = '-'.join(parts[:3])  # hsa-miR-124a
            return formatted_id
        else:
            # 如果只有3部分或更少，保持原样
            return mirbase_str

    # 规则3: 其他情况（如hsa-let-7a）保持不变
    return mirbase_str


def process_mirbase_file(input_file, output_file):
    """
    处理miRBase文件，格式化miRBase_ID列
    """
    try:
        # 读取文件
        print("正在读取文件...")
        df = pd.read_excel(input_file)

        # 检查miRBase_ID列是否存在
        if 'miRBase_ID' not in df.columns:
            raise ValueError("文件中未找到'miRBase_ID'列")

        print(f"原始文件包含 {len(df)} 行数据")

        # 显示原始数据示例
        print("\n原始数据示例:")
        unique_ids = df['miRBase_ID'].unique()
        for i, mirbase_id in enumerate(unique_ids[:10]):
            print(f"  {i + 1}: {mirbase_id}")

        # 处理每一行
        print("\n正在处理miRBase_ID...")
        processed_data = []
        kept_count = 0
        removed_count = 0
        format_changes = []
        removed_ids = []

        for idx, row in df.iterrows():
            original_id = row['miRBase_ID']
            formatted_id = format_mirbase_id(original_id)

            if formatted_id is not None:
                # 保留这一行
                new_row = row.copy()
                new_row['miRBase_ID'] = formatted_id
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
                removed_ids.append(original_id)
                print(f"  删除: '{original_id}' (含有*号)")

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
            for original, formatted in format_changes:
                print(f"  '{original}' -> '{formatted}'")

        if removed_ids:
            print(f"\n删除的ID详情:")
            unique_removed = list(set(removed_ids))
            for removed_id in unique_removed:
                count = removed_ids.count(removed_id)
                print(f"  '{removed_id}' (删除了 {count} 行)")

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

        if 'miRBase_ID' not in df.columns:
            raise ValueError("文件中未找到'miRBase_ID'列")

        print("=== 处理预览 ===")

        # 获取唯一的miRBase_ID进行预览
        unique_ids = df['miRBase_ID'].unique()

        print("唯一miRBase_ID处理结果:")
        keep_types = []
        remove_types = []
        format_types = []

        for mirbase_id in unique_ids:
            formatted_id = format_mirbase_id(mirbase_id)

            if formatted_id is not None:
                if str(mirbase_id) != str(formatted_id):
                    status = f"✓ 格式化: '{mirbase_id}' -> '{formatted_id}'"
                    format_types.append((mirbase_id, formatted_id))
                else:
                    status = f"✓ 保持: '{mirbase_id}'"
                    keep_types.append(mirbase_id)
            else:
                status = f"✗ 删除: '{mirbase_id}'"
                remove_types.append(mirbase_id)

            print(f"  {status}")

        # 统计每种类型的行数
        keep_count = 0
        remove_count = 0
        format_count = 0

        for idx, row in df.iterrows():
            mirbase_id = row['miRBase_ID']
            formatted_id = format_mirbase_id(mirbase_id)

            if formatted_id is not None:
                keep_count += 1
                if str(mirbase_id) != str(formatted_id):
                    format_count += 1
            else:
                remove_count += 1

        print(f"\n=== 全文件统计预览 ===")
        print(f"总行数: {len(df)}")
        print(f"将保留: {keep_count} 行")
        print(f"将删除: {remove_count} 行")
        print(f"将格式化: {format_count} 行")

        print(f"\n唯一ID类型:")
        print(f"  保持不变: {len(keep_types)} 种")
        print(f"  格式化: {len(format_types)} 种")
        print(f"  删除: {len(remove_types)} 种")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


def validate_format_rules():
    """
    验证格式化规则是否正确
    """
    test_cases = [
        "hsa-let-7a",  # 正常格式，应该保持不变
        "hsa-let-7d*",  # 含有*号，应该删除
        "hsa-miR-124a-1",  # 有第三个-，应该格式化为hsa-miR-124a
        "hsa-miR-124a-2",  # 有第三个-，应该格式化为hsa-miR-124a
        "hsa-miR-124b",  # 没有第三个-，应该保持不变
        "hsa-miR-125",  # 没有第三个-，应该保持不变
        "hsa-miR-200a-3p",  # 有第三个-，应该格式化为hsa-miR-200a
        "hsa-let-7b*",  # 含有*号，应该删除
    ]

    print("=== 格式化规则验证 ===")
    for test_case in test_cases:
        result = format_mirbase_id(test_case)
        if result is not None:
            if result != test_case:
                print(f"'{test_case}' -> '{result}' (格式化)")
            else:
                print(f"'{test_case}' -> 保持不变")
        else:
            print(f"'{test_case}' -> 删除")


def analyze_mirbase_patterns(input_file):
    """
    分析miRBase_ID的模式，帮助理解数据结构
    """
    try:
        df = pd.read_excel(input_file)

        if 'miRBase_ID' not in df.columns:
            return

        print("=== miRBase_ID模式分析 ===")

        unique_ids = df['miRBase_ID'].unique()

        patterns = {
            'has_star': [],  # 含有*的
            'has_third_dash': [],  # 有第三个-的
            'normal': []  # 正常的
        }

        for mirbase_id in unique_ids:
            if pd.notna(mirbase_id):
                id_str = str(mirbase_id).strip()

                if '*' in id_str:
                    patterns['has_star'].append(id_str)
                elif id_str.startswith('hsa-miR-') and len(id_str.split('-')) >= 4:
                    patterns['has_third_dash'].append(id_str)
                else:
                    patterns['normal'].append(id_str)

        print(f"含有*号的ID ({len(patterns['has_star'])}个):")
        for id_str in patterns['has_star'][:5]:
            print(f"  {id_str}")
        if len(patterns['has_star']) > 5:
            print(f"  ... 还有 {len(patterns['has_star']) - 5} 个")

        print(f"\n有第三个-的ID ({len(patterns['has_third_dash'])}个):")
        for id_str in patterns['has_third_dash'][:5]:
            print(f"  {id_str}")
        if len(patterns['has_third_dash']) > 5:
            print(f"  ... 还有 {len(patterns['has_third_dash']) - 5} 个")

        print(f"\n正常格式的ID ({len(patterns['normal'])}个):")
        for id_str in patterns['normal'][:5]:
            print(f"  {id_str}")
        if len(patterns['normal']) > 5:
            print(f"  ... 还有 {len(patterns['normal']) - 5} 个")

    except Exception as e:
        print(f"模式分析过程中出现错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    input_file = r"D:\Desktop\CDLLM\ing\row\各种工具\map操作\原始数据\miR2Disease收集的信息\第二部处理.xlsx"  # 输入文件路径
    output_file = "formatted_mirbase.xlsx"  # 输出文件路径

    try:
        # 验证格式化规则
        validate_format_rules()

        print("\n" + "=" * 50)

        # 分析miRBase_ID模式
        analyze_mirbase_patterns(input_file)

        print("\n" + "=" * 50)

        # 预览处理结果
        preview_processing(input_file)

        print("\n" + "=" * 50)

        # 执行实际处理
        result_df = process_mirbase_file(input_file, output_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 简化版本函数
def simple_format_mirbase(input_file, output_file):
    """
    简化版格式化函数
    """
    df = pd.read_excel(input_file)

    # 应用格式化函数
    df['formatted_id'] = df['miRBase_ID'].apply(format_mirbase_id)

    # 只保留格式化成功的行
    result_df = df[df['formatted_id'].notna()].copy()
    result_df['miRBase_ID'] = result_df['formatted_id']
    result_df = result_df.drop('formatted_id', axis=1)

    # 保存结果
    result_df.to_excel(output_file, index=False)

    return result_df