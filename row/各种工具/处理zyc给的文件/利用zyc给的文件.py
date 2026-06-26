import pandas as pd
import re


def parse_txt_file(txt_file_path):
    """
    解析TXT文件，提取化合物信息
    返回: {化合物名称: {'cid': CID, 'sid': SID, 'ctd': CTD, 'line': 原始行}}
    """
    compounds_dict = {}

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        print(f"处理第 {line_num} 行: {line[:80]}...")

        try:
            # 找到最后一个包含ID信息的部分
            # 先找到SID的位置来分割名称部分和ID部分
            sid_match = re.search(r'SID:\s*(\d+)', line)
            if not sid_match:
                print(f"  警告: 第 {line_num} 行未找到SID，跳过")
                continue

            sid_pos = line.find('SID:')
            names_part = line[:sid_pos].strip()
            ids_part = line[sid_pos:].strip()

            # 解析ID部分
            sid = sid_match.group(1)

            cid_match = re.search(r'CID:\s*(\d+)', ids_part)
            cid = cid_match.group(1) if cid_match else None

            ctd_match = re.search(r'CTD:\s*([A-Z0-9]+)', ids_part)
            ctd = ctd_match.group(1) if ctd_match else None

            # 解析名称部分 - 按分号分割
            if names_part.endswith(';'):
                names_part = names_part[:-1]  # 移除末尾分号

            names = [name.strip() for name in names_part.split(';')]
            names = [name for name in names if name]  # 移除空字符串

            # 存储每个名称
            compound_info = {
                'cid': cid,
                'sid': sid,
                'ctd': ctd,
                'line': line,
                'all_names': names
            }

            for name in names:
                # 清理名称
                clean_name = name.strip()
                if clean_name:
                    compounds_dict[clean_name] = compound_info.copy()
                    print(f"  添加名称: '{clean_name}' -> CID: {cid}")

        except Exception as e:
            print(f"  错误: 第 {line_num} 行解析失败 - {str(e)}")
            continue

    return compounds_dict


def exact_match_chemicals(txt_file_path, xlsx_file_path, output_file_path=None):
    """
    精确匹配化合物名称并提取CID
    """
    print("=" * 80)
    print("开始处理化合物CID精确匹配")
    print("=" * 80)

    # 解析TXT文件
    print("\n1. 解析TXT文件...")
    compounds_dict = parse_txt_file(txt_file_path)
    print(f"成功解析 {len(compounds_dict)} 个化合物名称")

    # 显示解析的化合物名称示例
    print("\n解析的化合物名称示例:")
    for i, name in enumerate(list(compounds_dict.keys())[:5]):
        info = compounds_dict[name]
        print(f"  {i + 1}. '{name}' -> CID: {info['cid']}, SID: {info['sid']}")

    # 读取XLSX文件
    print(f"\n2. 读取XLSX文件...")
    try:
        df = pd.read_excel(xlsx_file_path)
        print(f"XLSX文件包含 {len(df)} 行数据")

        # 检查列名
        print(f"列名: {list(df.columns)}")

        if 'ChemicalName' not in df.columns:
            print("错误: XLSX文件中没有找到 'ChemicalName' 列")
            return None

    except Exception as e:
        print(f"读取XLSX文件时出错: {e}")
        return None

    # 创建结果列
    df['Matched_CID'] = ''
    df['Matched_SID'] = ''
    df['Matched_CTD'] = ''
    df['Match_Status'] = ''
    df['Matched_Source_Name'] = ''

    # 进行精确匹配
    print(f"\n3. 进行精确匹配...")
    exact_matches = 0
    no_matches = 0
    empty_names = 0

    for idx, row in df.iterrows():
        chemical_name = str(row['ChemicalName']).strip()

        # 处理空值或NaN
        if not chemical_name or chemical_name.lower() == 'nan':
            df.at[idx, 'Match_Status'] = 'Empty Name'
            empty_names += 1
            continue

        # 精确匹配
        if chemical_name in compounds_dict:
            compound_info = compounds_dict[chemical_name]
            df.at[idx, 'Matched_CID'] = compound_info['cid'] if compound_info['cid'] else 'No CID'
            df.at[idx, 'Matched_SID'] = compound_info['sid']
            df.at[idx, 'Matched_CTD'] = compound_info['ctd'] if compound_info['ctd'] else 'No CTD'
            df.at[idx, 'Match_Status'] = 'Exact Match'
            df.at[idx, 'Matched_Source_Name'] = chemical_name
            exact_matches += 1

            if (idx + 1) <= 10:  # 显示前10个匹配结果
                print(f"  匹配 {idx + 1}: '{chemical_name}' -> CID: {compound_info['cid']}")
        else:
            df.at[idx, 'Match_Status'] = 'No Exact Match'
            no_matches += 1

            if (idx + 1) <= 5:  # 显示前5个未匹配的
                print(f"  未匹配 {idx + 1}: '{chemical_name}'")

        # 显示进度
        if (idx + 1) % 500 == 0:
            print(f"  已处理 {idx + 1} / {len(df)} 行")

    # 统计结果
    total_rows = len(df)
    print(f"\n4. 匹配结果统计:")
    print("=" * 50)
    print(f"总行数: {total_rows}")
    print(f"精确匹配: {exact_matches}")
    print(f"未找到匹配: {no_matches}")
    print(f"空白化合物名: {empty_names}")
    print(f"精确匹配率: {(exact_matches / total_rows * 100):.2f}%")

    # 按匹配状态分组统计
    status_counts = df['Match_Status'].value_counts()
    print(f"\n详细统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # 保存结果
    if output_file_path:
        try:
            df.to_excel(output_file_path, index=False)
            print(f"\n5. 结果已保存到: {output_file_path}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    return df


def show_detailed_results(df, show_matched=True, show_unmatched=True, max_each=10):
    """
    显示详细的匹配结果
    """
    if show_matched:
        print(f"\n精确匹配结果 (前 {max_each} 个):")
        print("-" * 120)
        matched_df = df[df['Match_Status'] == 'Exact Match'].head(max_each)

        for idx, row in matched_df.iterrows():
            print(f"{idx + 1:3d}. {row['ChemicalName'][:50]:50s} -> "
                  f"CID: {str(row['Matched_CID'])[:12]:12s} "
                  f"SID: {str(row['Matched_SID'])[:12]:12s}")

    if show_unmatched:
        print(f"\n未匹配结果 (前 {max_each} 个):")
        print("-" * 120)
        unmatched_df = df[df['Match_Status'] == 'No Exact Match'].head(max_each)

        for idx, row in unmatched_df.iterrows():
            print(f"{idx + 1:3d}. {row['ChemicalName'][:70]:70s} -> 未找到精确匹配")


def find_similar_names(target_name, compounds_dict, max_suggestions=3):
    """
    为未匹配的化合物名称寻找相似的名称（用于调试）
    """
    suggestions = []
    target_lower = target_name.lower()

    for name in compounds_dict.keys():
        name_lower = name.lower()

        # 检查部分包含
        if target_lower in name_lower or name_lower in target_lower:
            suggestions.append(name)

        if len(suggestions) >= max_suggestions:
            break

    return suggestions


# 主函数
if __name__ == "__main__":
    # 文件路径设置
    txt_file_path = r"D:\Desktop\CDLLM\ing\row\各种工具\处理zyc给的文件\mapID_SID_CID_CTDID.txt"  # TXT文件路径
    xlsx_file_path = r"D:\Desktop\CDLLM\ing\row\各种工具\处理zyc给的文件\药物名称与MESHID.xlsx"  # XLSX文件路径
    output_file_path = "exact_matched_results.xlsx"  # 输出文件路径

    try:
        # 执行精确匹配
        result_df = exact_match_chemicals(txt_file_path, xlsx_file_path, output_file_path)

        if result_df is not None:
            # 显示详细结果
            show_detailed_results(result_df, show_matched=True, show_unmatched=True, max_each=10)

            # 为未匹配的项目提供建议（仅前5个）
            print(f"\n未匹配项目的相似名称建议:")
            print("-" * 80)
            unmatched_df = result_df[result_df['Match_Status'] == 'No Exact Match'].head(5)

            if len(unmatched_df) > 0:
                compounds_dict = parse_txt_file(txt_file_path)
                for idx, row in unmatched_df.iterrows():
                    suggestions = find_similar_names(row['ChemicalName'], compounds_dict)
                    if suggestions:
                        print(f"'{row['ChemicalName']}' 的相似名称:")
                        for suggestion in suggestions:
                            print(f"  -> {suggestion}")
                    else:
                        print(f"'{row['ChemicalName']}' 没有找到相似名称")

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请检查以下文件是否存在:")
        print(f"  TXT文件: {txt_file_path}")
        print(f"  XLSX文件: {xlsx_file_path}")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback

        traceback.print_exc()

# 使用说明
print("""
精确匹配程序使用说明:
====================

1. 文件准备:
   - TXT文件格式: 化合物名称1; 别名1; 别名2; ... SID: xxx CID: xxx CTD: xxx
   - XLSX文件: 包含 'ChemicalName' 列

2. 设置文件路径:
   - txt_file_path: TXT文件路径
   - xlsx_file_path: XLSX文件路径  
   - output_file_path: 输出Excel文件路径

3. 匹配规则:
   - 只进行100%精确匹配
   - 区分大小写
   - 不进行任何相似度匹配

4. 输出列:
   - Matched_CID: 匹配的CID
   - Matched_SID: 匹配的SID
   - Matched_CTD: 匹配的CTD
   - Match_Status: 匹配状态 (Exact Match/No Exact Match/Empty Name)
   - Matched_Source_Name: 匹配的源名称

5. 运行程序:
   python exact_chemical_matcher.py
""")