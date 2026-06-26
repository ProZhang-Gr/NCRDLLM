import pandas as pd
import re


def parse_doid_obo_file(file_path):
    """解析OBO文件，提取DOID的新旧ID映射关系"""

    print(f"开始解析OBO文件: {file_path}")

    # 存储映射关系
    id_mappings = []

    # 当前处理的term信息
    current_term = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"文件总行数: {len(lines)}")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('!'):
                continue

            # 新的Term开始
            if line == '[Term]':
                # 处理上一个term（如果有alt_id）
                if current_term and 'alt_ids' in current_term:
                    process_term(current_term, id_mappings)

                # 重置当前term
                current_term = {'alt_ids': []}
                continue

            # 解析字段
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'id':
                    current_term['id'] = value
                elif key == 'name':
                    current_term['name'] = value
                elif key == 'alt_id':
                    current_term['alt_ids'].append(value)

            # 每处理1000行显示进度
            if line_num % 10000 == 0:
                print(f"已处理 {line_num} 行...")

        # 处理最后一个term
        if current_term and 'alt_ids' in current_term:
            process_term(current_term, id_mappings)

        print(f"解析完成，找到 {len(id_mappings)} 个有旧ID的条目")
        return id_mappings

    except Exception as e:
        print(f"解析文件时出错: {e}")
        return []


def process_term(term, mappings):
    """处理单个term，如果有alt_id则添加到映射列表"""
    if 'id' in term and term['alt_ids']:
        mapping = {
            'newID': term['id'],
            'name': term.get('name', ''),
            'oldIDs': term['alt_ids']
        }
        mappings.append(mapping)

        # 显示找到的映射
        old_ids_str = ', '.join(term['alt_ids'])
        print(f"找到映射: {term['id']} <- {old_ids_str} ({term.get('name', 'Unknown')})")


def create_mapping_excel(mappings, output_file):
    """创建Excel文件，包含新旧ID映射"""

    if not mappings:
        print("没有找到任何ID映射关系")
        return

    print(f"开始创建Excel文件...")

    # 准备数据 - 找出最大的旧ID数量来确定列数
    max_old_ids = max(len(mapping['oldIDs']) for mapping in mappings)
    print(f"最多的旧ID数量: {max_old_ids}")

    # 创建DataFrame的数据
    excel_data = []

    for mapping in mappings:
        row = {
            'newID': mapping['newID'],
            'name': mapping['name']
        }

        # 添加旧ID列
        for i, old_id in enumerate(mapping['oldIDs'], 1):
            row[f'oldID{i}'] = old_id

        # 填充空的旧ID列
        for i in range(len(mapping['oldIDs']) + 1, max_old_ids + 1):
            row[f'oldID{i}'] = ''

        excel_data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(excel_data)

    # 确保列的顺序
    columns = ['newID', 'name'] + [f'oldID{i}' for i in range(1, max_old_ids + 1)]
    df = df[columns]

    # 保存Excel文件
    df.to_excel(output_file, index=False, sheet_name='DOID_ID_Mapping')

    print(f"Excel文件已保存: {output_file}")
    print(f"总条目数: {len(df)}")
    print(f"列数: {len(df.columns)}")

    # 显示预览
    print("\n数据预览:")
    print(df.head().to_string(index=False))

    return df


def main():
    # 文件配置
    obo_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\Disease_DOID_Map\Doid新旧map\新建 文本文档.txt'  # 输入的OBO文件
    output_file = 'doid_id_mapping.xlsx'  # 输出的Excel文件

    print("=" * 60)
    print("Disease Ontology ID映射提取器")
    print("=" * 60)

    # 检查输入文件
    import os
    if not os.path.exists(obo_file):
        print(f"错误: 找不到OBO文件 {obo_file}")
        print("请将Disease Ontology的OBO文件重命名为此文件名")
        return

    # 解析OBO文件
    mappings = parse_doid_obo_file(obo_file)

    if not mappings:
        print("未找到任何新旧ID映射关系")
        return

    # 创建Excel文件
    df = create_mapping_excel(mappings, output_file)

    # 统计信息
    print(f"\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"输入文件: {obo_file}")
    print(f"输出文件: {output_file}")
    print(f"有旧ID的疾病条目: {len(mappings)}")

    # 显示一些统计
    total_old_ids = sum(len(mapping['oldIDs']) for mapping in mappings)
    print(f"旧ID总数: {total_old_ids}")
    print(f"平均每个疾病的旧ID数: {total_old_ids / len(mappings):.1f}")


if __name__ == "__main__":
    main()