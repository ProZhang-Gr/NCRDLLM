import pandas as pd
import re
import os


def extract_mesh_to_doid_mapping(obo_file):
    """从OBO文件中提取MESH ID到DOID的映射关系"""

    print(f"开始解析OBO文件: {obo_file}")

    mappings = []
    current_term = {}

    try:
        with open(obo_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"文件总行数: {len(lines)}")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('!'):
                continue

            # 新的Term开始
            if line == '[Term]':
                # 处理上一个term
                if current_term:
                    process_term_for_mesh_mapping(current_term, mappings)

                # 重置当前term
                current_term = {'mesh_ids': []}
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
                elif key == 'xref' and value.startswith('MESH:'):
                    # 提取MESH ID
                    mesh_id = value.replace('MESH:', '').strip()
                    current_term['mesh_ids'].append(mesh_id)
                elif key == 'is_obsolete' and value == 'true':
                    # 标记为已废弃的条目
                    current_term['obsolete'] = True

            # 显示进度
            if line_num % 10000 == 0:
                print(f"已处理 {line_num} 行，找到 {len(mappings)} 个MESH映射...")

        # 处理最后一个term
        if current_term:
            process_term_for_mesh_mapping(current_term, mappings)

        print(f"解析完成，找到 {len(mappings)} 个MESH-DOID映射")
        return mappings

    except Exception as e:
        print(f"解析文件时出错: {e}")
        return []


def process_term_for_mesh_mapping(term, mappings):
    """处理单个term，提取MESH-DOID映射"""

    # 跳过废弃的条目
    if term.get('obsolete', False):
        return

    # 只处理有DOID和MESH ID的条目
    if 'id' in term and term['mesh_ids']:
        doid = term['id']
        name = term.get('name', '')

        # 为每个MESH ID创建映射
        for mesh_id in term['mesh_ids']:
            mapping = {
                'MESH_ID': mesh_id,
                'DOID': doid,
                'Disease_Name': name
            }
            mappings.append(mapping)

            # 显示找到的映射
            if len(mappings) <= 10:  # 只显示前10个
                print(f"找到映射: {mesh_id} -> {doid} ({name})")


def create_mesh_doid_excel(mappings, output_file):
    """创建MESH-DOID映射的Excel文件"""

    if not mappings:
        print("没有找到任何MESH-DOID映射")
        return None

    print(f"\n创建Excel文件: {output_file}")

    # 转换为DataFrame
    df = pd.DataFrame(mappings)

    # 去重（某些疾病可能有多个MESH ID）
    original_count = len(df)
    df = df.drop_duplicates(subset=['MESH_ID', 'DOID'])
    print(f"去重前: {original_count}, 去重后: {len(df)}")

    # 按MESH ID排序
    df = df.sort_values('MESH_ID').reset_index(drop=True)

    # 保存Excel文件
    df.to_excel(output_file, index=False, sheet_name='MESH_to_DOID')

    print(f"Excel文件已保存: {output_file}")
    print(f"总映射数: {len(df)}")

    # 显示数据预览
    print(f"\n数据预览:")
    print(df.head(10).to_string(index=False))

    # 统计信息
    unique_mesh = df['MESH_ID'].nunique()
    unique_doid = df['DOID'].nunique()
    print(f"\n统计信息:")
    print(f"唯一MESH ID数: {unique_mesh}")
    print(f"唯一DOID数: {unique_doid}")
    print(f"平均每个MESH ID对应的DOID数: {len(df) / unique_mesh:.2f}")

    return df


def validate_with_existing_doids(mapping_df, existing_doid_file):
    """验证映射中的DOID是否在现有的疾病列表中"""

    if not os.path.exists(existing_doid_file):
        print(f"警告: 找不到现有DOID文件 {existing_doid_file}")
        return

    print(f"\n验证DOID与现有疾病列表...")

    try:
        # 读取现有的疾病-DOID文件
        existing_df = pd.read_excel(existing_doid_file)
        existing_doids = set(existing_df['DOID'].tolist())

        # 检查映射中的DOID
        mapping_doids = set(mapping_df['DOID'].tolist())

        # 找出交集
        common_doids = mapping_doids.intersection(existing_doids)
        only_in_mapping = mapping_doids - existing_doids
        only_in_existing = existing_doids - mapping_doids

        print(f"现有疾病列表DOID数: {len(existing_doids)}")
        print(f"映射中的DOID数: {len(mapping_doids)}")
        print(f"共同的DOID数: {len(common_doids)}")
        print(f"只在映射中的DOID数: {len(only_in_mapping)}")
        print(f"只在现有列表中的DOID数: {len(only_in_existing)}")
        print(f"覆盖率: {len(common_doids) / len(existing_doids) * 100:.1f}%")

    except Exception as e:
        print(f"验证时出错: {e}")


def main():
    # 文件配置
    obo_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\Disease_DOID_Map\Doid新旧map\新建 文本文档.txt'  # Disease Ontology OBO文件
    output_file = 'mesh_to_doid_mapping.xlsx'  # 输出的MESH-DOID映射文件
    existing_doid_file = 'disease_doid_list_updated.xlsx'  # 现有的疾病-DOID列表

    print("=" * 60)
    print("MESH ID到DOID映射提取器")
    print("=" * 60)

    # 检查OBO文件
    if not os.path.exists(obo_file):
        print(f"错误: 找不到OBO文件 {obo_file}")
        print("请将Disease Ontology的OBO文件重命名为此文件名")
        return

    # 提取MESH-DOID映射
    mappings = extract_mesh_to_doid_mapping(obo_file)

    if not mappings:
        print("未找到任何MESH-DOID映射")
        return

    # 创建Excel文件
    mapping_df = create_mesh_doid_excel(mappings, output_file)

    if mapping_df is not None:
        # 验证与现有DOID的匹配情况
        validate_with_existing_doids(mapping_df, existing_doid_file)

        print(f"\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"输出文件: {output_file}")
        print("现在你可以使用这个映射表将MESH ID转换为DOID")


if __name__ == "__main__":
    main()