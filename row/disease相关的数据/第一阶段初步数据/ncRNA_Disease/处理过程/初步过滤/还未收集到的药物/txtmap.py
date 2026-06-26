import pandas as pd
import numpy as np
import re


def parse_obo_file(obo_file_path):
    """
    解析OBO格式的疾病本体文件
    返回疾病名称到DOID的映射字典
    """
    disease_mapping = {}

    try:
        with open(obo_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 按[Term]分割
        terms = content.split('[Term]')

        print(f"找到 {len(terms) - 1} 个疾病条目")

        for term_text in terms[1:]:  # 跳过第一个空的部分
            term_data = parse_single_term(term_text)
            if term_data:
                doid = term_data['id']
                name = term_data['name']
                synonyms = term_data['synonyms']

                # 添加主要名称的映射
                if name:
                    normalized_name = normalize_disease_name(name)
                    if normalized_name:
                        disease_mapping[normalized_name] = doid

                # 添加同义词的映射
                for synonym in synonyms:
                    normalized_synonym = normalize_disease_name(synonym)
                    if normalized_synonym:
                        disease_mapping[normalized_synonym] = doid

        print(f"成功创建 {len(disease_mapping)} 个疾病名称映射")
        return disease_mapping

    except Exception as e:
        print(f"解析OBO文件时出错: {str(e)}")
        raise


def parse_single_term(term_text):
    """
    解析单个[Term]条目
    """
    lines = term_text.strip().split('\n')

    term_data = {
        'id': None,
        'name': None,
        'synonyms': []
    }

    for line in lines:
        line = line.strip()

        # 提取ID
        if line.startswith('id: DOID:'):
            term_data['id'] = line.replace('id: ', '')

        # 提取名称
        elif line.startswith('name: '):
            term_data['name'] = line.replace('name: ', '')

        # 提取同义词
        elif line.startswith('synonym: '):
            synonym = extract_synonym_from_line(line)
            if synonym:
                term_data['synonyms'].append(synonym)

    # 只返回有完整ID和名称的条目
    if term_data['id'] and term_data['name']:
        return term_data

    return None


def extract_synonym_from_line(line):
    """
    从synonym行中提取同义词名称
    例: synonym: "hemangiosarcoma" EXACT [] -> hemangiosarcoma
    """
    # 使用正则表达式提取引号内的内容
    match = re.search(r'synonym:\s*"([^"]+)"', line)
    if match:
        return match.group(1)

    # 如果没有引号，尝试提取EXACT前的内容
    if 'EXACT' in line:
        parts = line.split('EXACT')[0]
        synonym = parts.replace('synonym:', '').strip()
        # 移除可能的引号
        synonym = synonym.strip('"\'')
        return synonym if synonym else None

    return None


def normalize_disease_name(name):
    """
    标准化疾病名称用于匹配
    """
    if not name:
        return ""

    # 转换为字符串，转小写，去除首尾空格
    name = str(name).lower().strip()

    # 去除多余的空格
    name = re.sub(r'\s+', ' ', name)

    # 去除标点符号的影响
    name = re.sub(r'[,\-\(\)]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def has_doid(doid_value):
    """
    检查DOID列是否已有值
    """
    if pd.isna(doid_value):
        return False

    doid_str = str(doid_value).strip()
    return bool(doid_str and doid_str.upper().startswith('DO:DOID:'))


def map_diseases_with_obo(xlsx_file, obo_file, output_file):
    """
    使用OBO文件映射疾病名称到DOID
    """
    try:
        # 读取xlsx文件
        print("正在读取xlsx文件...")
        df = pd.read_excel(xlsx_file)

        # 检查必要的列
        if 'Disease_Name' not in df.columns or 'DOID' not in df.columns:
            raise ValueError("xlsx文件中未找到'Disease_Name'或'DOID'列")

        print(f"xlsx文件包含 {len(df)} 行数据")

        # 解析OBO文件
        print("正在解析OBO文件...")
        disease_mapping = parse_obo_file(obo_file)

        # 显示映射字典示例
        print("\n疾病映射示例（前10个）:")
        for i, (disease, doid) in enumerate(list(disease_mapping.items())[:10]):
            print(f"  '{disease}' -> '{doid}'")

        # 统计需要映射的行
        need_mapping = df['DOID'].apply(lambda x: not has_doid(x))
        need_mapping_count = need_mapping.sum()
        already_mapped_count = len(df) - need_mapping_count

        print(f"\n=== 映射统计 ===")
        print(f"总行数: {len(df)}")
        print(f"已有DOID: {already_mapped_count}")
        print(f"需要映射: {need_mapping_count}")

        # 执行映射
        print("\n正在执行映射...")
        result_df = df.copy()

        mapped_count = 0
        unmapped_count = 0
        mapping_details = []
        unmapped_diseases = []

        for idx, row in result_df.iterrows():
            if not has_doid(row['DOID']):
                # 需要映射
                disease_name = row['Disease_Name']
                normalized_name = normalize_disease_name(disease_name)

                if normalized_name in disease_mapping:
                    # 找到映射
                    mapped_doid = disease_mapping[normalized_name]
                    result_df.at[idx, 'DOID'] = mapped_doid
                    mapped_count += 1
                    mapping_details.append((disease_name, mapped_doid))
                    print(f"  ✓ '{disease_name}' -> '{mapped_doid}'")
                else:
                    # 未找到映射
                    unmapped_count += 1
                    unmapped_diseases.append(disease_name)
                    print(f"  ✗ '{disease_name}' -> 未找到映射")

        # 保存结果
        print(f"\n正在保存结果到 {output_file}...")
        result_df.to_excel(output_file, index=False)

        # 最终统计
        print(f"\n=== 最终统计 ===")
        print(f"原本已有DOID: {already_mapped_count}")
        print(f"新映射成功: {mapped_count}")
        print(f"映射失败: {unmapped_count}")
        print(f"总DOID覆盖率: {(already_mapped_count + mapped_count) / len(df) * 100:.1f}%")

        # 显示映射详情
        if mapping_details:
            print(f"\n新映射成功的疾病（前10个）:")
            for i, (disease, doid) in enumerate(mapping_details[:10]):
                print(f"  {i + 1}: '{disease}' -> '{doid}'")
            if len(mapping_details) > 10:
                print(f"  ... 还有 {len(mapping_details) - 10} 个")

        # 显示未映射的疾病
        if unmapped_diseases:
            unique_unmapped = sorted(list(set(unmapped_diseases)))
            print(f"\n未找到映射的疾病（前10个）:")
            for i, disease in enumerate(unique_unmapped[:10]):
                print(f"  {i + 1}: '{disease}'")
            if len(unique_unmapped) > 10:
                print(f"  ... 还有 {len(unique_unmapped) - 10} 个")

        print(f"\n结果已保存到: {output_file}")

        return result_df

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        raise


def preview_obo_parsing(obo_file):
    """
    预览OBO文件解析结果
    """
    try:
        print("=== OBO文件解析预览 ===")

        with open(obo_file, 'r', encoding='utf-8') as file:
            content = file.read()

        # 获取前几个[Term]进行预览
        terms = content.split('[Term]')

        print(f"文件包含 {len(terms) - 1} 个疾病条目")
        print("\n前5个条目解析结果:")

        for i, term_text in enumerate(terms[1:6]):  # 预览前5个
            term_data = parse_single_term(term_text)
            if term_data:
                print(f"\n条目 {i + 1}:")
                print(f"  ID: {term_data['id']}")
                print(f"  名称: {term_data['name']}")
                if term_data['synonyms']:
                    print(f"  同义词: {', '.join(term_data['synonyms'])}")
                else:
                    print(f"  同义词: 无")
            else:
                print(f"\n条目 {i + 1}: 解析失败或数据不完整")

    except Exception as e:
        print(f"预览过程中出现错误: {str(e)}")


def find_similar_diseases(target_disease, disease_mapping, threshold=0.6):
    """
    寻找相似的疾病名称，帮助调试未匹配的情况
    """
    try:
        from difflib import SequenceMatcher

        target_normalized = normalize_disease_name(target_disease)
        similarities = []

        for disease_name in disease_mapping.keys():
            similarity = SequenceMatcher(None, target_normalized, disease_name).ratio()
            if similarity >= threshold:
                similarities.append((disease_name, disease_mapping[disease_name], similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[2], reverse=True)

        return similarities[:5]  # 返回前5个最相似的

    except ImportError:
        return []


def analyze_unmapped_diseases(xlsx_file, obo_file):
    """
    分析未映射的疾病，提供相似匹配建议
    """
    try:
        df = pd.read_excel(xlsx_file)
        disease_mapping = parse_obo_file(obo_file)

        print("=== 未映射疾病分析 ===")

        unmapped_diseases = []
        for _, row in df.iterrows():
            if not has_doid(row['DOID']):
                disease_name = row['Disease_Name']
                normalized_name = normalize_disease_name(disease_name)

                if normalized_name not in disease_mapping:
                    unmapped_diseases.append(disease_name)

        unique_unmapped = list(set(unmapped_diseases))
        print(f"找到 {len(unique_unmapped)} 个唯一的未映射疾病")

        print("\n相似匹配建议（前10个）:")
        for i, disease in enumerate(unique_unmapped[:10]):
            print(f"\n{i + 1}. '{disease}':")
            similar = find_similar_diseases(disease, disease_mapping)
            if similar:
                for name, doid, score in similar[:3]:
                    print(f"  相似: '{name}' -> {doid} (相似度: {score:.3f})")
            else:
                print(f"  未找到相似匹配")

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    xlsx_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\还未收集到的药物\mapped_result.xlsx"  # xlsx数据文件
    obo_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\ncRNA_Disease\初步过滤\还未收集到的药物\map.txt"  # OBO本体文件
    output_file = "mapped_diseases.xlsx"  # 输出文件

    try:
        # 预览OBO文件解析
        preview_obo_parsing(obo_file)

        print("\n" + "=" * 60)

        # 执行映射
        result_df = map_diseases_with_obo(xlsx_file, obo_file, output_file)

        print("\n" + "=" * 60)

        # 分析未映射的疾病
        analyze_unmapped_diseases(xlsx_file, obo_file)

        print("\n任务完成！")

    except Exception as e:
        print(f"执行失败: {str(e)}")


# 简化版本函数
def simple_obo_mapping(xlsx_file, obo_file, output_file):
    """
    简化版OBO映射函数
    """
    # 解析OBO文件
    disease_mapping = parse_obo_file(obo_file)

    # 读取xlsx文件
    df = pd.read_excel(xlsx_file)

    # 执行映射
    for idx, row in df.iterrows():
        if not has_doid(row['DOID']):
            normalized_name = normalize_disease_name(row['Disease_Name'])
            if normalized_name in disease_mapping:
                df.at[idx, 'DOID'] = disease_mapping[normalized_name]

    # 保存结果
    df.to_excel(output_file, index=False)

    return df