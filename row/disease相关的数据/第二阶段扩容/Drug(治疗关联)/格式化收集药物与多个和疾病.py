import pandas as pd
import re
from fuzzywuzzy import fuzz, process
import numpy as np


def load_disease_mapping(disease_file):
    """加载疾病-DOID映射表"""
    disease_df = pd.read_excel(disease_file)
    # 创建疾病名称到DOID的映射字典
    disease_to_doid = dict(zip(disease_df['Disease Name'], disease_df['DOID']))

    # 也创建小写版本用于模糊匹配
    disease_names = list(disease_df['Disease Name'])

    return disease_to_doid, disease_names


def fuzzy_match_disease(extracted_disease, disease_names, threshold=80):
    """使用模糊匹配找到最相似的标准疾病名称"""
    match = process.extractOne(extracted_disease, disease_names, scorer=fuzz.ratio)

    if match and match[1] >= threshold:
        return match[0], match[1]
    return None, 0


def extract_and_map_diseases(indication_text, disease_to_doid, disease_names):
    """从indication文本中提取疾病并映射到DOID"""

    # 首先尝试直接匹配已知疾病名称
    direct_matches = []
    for disease_name in disease_names:
        if disease_name.lower() in indication_text.lower():
            direct_matches.append({
                'disease': disease_name,
                'doid': disease_to_doid[disease_name],
                'match_type': 'direct',
                'confidence': 100
            })

    if direct_matches:
        return direct_matches

    # 如果没有直接匹配，使用之前的提取方法
    extracted_diseases = extract_multiple_diseases(indication_text)

    mapped_diseases = []
    for extracted in extracted_diseases:
        # 尝试模糊匹配
        matched_disease, confidence = fuzzy_match_disease(extracted, disease_names)

        if matched_disease:
            mapped_diseases.append({
                'disease': matched_disease,
                'doid': disease_to_doid[matched_disease],
                'match_type': 'fuzzy',
                'confidence': confidence,
                'original_extracted': extracted
            })
        else:
            # 保留无法匹配的疾病，供手动处理
            mapped_diseases.append({
                'disease': extracted,
                'doid': 'UNMAPPED',
                'match_type': 'extracted_only',
                'confidence': 0,
                'original_extracted': extracted
            })

    return mapped_diseases


def process_with_disease_mapping(drug_df, disease_file):
    """使用疾病映射表处理药物数据"""

    # 加载疾病映射
    disease_to_doid, disease_names = load_disease_mapping(disease_file)
    print(f"加载了 {len(disease_names)} 个标准疾病名称")

    all_records = []
    stats = {
        'direct_matches': 0,
        'fuzzy_matches': 0,
        'unmapped': 0,
        'no_diseases': 0
    }

    for idx, row in drug_df.iterrows():
        cid = row['CID']
        drug_name = row['Drug name']
        indication_text = row['Disease name']

        # 提取并映射疾病
        mapped_diseases = extract_and_map_diseases(indication_text, disease_to_doid, disease_names)

        if mapped_diseases:
            for disease_info in mapped_diseases:
                all_records.append({
                    'CID': cid,
                    'Drug_name': drug_name,
                    'Disease_name': disease_info['disease'],
                    'DOID': disease_info['doid'],
                    'Match_type': disease_info['match_type'],
                    'Confidence': disease_info['confidence'],
                    'Original_indication': indication_text,
                    'Original_extracted': disease_info.get('original_extracted', '')
                })

                # 统计
                if disease_info['match_type'] == 'direct':
                    stats['direct_matches'] += 1
                elif disease_info['match_type'] == 'fuzzy':
                    stats['fuzzy_matches'] += 1
                else:
                    stats['unmapped'] += 1
        else:
            # 没有提取到任何疾病
            all_records.append({
                'CID': cid,
                'Drug_name': drug_name,
                'Disease_name': 'NO_DISEASE_EXTRACTED',
                'DOID': 'N/A',
                'Match_type': 'failed',
                'Confidence': 0,
                'Original_indication': indication_text,
                'Original_extracted': ''
            })
            stats['no_diseases'] += 1

    result_df = pd.DataFrame(all_records)

    # 打印统计信息
    print("\n处理统计:")
    print(f"总记录数: {len(result_df)}")
    print(f"直接匹配: {stats['direct_matches']}")
    print(f"模糊匹配: {stats['fuzzy_matches']}")
    print(f"未映射: {stats['unmapped']}")
    print(f"无疾病提取: {stats['no_diseases']}")

    return result_df, stats


# 使用之前的疾病提取函数
def extract_multiple_diseases(text):
    """从适应症文本中提取多个疾病"""
    diseases = []

    patterns = [
        r'(?:treatment|management|therapy) of ([^.,;]+)',
        r'indicated for (?:the )?(?:treatment of |management of )?([^.,;]+)',
        r'used (?:to treat|for treating|in the treatment of) ([^.,;]+)',
        r'prevention of ([^.,;]+)',
        r'\b([A-Za-z\s]*(?:syndrome|disease|cancer|carcinoma|infection|disorder|condition|deficiency)[A-Za-z\s]*)\b'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            diseases.extend(split_disease_list(match))

    # 清理
    cleaned_diseases = []
    for disease in diseases:
        disease = clean_disease_name(disease)
        if is_valid_disease(disease):
            cleaned_diseases.append(disease)

    return list(set(cleaned_diseases))


def split_disease_list(text):
    separators = [' and ', ' or ', ', ', '; ']
    diseases = [text]
    for sep in separators:
        new_diseases = []
        for d in diseases:
            new_diseases.extend([x.strip() for x in d.split(sep)])
        diseases = new_diseases
    return diseases


def clean_disease_name(disease):
    disease = re.sub(r'\[.*?\]', '', disease)
    disease = re.sub(r'\s+', ' ', disease)
    disease = re.sub(r'^(the|of|in|for|with|due to|associated)\s+', '', disease, flags=re.IGNORECASE)
    return disease.strip()


def is_valid_disease(disease):
    if len(disease) < 3 or len(disease) > 100:
        return False
    invalid_terms = ['patients', 'adults', 'children', 'combination', 'use', 'treatment', 'prevention']
    return disease.lower() not in invalid_terms


# 主要处理函数
def main():
    # 文件路径
    drug_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\Drug(治疗关联)\Drugbank收集的数据\CID_DrugName_文本描述.xlsx'  # 你的药物-适应症文件
    disease_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\map文件\整合的name_DOID.xlsx"  # 疾病-DOID映射文件

    # 读取数据
    drug_df = pd.read_excel(drug_file)

    # 处理数据
    result_df, stats = process_with_disease_mapping(drug_df, disease_file)

    # 保存结果
    result_df.to_excel('mapped_drug_disease_associations.xlsx', index=False)

    # 分别保存不同质量的结果
    high_quality = result_df[result_df['Match_type'].isin(['direct', 'fuzzy']) & (result_df['Confidence'] >= 80)]
    high_quality.to_excel('high_quality_associations.xlsx', index=False)

    need_review = result_df[result_df['Match_type'].isin(['extracted_only', 'failed'])]
    need_review.to_excel('need_manual_review.xlsx', index=False)

    print(f"\n高质量关联: {len(high_quality)} 条")
    print(f"需要手动审核: {len(need_review)} 条")


if __name__ == "__main__":
    main()