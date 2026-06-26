import pandas as pd
import re
import numpy as np
from collections import Counter, defaultdict


def check_sequence_files():
    """
    检查四个序列/结构文件的数据质量
    """

    files_config = {
        'ALLmiRNA-seq.xlsx': {
            'id_col': 'miRBase_ID',
            'data_col': 'seq',
            'data_type': 'RNA序列',
            'expected_pattern': r'^[AUGC]+$',  # miRNA序列应该只包含AUGC
            'id_pattern': r'^hsa-'  # miRNA ID通常以hsa-开头
        },
        'ALLlncRNA-seq.xlsx': {
            'id_col': 'ENSEMBL_ID',
            'data_col': 'seq',
            'data_type': 'RNA序列',
            'expected_pattern': r'^[AUGC]+$',  # lncRNA序列应该只包含AUGC
            'id_pattern': r'^ENSG\d+'  # Ensembl ID格式
        },
        'ALLcircRNA-seq.xlsx': {
            'id_col': 'circBase_ID',
            'data_col': 'seq',
            'data_type': 'RNA序列',
            'expected_pattern': r'^[AUGC]+$',  # circRNA序列应该只包含AUGC
            'id_pattern': r'^hsa_circ_'  # circRNA ID格式
        },
        'ALLdrug-smiles.xlsx': {
            'id_col': 'CID',
            'data_col': 'SMILES',
            'data_type': 'SMILES结构',
            'expected_pattern': None,  # SMILES比较复杂，不用正则检查
            'id_pattern': r'^\d+$'  # CID应该是纯数字
        }
    }

    print("检查序列和化学结构文件数据质量...")
    print("=" * 60)

    all_results = {}

    for filename, config in files_config.items():
        print(f"\n📁 检查文件: {filename}")
        print("-" * 40)

        try:
            df = pd.read_excel(filename)
            print(f"✅ 文件读取成功，共 {len(df)} 条记录")

            result = check_single_file(df, config, filename)
            all_results[filename] = result

        except FileNotFoundError:
            print(f"❌ 文件不存在: {filename}")
            all_results[filename] = None
        except Exception as e:
            print(f"❌ 读取文件出错: {e}")
            all_results[filename] = None

    # 生成总结报告
    print(f"\n{'=' * 60}")
    print("数据质量检查总结")
    print(f"{'=' * 60}")

    for filename, result in all_results.items():
        if result is None:
            print(f"❌ {filename}: 无法处理")
        else:
            print(f"✅ {filename}:")
            print(f"   记录数: {result['total_records']}")
            print(f"   ID重复率: {result['id_duplicates_rate']:.2f}%")
            print(f"   数据缺失率: {result['missing_data_rate']:.2f}%")
            print(f"   ID格式问题: {result['id_format_issues']}")
            print(f"   数据格式问题: {result['data_format_issues']}")

    return all_results


def check_single_file(df, config, filename):
    """
    检查单个文件的数据质量
    """

    id_col = config['id_col']
    data_col = config['data_col']
    data_type = config['data_type']

    result = {
        'filename': filename,
        'total_records': len(df),
        'columns': list(df.columns),
        'id_duplicates': 0,
        'id_duplicates_rate': 0,
        'missing_ids': 0,
        'missing_data': 0,
        'missing_data_rate': 0,
        'id_format_issues': 0,
        'data_format_issues': 0,
        'id_issues_details': [],
        'data_issues_details': [],
        'statistics': {}
    }

    # 检查列是否存在
    if id_col not in df.columns:
        print(f"❌ 缺少ID列: {id_col}")
        print(f"   实际列名: {list(df.columns)}")
        return result

    if data_col not in df.columns:
        print(f"❌ 缺少数据列: {data_col}")
        print(f"   实际列名: {list(df.columns)}")
        return result

    print(f"📊 列检查通过: {id_col}, {data_col}")

    # 1. 检查ID列
    print(f"\n🔍 检查{id_col}列...")

    # ID缺失值检查
    id_missing = df[id_col].isna().sum()
    result['missing_ids'] = id_missing
    if id_missing > 0:
        print(f"⚠️  发现 {id_missing} 个缺失的ID")

    # ID重复检查
    id_counts = df[id_col].value_counts()
    duplicated_ids = id_counts[id_counts > 1]
    result['id_duplicates'] = len(duplicated_ids)
    result['id_duplicates_rate'] = (len(duplicated_ids) / len(df)) * 100

    if len(duplicated_ids) > 0:
        print(f"⚠️  发现 {len(duplicated_ids)} 个重复的ID")
        print(f"   重复最多的前5个:")
        for id_name, count in duplicated_ids.head().items():
            print(f"     {id_name}: {count} 次")

    # ID格式检查
    if config['id_pattern']:
        pattern = re.compile(config['id_pattern'])
        valid_ids = df[id_col].astype(str).apply(lambda x: bool(pattern.match(x)) if pd.notna(x) else False)
        invalid_ids = df[~valid_ids][id_col].dropna()

        result['id_format_issues'] = len(invalid_ids)
        if len(invalid_ids) > 0:
            print(f"⚠️  发现 {len(invalid_ids)} 个格式异常的ID")
            print(f"   异常ID示例 (前10个):")
            for invalid_id in invalid_ids.head(10):
                print(f"     {repr(invalid_id)}")
            result['id_issues_details'] = invalid_ids.tolist()

    # 2. 检查数据列
    print(f"\n🔍 检查{data_col}列({data_type})...")

    # 数据缺失值检查
    data_missing = df[data_col].isna().sum()
    result['missing_data'] = data_missing
    result['missing_data_rate'] = (data_missing / len(df)) * 100
    if data_missing > 0:
        print(f"⚠️  发现 {data_missing} 个缺失的{data_type}")

    # 空字符串检查
    empty_data = (df[data_col].astype(str).str.strip() == '').sum()
    if empty_data > 0:
        print(f"⚠️  发现 {empty_data} 个空的{data_type}")

    # 数据格式检查
    if data_type == 'RNA序列':
        result.update(check_rna_sequences(df, data_col))
    elif data_type == 'SMILES结构':
        result.update(check_smiles_structures(df, data_col))

    # 3. 基本统计
    print(f"\n📈 基本统计:")
    if data_type in ['RNA序列', 'SMILES结构']:
        lengths = df[data_col].astype(str).str.len()
        result['statistics'] = {
            'length_mean': float(lengths.mean()),
            'length_std': float(lengths.std()),
            'length_min': int(lengths.min()),
            'length_max': int(lengths.max()),
            'length_median': float(lengths.median())
        }
        print(f"   长度统计: 均值={lengths.mean():.1f}, 标准差={lengths.std():.1f}")
        print(f"   长度范围: {lengths.min()} - {lengths.max()}")

    return result


def check_rna_sequences(df, seq_col):
    """
    检查RNA序列的特定问题
    """
    result = {
        'data_format_issues': 0,
        'data_issues_details': []
    }

    print(f"   🧬 RNA序列特定检查:")

    # 检查非法字符
    valid_bases = set('AUGCT')  # 允许A,U,G,C,T

    invalid_sequences = []
    sequences = df[seq_col].dropna().astype(str)

    for idx, seq in sequences.items():
        seq_upper = seq.upper().strip()
        if seq_upper:  # 非空序列
            invalid_chars = set(seq_upper) - valid_bases
            if invalid_chars:
                invalid_sequences.append({
                    'index': idx,
                    'sequence': seq[:50] + ('...' if len(seq) > 50 else ''),
                    'invalid_chars': list(invalid_chars)
                })

    result['data_format_issues'] = len(invalid_sequences)
    result['data_issues_details'] = invalid_sequences

    if invalid_sequences:
        print(f"     ⚠️  发现 {len(invalid_sequences)} 个包含非法字符的序列")
        print(f"     非法字符示例 (前5个):")
        for item in invalid_sequences[:5]:
            print(f"       行{item['index']}: 非法字符 {item['invalid_chars']}")
            print(f"       序列: {item['sequence']}")
    else:
        print(f"     ✅ 所有序列字符都合法")

    # 统计碱基组成
    all_sequences = ''.join(sequences.str.upper())
    if all_sequences:
        base_counts = Counter(all_sequences)
        print(f"     碱基组成统计:")
        for base in 'AUGCT':
            if base in base_counts:
                percentage = (base_counts[base] / len(all_sequences)) * 100
                print(f"       {base}: {base_counts[base]} ({percentage:.1f}%)")

    return result


def check_smiles_structures(df, smiles_col):
    """
    检查SMILES结构的特定问题
    """
    result = {
        'data_format_issues': 0,
        'data_issues_details': []
    }

    print(f"   🧪 SMILES结构特定检查:")

    smiles_data = df[smiles_col].dropna().astype(str)

    # 基本的SMILES格式检查
    suspicious_smiles = []
    common_smiles_chars = set('()[]@+-=#\\/.%0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

    for idx, smiles in smiles_data.items():
        smiles_clean = smiles.strip()
        if smiles_clean:
            # 检查是否包含异常字符
            unusual_chars = set(smiles_clean) - common_smiles_chars
            if unusual_chars:
                suspicious_smiles.append({
                    'index': idx,
                    'smiles': smiles[:100] + ('...' if len(smiles) > 100 else ''),
                    'unusual_chars': list(unusual_chars)
                })

    result['data_format_issues'] = len(suspicious_smiles)
    result['data_issues_details'] = suspicious_smiles

    if suspicious_smiles:
        print(f"     ⚠️  发现 {len(suspicious_smiles)} 个可能有问题的SMILES")
        print(f"     异常字符示例 (前3个):")
        for item in suspicious_smiles[:3]:
            print(f"       行{item['index']}: 异常字符 {item['unusual_chars']}")
    else:
        print(f"     ✅ 所有SMILES看起来格式正常")

    # 统计常见元素
    all_smiles = ''.join(smiles_data)
    if all_smiles:
        # 统计常见的化学元素符号
        elements = re.findall(r'[A-Z][a-z]?', all_smiles)
        element_counts = Counter(elements)
        print(f"     常见元素统计 (前10个):")
        for element, count in element_counts.most_common(10):
            print(f"       {element}: {count}")

    return result


def generate_cleaning_suggestions(results):
    """
    根据检查结果生成清洗建议
    """
    print(f"\n{'=' * 60}")
    print("数据清洗建议")
    print(f"{'=' * 60}")

    for filename, result in results.items():
        if result is None:
            continue

        print(f"\n📁 {filename}:")

        issues = []

        if result['id_duplicates'] > 0:
            issues.append(f"🔴 {result['id_duplicates']} 个重复ID需要处理")

        if result['missing_ids'] > 0:
            issues.append(f"🟡 {result['missing_ids']} 个缺失ID需要删除")

        if result['missing_data'] > 0:
            issues.append(f"🟡 {result['missing_data']} 个缺失数据需要删除")

        if result['id_format_issues'] > 0:
            issues.append(f"🔴 {result['id_format_issues']} 个ID格式问题需要检查")

        if result['data_format_issues'] > 0:
            issues.append(f"🔴 {result['data_format_issues']} 个数据格式问题需要检查")

        if not issues:
            print("   ✅ 数据质量良好，无需清洗")
        else:
            print("   需要关注的问题:")
            for issue in issues:
                print(f"     {issue}")


if __name__ == "__main__":
    results = check_sequence_files()
    generate_cleaning_suggestions(results)