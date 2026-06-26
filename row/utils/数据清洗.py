import pandas as pd
import re
import os
from datetime import datetime


def clean_disease_data(cleaning_strategy='conservative'):
    """
    清洗疾病关联数据并生成新文件

    Parameters:
    cleaning_strategy: str
        - 'conservative': 只清理明显的空格问题，保留所有可能有效的数据
        - 'standard': 只保留标准DOID格式的数据
        - 'manual': 显示所有问题数据，让用户手动决定
    """

    # 读取原始文件
    print("读取原始疾病关联文件...")
    files_info = {
        'circRNA_Disease.xlsx': {'df': None, 'entity_col': 'circBase_ID', 'disease_col': 'DOID'},
        'miRNA_Disease.xlsx': {'df': None, 'entity_col': 'miRBase_ID', 'disease_col': 'DOID'},
        'lncRNA_Disease.xlsx': {'df': None, 'entity_col': 'ENSEMBL_ID', 'disease_col': 'DOID'},
        'Drug_Disease.xlsx': {'df': None, 'entity_col': 'CID', 'disease_col': 'DOID'}
    }

    for filename, info in files_info.items():
        if os.path.exists(filename):
            info['df'] = pd.read_excel(filename)
            print(f"✅ {filename}: {len(info['df'])} 条记录")
        else:
            print(f"❌ 文件不存在: {filename}")
            return

    # 收集所有原始Disease ID进行分析
    print("\n分析Disease ID格式...")
    all_disease_ids = set()

    for info in files_info.values():
        if info['df'] is not None:
            disease_ids = info['df'][info['disease_col']].astype(str).unique()
            all_disease_ids.update(disease_ids)

    print(f"发现 {len(all_disease_ids)} 个独特的Disease ID")

    # 分析ID格式
    standard_pattern = re.compile(r'^DOID:\d+$')

    analysis = {
        'standard': [],  # 标准格式
        'whitespace_only': [],  # 只有空格问题
        'other_issues': [],  # 其他格式问题
        'empty_or_nan': []  # 空值或NaN
    }

    for disease_id in all_disease_ids:
        if pd.isna(disease_id) or disease_id in ['nan', 'None', '']:
            analysis['empty_or_nan'].append(disease_id)
        elif standard_pattern.match(disease_id):
            analysis['standard'].append(disease_id)
        elif standard_pattern.match(disease_id.strip()):
            analysis['whitespace_only'].append(disease_id)
        else:
            analysis['other_issues'].append(disease_id)

    # 显示分析结果
    print("\nDisease ID格式分析:")
    print(f"  标准格式: {len(analysis['standard'])} 个")
    print(f"  仅空格问题: {len(analysis['whitespace_only'])} 个")
    print(f"  其他格式问题: {len(analysis['other_issues'])} 个")
    print(f"  空值/NaN: {len(analysis['empty_or_nan'])} 个")

    if analysis['whitespace_only']:
        print(f"\n仅空格问题的示例 (前5个):")
        for i, disease_id in enumerate(analysis['whitespace_only'][:5]):
            print(f"  {repr(disease_id)} -> {repr(disease_id.strip())}")

    if analysis['other_issues']:
        print(f"\n其他格式问题的示例 (前10个):")
        for i, disease_id in enumerate(analysis['other_issues'][:10]):
            print(f"  {repr(disease_id)}")

    # 根据策略进行清洗
    print(f"\n使用清洗策略: {cleaning_strategy}")

    def clean_disease_id(disease_id, strategy):
        """清洗单个Disease ID"""
        if pd.isna(disease_id):
            return None

        disease_id = str(disease_id)

        if strategy == 'conservative':
            # 保守策略：只处理明显的空格问题
            if disease_id in analysis['empty_or_nan']:
                return None
            elif disease_id in analysis['whitespace_only']:
                return disease_id.strip()
            else:
                return disease_id  # 保留原样，包括有问题的

        elif strategy == 'standard':
            # 标准策略：只保留标准格式
            stripped = disease_id.strip()
            if standard_pattern.match(stripped):
                return stripped
            else:
                return None

        elif strategy == 'manual':
            # 手动策略：显示问题让用户决定
            if disease_id in analysis['other_issues']:
                print(f"遇到问题ID: {repr(disease_id)}")
                choice = input("保留(k)/删除(d)/替换(r): ").lower()
                if choice == 'k':
                    return disease_id
                elif choice == 'd':
                    return None
                elif choice == 'r':
                    new_id = input("输入新ID: ")
                    return new_id if new_id else None
            else:
                return disease_id.strip() if disease_id.strip() else None

    # 处理每个文件
    cleaning_stats = {
        'total_records': 0,
        'removed_records': 0,
        'modified_disease_ids': 0,
        'files_processed': 0
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for filename, info in files_info.items():
        if info['df'] is None:
            continue

        print(f"\n处理文件: {filename}")
        df = info['df'].copy()
        original_count = len(df)

        # 清洗Disease ID
        df['原始_DOID'] = df[info['disease_col']].copy()  # 保存原始值用于对比
        df['清洗后_DOID'] = df[info['disease_col']].apply(
            lambda x: clean_disease_id(x, cleaning_strategy)
        )

        # 统计修改情况
        modified = df['原始_DOID'].astype(str) != df['清洗后_DOID'].astype(str)
        modified_count = modified.sum()

        # 移除清洗后为None的记录
        df_cleaned = df[df['清洗后_DOID'].notna()].copy()
        removed_count = original_count - len(df_cleaned)

        # 更新DOID列为清洗后的值
        df_cleaned[info['disease_col']] = df_cleaned['清洗后_DOID']
        df_cleaned = df_cleaned.drop(['原始_DOID', '清洗后_DOID'], axis=1)

        print(f"  原始记录: {original_count}")
        print(f"  清洗后记录: {len(df_cleaned)}")
        print(f"  删除记录: {removed_count}")
        print(f"  修改Disease ID: {modified_count}")

        # 保存清洗后的文件
        output_filename = f"cleaned_{filename.replace('.xlsx', f'_{timestamp}.xlsx')}"
        df_cleaned.to_excel(output_filename, index=False)
        print(f"  保存到: {output_filename}")

        # 更新统计
        cleaning_stats['total_records'] += original_count
        cleaning_stats['removed_records'] += removed_count
        cleaning_stats['modified_disease_ids'] += modified_count
        cleaning_stats['files_processed'] += 1

        # 如果有修改，生成对比文件
        if modified_count > 0:
            comparison_df = df[modified].copy()
            comparison_filename = f"changes_{filename.replace('.xlsx', f'_{timestamp}.xlsx')}"
            comparison_df.to_excel(comparison_filename, index=False)
            print(f"  修改对比保存到: {comparison_filename}")

    # 生成清洗报告
    print(f"\n{'=' * 50}")
    print("数据清洗完成!")
    print(f"{'=' * 50}")
    print(f"处理文件数: {cleaning_stats['files_processed']}")
    print(f"总记录数: {cleaning_stats['total_records']}")
    print(f"删除记录数: {cleaning_stats['removed_records']}")
    print(f"修改Disease ID数: {cleaning_stats['modified_disease_ids']}")
    print(
        f"数据保留率: {((cleaning_stats['total_records'] - cleaning_stats['removed_records']) / cleaning_stats['total_records'] * 100):.2f}%")

    # 生成最终的Disease统计
    print(f"\n重新统计清洗后的Disease ID...")

    cleaned_diseases = set()
    for filename, info in files_info.items():
        output_filename = f"cleaned_{filename.replace('.xlsx', f'_{timestamp}.xlsx')}"
        if os.path.exists(output_filename):
            df = pd.read_excel(output_filename)
            disease_ids = df[info['disease_col']].astype(str).unique()
            cleaned_diseases.update(disease_ids)

    print(f"清洗后独特Disease ID数量: {len(cleaned_diseases)}")

    # 保存清洗报告
    report = {
        'cleaning_strategy': cleaning_strategy,
        'timestamp': timestamp,
        'original_analysis': analysis,
        'cleaning_stats': cleaning_stats,
        'final_disease_count': len(cleaned_diseases)
    }

    import json
    with open(f'cleaning_report_{timestamp}.json', 'w', encoding='utf-8') as f:
        # 将set转换为list，将numpy int转换为python int以便JSON序列化
        report_copy = report.copy()
        for key, value in report_copy['original_analysis'].items():
            report_copy['original_analysis'][key] = list(value)

        # 转换cleaning_stats中的numpy int64
        for key, value in report_copy['cleaning_stats'].items():
            if hasattr(value, 'item'):  # numpy数值类型
                report_copy['cleaning_stats'][key] = value.item()
            else:
                report_copy['cleaning_stats'][key] = int(value)

        report_copy['final_disease_count'] = int(report_copy['final_disease_count'])

        json.dump(report_copy, f, ensure_ascii=False, indent=2)

    print(f"清洗报告保存到: cleaning_report_{timestamp}.json")

    return cleaned_diseases, cleaning_stats


def batch_clean_with_different_strategies():
    """
    使用不同策略批量清洗，便于比较效果
    """
    strategies = ['conservative', 'standard']

    print("批量清洗模式 - 将使用多种策略进行清洗")
    print("=" * 60)

    results = {}
    for strategy in strategies:
        print(f"\n使用策略: {strategy.upper()}")
        print("-" * 40)

        try:
            cleaned_diseases, stats = clean_disease_data(strategy)
            results[strategy] = {
                'disease_count': len(cleaned_diseases),
                'stats': stats
            }
        except Exception as e:
            print(f"策略 {strategy} 执行失败: {e}")
            results[strategy] = None

    # 比较结果
    print(f"\n{'=' * 60}")
    print("不同策略效果比较:")
    print(f"{'=' * 60}")

    for strategy, result in results.items():
        if result:
            print(f"{strategy.upper()}策略:")
            print(f"  保留Disease数: {result['disease_count']}")
            print(
                f"  数据保留率: {((result['stats']['total_records'] - result['stats']['removed_records']) / result['stats']['total_records'] * 100):.2f}%")
            print(f"  删除记录数: {result['stats']['removed_records']}")
        print()


if __name__ == "__main__":
    print("疾病关联数据清洗工具")
    print("=" * 40)

    print("\n选择清洗模式:")
    print("1. 保守清洗 (只处理空格问题)")
    print("2. 标准清洗 (只保留标准DOID格式)")
    print("3. 手动清洗 (遇到问题时询问)")
    print("4. 批量对比 (使用多种策略)")

    choice = input("\n请选择 (1-4): ").strip()

    if choice == '1':
        clean_disease_data('conservative')
    elif choice == '2':
        clean_disease_data('standard')
    elif choice == '3':
        clean_disease_data('manual')
    elif choice == '4':
        batch_clean_with_different_strategies()
    else:
        print("无效选择，使用默认保守策略")
        clean_disease_data('conservative')