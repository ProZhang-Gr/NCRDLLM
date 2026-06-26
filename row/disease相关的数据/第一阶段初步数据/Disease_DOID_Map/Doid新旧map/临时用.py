import pandas as pd
import os
from itertools import combinations


def load_doids_from_file(file_path):
    """从文件中加载第二列的DOID"""

    try:
        df = pd.read_excel(file_path)

        # 获取第二列（DOID列）
        doid_column = df.columns[1]  # 第二列

        # 提取DOID，转换为字符串集合
        doid_set = set(str(val).strip() for val in df.iloc[:, 1].dropna())

        return doid_set, doid_column, len(df)

    except Exception as e:
        print(f"错误读取文件 {file_path}: {e}")
        return set(), "Unknown", 0


def analyze_doid_intersections(file_data):
    """分析DOID的交集情况"""

    file_names = list(file_data.keys())
    doid_sets = {name: data['doids'] for name, data in file_data.items()}

    print("=" * 70)
    print("四个Final文件DOID分析")
    print("=" * 70)

    # 1. 显示各文件的基本信息
    print("\n各文件DOID统计:")
    print("-" * 50)
    for name, data in file_data.items():
        print(f"{name}:")
        print(f"  总记录数: {data['total_records']}")
        print(f"  DOID列名: {data['column_name']}")
        print(f"  唯一DOID数: {len(data['doids'])}")
        # 显示前5个DOID示例
        sample_doids = sorted(list(data['doids']))[:5]
        print(f"  DOID示例: {sample_doids}")
        print()

    # 2. 计算四个文件的完全交集
    print("\n四文件完全交集:")
    print("-" * 50)
    complete_intersection = set.intersection(*doid_sets.values())
    print(f"四个文件共同的DOID数量: {len(complete_intersection)}")

    if len(complete_intersection) > 0:
        if len(complete_intersection) <= 20:
            print(f"共同DOID: {sorted(list(complete_intersection))}")
        else:
            sorted_common = sorted(list(complete_intersection))
            print(f"前20个共同DOID: {sorted_common[:20]}")

    # 3. 计算两两交集
    print(f"\n两两交集分析:")
    print("-" * 50)
    for combo in combinations(file_names, 2):
        name1, name2 = combo
        intersection = doid_sets[name1].intersection(doid_sets[name2])

        # 计算交集比例
        total1 = len(doid_sets[name1])
        total2 = len(doid_sets[name2])
        min_size = min(total1, total2)

        print(f"{name1} ∩ {name2}:")
        print(f"  交集数量: {len(intersection)}")
        print(f"  相对较小集合的比例: {len(intersection) / min_size * 100:.1f}%")
        print()

    # 4. 计算三个文件的交集
    print(f"\n三文件交集分析:")
    print("-" * 50)
    for combo in combinations(file_names, 3):
        intersection = set.intersection(*[doid_sets[name] for name in combo])
        combo_name = " ∩ ".join(combo)
        print(f"{combo_name}: {len(intersection)} 个DOID")

    # 5. 计算并集
    print(f"\n并集分析:")
    print("-" * 50)
    all_doids = set()
    for doid_set in doid_sets.values():
        all_doids.update(doid_set)

    print(f"四个文件DOID并集数量: {len(all_doids)}")

    # 6. 分析每个文件的独有DOID
    print(f"\n独有DOID分析:")
    print("-" * 50)
    for name, doid_set in doid_sets.items():
        others = set()
        for other_name, other_set in doid_sets.items():
            if other_name != name:
                others.update(other_set)

        unique_doids = doid_set - others
        print(f"{name}独有DOID: {len(unique_doids)} 个")
        if len(unique_doids) > 0 and len(unique_doids) <= 10:
            print(f"  独有DOID: {sorted(list(unique_doids))}")
        elif len(unique_doids) > 10:
            print(f"  前5个独有DOID: {sorted(list(unique_doids))[:5]}")

    return {
        'complete_intersection': complete_intersection,
        'union': all_doids,
        'individual_counts': {name: len(data['doids']) for name, data in file_data.items()}
    }


def create_summary_report(analysis_results, file_data, output_file):
    """创建汇总报告"""

    print(f"\n生成汇总报告: {output_file}")

    # 准备汇总数据
    summary_data = []

    for name, data in file_data.items():
        summary_data.append({
            'File_Name': name,
            'Total_Records': data['total_records'],
            'Unique_DOIDs': len(data['doids']),
            'DOID_Column': data['column_name']
        })

    # 添加统计信息
    summary_data.append({
        'File_Name': 'Complete_Intersection',
        'Total_Records': len(analysis_results['complete_intersection']),
        'Unique_DOIDs': len(analysis_results['complete_intersection']),
        'DOID_Column': 'Intersection'
    })

    summary_data.append({
        'File_Name': 'Union',
        'Total_Records': len(analysis_results['union']),
        'Unique_DOIDs': len(analysis_results['union']),
        'DOID_Column': 'Union'
    })

    # 创建DataFrame并保存
    summary_df = pd.DataFrame(summary_data)

    # 创建详细的交集DOID列表
    if len(analysis_results['complete_intersection']) > 0:
        intersection_df = pd.DataFrame({
            'DOID': sorted(list(analysis_results['complete_intersection'])),
            'Appears_In_All_Files': 'Yes'
        })
    else:
        intersection_df = pd.DataFrame({
            'DOID': ['No common DOIDs'],
            'Appears_In_All_Files': 'No'
        })

    # 保存到Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        intersection_df.to_excel(writer, sheet_name='Common_DOIDs', index=False)

    print("汇总报告已保存")


def main():
    # =================================================================
    # 四个Final文件路径配置
    # =================================================================

    final_files = {
        'circRNA-Disease': r'D:\Desktop\CDLLM\ing\row\disease相关的数据\疾病的第一次产出\circRNA-Disease_Final.xlsx',
        'Drug-Disease': r'D:\Desktop\CDLLM\ing\row\disease相关的数据\疾病的第一次产出\Drug_Disease_Final.xlsx',
        'lncRNA-Disease': r'D:\Desktop\CDLLM\ing\row\disease相关的数据\疾病的第一次产出\lncRNA_Disease_Final.xlsx',
        'miRNA-Disease': r'D:\Desktop\CDLLM\ing\row\disease相关的数据\疾病的第一次产出\miRNA_Disease_Final.xlsx'
    }

    # =================================================================
    # 加载所有文件的DOID数据
    # =================================================================

    file_data = {}

    for name, file_path in final_files.items():
        if os.path.exists(file_path):
            doids, column_name, total_records = load_doids_from_file(file_path)
            file_data[name] = {
                'doids': doids,
                'column_name': column_name,
                'total_records': total_records
            }
        else:
            print(f"警告: 文件不存在 {file_path}")

    if len(file_data) == 0:
        print("错误: 没有找到任何有效文件")
        return

    # =================================================================
    # 分析DOID交集
    # =================================================================

    analysis_results = analyze_doid_intersections(file_data)

    # =================================================================
    # 生成汇总报告
    # =================================================================

    create_summary_report(analysis_results, file_data, 'final_files_doid_analysis.xlsx')

    # =================================================================
    # 最终汇总
    # =================================================================

    print(f"\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)

    for name, data in file_data.items():
        print(f"{name}: {len(data['doids'])} 个唯一DOID")

    print(f"\n四文件完全交集: {len(analysis_results['complete_intersection'])} 个DOID")
    print(f"四文件并集: {len(analysis_results['union'])} 个DOID")

    print(f"\n输出文件: final_files_doid_analysis.xlsx")
    print("=" * 70)


if __name__ == "__main__":
    main()