import pandas as pd
import os


def read_unmatched_ids(unmatched_file):
    """
    读取未匹配ID文件，返回ID集合
    """
    df = pd.read_excel(unmatched_file)
    unmatched_ids = set(df['Unmatched_miRBase_ID'].tolist())
    return unmatched_ids


def filter_target_file(target_file, unmatched_ids, id_column='miRBase_ID'):
    """
    过滤目标文件，删除在未匹配列表中的行
    """
    # 读取目标文件
    df = pd.read_excel(target_file)

    # 记录原始行数
    original_rows = len(df)

    # 找出需要删除的行
    rows_to_delete = df[df[id_column].isin(unmatched_ids)]
    deleted_ids = rows_to_delete[id_column].tolist()

    # 统计每个未匹配ID被删除的次数
    deletion_stats = {}
    for uid in unmatched_ids:
        count = sum(1 for row_id in deleted_ids if row_id == uid)
        if count > 0:
            deletion_stats[uid] = count

    # 过滤数据，保留不在未匹配列表中的行
    filtered_df = df[~df[id_column].isin(unmatched_ids)]

    # 记录过滤后行数
    filtered_rows = len(filtered_df)
    deleted_rows = original_rows - filtered_rows

    return filtered_df, original_rows, filtered_rows, deleted_rows, deletion_stats, deleted_ids


def create_deletion_report(unmatched_ids, deletion_stats, deleted_ids, original_rows, filtered_rows, deleted_rows):
    """
    创建删除统计报告
    """
    # 统计报告数据
    summary_stats = {
        'Metric': [
            'Original Total Rows',
            'Rows After Filtering',
            'Total Deleted Rows',
            'Deletion Rate (%)',
            'Unique IDs in Unmatched List',
            'Unique IDs Actually Deleted',
            'IDs Not Found in Target File'
        ],
        'Count': [
            original_rows,
            filtered_rows,
            deleted_rows,
            round((deleted_rows / original_rows * 100), 2) if original_rows > 0 else 0,
            len(unmatched_ids),
            len(deletion_stats),
            len(unmatched_ids) - len(deletion_stats)
        ]
    }

    # 详细删除统计
    detailed_deletion = []
    for uid in sorted(unmatched_ids):
        count = deletion_stats.get(uid, 0)
        status = 'Deleted' if count > 0 else 'Not Found in Target'
        detailed_deletion.append({
            'Unmatched_ID': uid,
            'Rows_Deleted': count,
            'Status': status
        })

    return summary_stats, detailed_deletion


def main():
    # 文件路径设置
    unmatched_file = 'unmatched_ids.xlsx'  # 未匹配ID文件
    target_file = r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\RNAInter第一步收集.xlsx'  # 需要过滤的目标文件

    # 输出文件路径
    output_filtered = 'filtered_data.xlsx'
    output_deletion_report = 'deletion_report.xlsx'
    output_deletion_summary = 'deletion_summary.xlsx'

    try:
        # 检查文件是否存在
        if not os.path.exists(unmatched_file):
            print(f"错误: 未找到文件 {unmatched_file}")
            return

        if not os.path.exists(target_file):
            print(f"错误: 未找到文件 {target_file}")
            return

        # 读取未匹配ID列表
        print("读取未匹配ID文件...")
        unmatched_ids = read_unmatched_ids(unmatched_file)
        print(f"未匹配ID列表中共有 {len(unmatched_ids)} 个唯一ID")

        # 显示未匹配ID列表
        print("未匹配ID列表:")
        for i, uid in enumerate(sorted(unmatched_ids)):
            print(f"  {i + 1}. {uid}")

        # 过滤目标文件
        print(f"\n处理目标文件: {target_file}")
        filtered_df, original_rows, filtered_rows, deleted_rows, deletion_stats, deleted_ids = filter_target_file(
            target_file, unmatched_ids
        )

        # 保存过滤后的文件
        filtered_df.to_excel(output_filtered, index=False)
        print(f"过滤后的数据已保存到: {output_filtered}")

        # 创建删除统计报告
        summary_stats, detailed_deletion = create_deletion_report(
            unmatched_ids, deletion_stats, deleted_ids, original_rows, filtered_rows, deleted_rows
        )

        # 保存汇总统计
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(output_deletion_summary, index=False)
        print(f"删除汇总统计已保存到: {output_deletion_summary}")

        # 保存详细删除报告
        detailed_df = pd.DataFrame(detailed_deletion)
        detailed_df.to_excel(output_deletion_report, index=False)
        print(f"详细删除报告已保存到: {output_deletion_report}")

        # 打印处理结果摘要
        print("\n" + "=" * 50)
        print("处理结果摘要")
        print("=" * 50)
        print(f"原始总行数: {original_rows}")
        print(f"删除行数: {deleted_rows}")
        print(f"保留行数: {filtered_rows}")
        print(f"删除比例: {(deleted_rows / original_rows * 100):.2f}%")

        print(f"\n未匹配ID处理情况:")
        print(f"  - 未匹配列表中的ID总数: {len(unmatched_ids)}")
        print(f"  - 实际在目标文件中找到并删除的ID数: {len(deletion_stats)}")
        print(f"  - 在未匹配列表中但目标文件中不存在的ID数: {len(unmatched_ids) - len(deletion_stats)}")

        if deletion_stats:
            print(f"\n各ID删除统计 (前10个):")
            sorted_deletion = sorted(deletion_stats.items(), key=lambda x: x[1], reverse=True)
            for i, (uid, count) in enumerate(sorted_deletion[:10]):
                print(f"  {i + 1}. {uid}: 删除了 {count} 行")
            if len(sorted_deletion) > 10:
                print(f"  ... 还有 {len(sorted_deletion) - 10} 个ID被删除")

        # 找出未在目标文件中发现的未匹配ID
        not_found_ids = unmatched_ids - set(deletion_stats.keys())
        if not_found_ids:
            print(f"\n以下未匹配ID在目标文件中未找到 (前5个):")
            for i, uid in enumerate(sorted(list(not_found_ids))[:5]):
                print(f"  {i + 1}. {uid}")
            if len(not_found_ids) > 5:
                print(f"  ... 还有 {len(not_found_ids) - 5} 个ID未找到")

        print(f"\n处理完成! 过滤后的文件已保存为: {output_filtered}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()