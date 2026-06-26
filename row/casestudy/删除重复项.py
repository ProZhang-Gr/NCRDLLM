import pandas as pd


def remove_known_pairs(predictions_csv, known_pairs_xlsx, output_csv):
    """
    从预测结果中删除已知的正样本对

    Args:
        predictions_csv: details_predictions_simple.csv 的路径
        known_pairs_xlsx: responsed_circRNA_drug.xlsx 的路径
        output_csv: 输出文件路径
    """
    # 读取预测结果 - 尝试自动检测分隔符
    print("📖 读取预测结果...")
    try:
        # 先尝试逗号分隔
        predictions = pd.read_csv(predictions_csv, sep=',')
        if len(predictions.columns) == 1:
            # 如果只有一列，尝试Tab分隔
            predictions = pd.read_csv(predictions_csv, sep='\t')
    except:
        # 如果都失败，让pandas自动检测
        predictions = pd.read_csv(predictions_csv, sep=None, engine='python')

    print(f"   原始预测结果: {len(predictions)} 条")
    print(f"   预测结果的列名: {predictions.columns.tolist()}")

    # 读取已知配对
    print("\n📖 读取已知正样本配对...")
    known_pairs = pd.read_excel(known_pairs_xlsx)
    print(f"   已知配对: {len(known_pairs)} 条")
    print(f"   已知配对的列名: {known_pairs.columns.tolist()}")

    # 统一列名（处理可能的空格）
    known_pairs.columns = known_pairs.columns.str.strip()
    predictions.columns = predictions.columns.str.strip()

    # 检查必要的列是否存在
    required_pred_cols = ['RNA_ID', 'CID', 'true_label', 'predicted_score']
    required_known_cols = ['RNA_ID', 'CID']

    # 检查预测结果的列
    missing_pred_cols = [col for col in required_pred_cols if col not in predictions.columns]
    if missing_pred_cols:
        print(f"\n❌ 错误：预测结果缺少列: {missing_pred_cols}")
        print(f"   实际列名: {predictions.columns.tolist()}")
        return None

    # 检查已知配对的列
    missing_known_cols = [col for col in required_known_cols if col not in known_pairs.columns]
    if missing_known_cols:
        print(f"\n❌ 错误：已知配对缺少列: {missing_known_cols}")
        print(f"   实际列名: {known_pairs.columns.tolist()}")
        return None

    # 创建已知配对的标识集合
    print(f"\n🔍 创建已知配对索引...")
    known_keys = set(
        known_pairs['RNA_ID'].astype(str).str.strip() + '|' +
        known_pairs['CID'].astype(str).str.strip()
    )
    print(f"   已知配对集合大小: {len(known_keys)}")

    # 创建预测结果的配对标识
    predictions['pair_key'] = (
            predictions['RNA_ID'].astype(str).str.strip() + '|' +
            predictions['CID'].astype(str).str.strip()
    )

    # 标记是否为已知配对
    predictions['is_known'] = predictions['pair_key'].isin(known_keys)

    # 统计
    n_known = predictions['is_known'].sum()
    n_novel = (~predictions['is_known']).sum()

    print(f"\n📊 统计信息:")
    print(f"   预测结果中的已知配对: {n_known} 条 ({n_known / len(predictions) * 100:.2f}%)")
    print(f"   预测结果中的新颖配对: {n_novel} 条 ({n_novel / len(predictions) * 100:.2f}%)")

    # 显示一些已知配对的例子（用于验证）
    if n_known > 0:
        print(f"\n🔍 已知配对示例（前5条）:")
        known_examples = predictions[predictions['is_known']].head()
        print(known_examples[['RNA_ID', 'CID', 'true_label', 'predicted_score']].to_string(index=False))

    # 删除已知配对
    filtered_predictions = predictions[~predictions['is_known']].copy()

    # 删除辅助列
    filtered_predictions = filtered_predictions.drop(columns=['pair_key', 'is_known'])

    # 保存结果
    print(f"\n💾 保存过滤后的结果...")
    filtered_predictions.to_csv(output_csv, index=False)
    print(f"   ✅ 已保存到: {output_csv}")
    print(f"   保留的记录数: {len(filtered_predictions)} 条")

    # 显示一些统计
    if len(filtered_predictions) > 0:
        print(f"\n📈 过滤后的预测分数统计:")
        print(f"   平均分数: {filtered_predictions['predicted_score'].mean():.4f}")
        print(f"   最高分数: {filtered_predictions['predicted_score'].max():.4f}")
        print(f"   最低分数: {filtered_predictions['predicted_score'].min():.4f}")

        # 按标签分组统计
        print(f"\n📊 按真实标签分组:")
        for label in sorted(filtered_predictions['true_label'].unique()):
            subset = filtered_predictions[filtered_predictions['true_label'] == label]
            print(f"   Label {label}: {len(subset)} 条, 平均分数: {subset['predicted_score'].mean():.4f}")

        # 显示高分预测
        print(f"\n🌟 Top 10 高分预测（新颖配对）:")
        top10 = filtered_predictions.nlargest(10, 'predicted_score')
        print(top10[['RNA_ID', 'CID', 'true_label', 'predicted_score']].to_string(index=False))

    return filtered_predictions


# 使用示例
if __name__ == "__main__":
    # 设置文件路径
    predictions_file = "midetails_predictions_simple.csv"
    known_pairs_file = "responsed_miRNA-drug.xlsx"
    output_file = "details_predictions_novel_only.csv"

    print("=" * 60)
    print("🔧 RNA-Drug 配对过滤工具")
    print("=" * 60)

    # 执行过滤
    filtered_df = remove_known_pairs(
        predictions_file,
        known_pairs_file,
        output_file
    )

    if filtered_df is not None:
        print(f"\n" + "=" * 60)
        print(f"✅ 完成！")
        print("=" * 60)
    else:
        print(f"\n" + "=" * 60)
        print(f"❌ 处理失败，请检查文件格式")
        print("=" * 60)