import pandas as pd
import os


def load_target_cids(cid_file):
    """加载目标CID列表"""

    print(f"加载CID文件: {cid_file}")

    try:
        # 读取CID文件
        df_cid = pd.read_excel(cid_file)
        print(f"CID文件列名: {list(df_cid.columns)}")
        print(f"CID文件行数: {len(df_cid)}")

        # 检查CID列是否存在
        if 'CID' not in df_cid.columns:
            print("错误: 文件中没有找到'CID'列")
            return set()

        # 提取CID列表并转换为集合
        target_cids = set(df_cid['CID'].dropna().astype(int))

        print(f"成功加载 {len(target_cids)} 个唯一CID")

        # 显示前10个CID
        cid_list = sorted(list(target_cids))
        print(f"CID示例: {cid_list[:10]}")

        # 显示CID文件预览
        print(f"\nCID文件预览:")
        print(df_cid.head(5).to_string(index=False))

        return target_cids

    except Exception as e:
        print(f"加载CID文件出错: {e}")
        return set()


def filter_ctd_by_cids(ctd_file, target_cids, output_file):
    """根据CID列表筛选CTD数据"""

    print(f"\n开始筛选CTD文件: {ctd_file}")

    try:
        # 读取CTD数据
        df_ctd = pd.read_excel(ctd_file)
        print(f"CTD文件行数: {len(df_ctd)}")
        print(f"CTD文件列名: {list(df_ctd.columns)}")

        # 检查CID列是否存在
        if 'CID' not in df_ctd.columns:
            print("错误: CTD文件中没有找到'CID'列")
            return None

        # 显示原始数据预览
        print(f"\nCTD原始数据预览:")
        print(df_ctd.head(3).to_string(index=False))

        # 统计原始数据中的CID
        original_unique_cids = df_ctd['CID'].nunique()
        original_cid_set = set(df_ctd['CID'].dropna())

        print(f"\nCTD数据统计:")
        print(f"原始记录数: {len(df_ctd)}")
        print(f"原始唯一CID数: {original_unique_cids}")

        # 检查目标CID在CTD数据中的覆盖情况
        found_cids = target_cids.intersection(original_cid_set)
        not_found_cids = target_cids - original_cid_set

        print(f"\nCID匹配情况:")
        print(f"目标CID数: {len(target_cids)}")
        print(f"在CTD中找到的CID数: {len(found_cids)}")
        print(f"未在CTD中找到的CID数: {len(not_found_cids)}")
        print(f"覆盖率: {len(found_cids) / len(target_cids) * 100:.1f}%")

        # 显示未找到的CID（前10个）
        if not_found_cids:
            not_found_list = sorted(list(not_found_cids))
            print(f"未找到的CID示例: {not_found_list[:10]}")

        # 筛选CTD数据
        print(f"\n开始筛选数据...")
        filtered_df = df_ctd[df_ctd['CID'].isin(target_cids)].copy()

        print(f"筛选完成!")
        print(f"筛选后记录数: {len(filtered_df)}")
        print(f"筛选后唯一CID数: {filtered_df['CID'].nunique()}")
        print(
            f"筛选后唯一化合物数: {filtered_df['ChemicalName'].nunique() if 'ChemicalName' in filtered_df.columns else '未知'}")
        print(
            f"筛选后唯一疾病数: {filtered_df['DiseaseID'].nunique() if 'DiseaseID' in filtered_df.columns else '未知'}")

        # 保存筛选后的数据
        filtered_df.to_excel(output_file, index=False)
        print(f"\n筛选后的数据已保存: {output_file}")

        # 显示筛选后数据预览
        print(f"\n筛选后数据预览:")
        print(filtered_df.head(5).to_string(index=False))

        # 统计每个CID的记录数
        cid_counts = filtered_df['CID'].value_counts().head(10)
        print(f"\n记录数最多的CID (前10个):")
        for cid, count in cid_counts.items():
            chemical_name = filtered_df[filtered_df['CID'] == cid]['ChemicalName'].iloc[
                0] if 'ChemicalName' in filtered_df.columns else 'Unknown'
            print(f"  CID {cid}: {count} 条记录 ({chemical_name})")

        return filtered_df

    except Exception as e:
        print(f"筛选CTD数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_filtered_data(filtered_df):
    """验证筛选后的数据质量"""

    print(f"\n验证筛选后的数据...")

    if filtered_df is None or len(filtered_df) == 0:
        print("警告: 筛选后没有数据")
        return

    # 检查数据完整性
    print(f"数据完整性检查:")
    for col in filtered_df.columns:
        null_count = filtered_df[col].isnull().sum()
        null_percent = null_count / len(filtered_df) * 100
        print(f"  {col}: {null_count} 个空值 ({null_percent:.1f}%)")

    # 检查DOID格式（如果存在）
    if 'DiseaseID' in filtered_df.columns:
        doid_pattern = filtered_df['DiseaseID'].str.contains('DOID:', na=False)
        valid_doid_count = doid_pattern.sum()
        print(f"\nDOID格式检查:")
        print(
            f"  有效DOID格式: {valid_doid_count} / {len(filtered_df)} ({valid_doid_count / len(filtered_df) * 100:.1f}%)")


def main():
    # 文件配置
    cid_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\Drug_Disease\CTD获取的数据\ALLdrug-smiles.xlsx'  # 包含目标CID的文件
    ctd_file = 'ctd_with_doid_results.xlsx'  # CTD数据文件
    output_file = 'ctd_filtered_by_cid.xlsx'  # 筛选后的输出文件

    print("=" * 70)
    print("基于CID筛选CTD数据")
    print("=" * 70)

    # 检查文件存在性
    if not os.path.exists(cid_file):
        print(f"错误: 找不到CID文件 {cid_file}")
        print("请将包含CID和SMILES的Excel文件重命名为此文件名")
        return

    if not os.path.exists(ctd_file):
        print(f"错误: 找不到CTD文件 {ctd_file}")
        print("请确保CTD数据文件存在")
        return

    # 加载目标CID列表
    target_cids = load_target_cids(cid_file)

    if not target_cids:
        print("无法加载目标CID列表，程序终止")
        return

    # 筛选CTD数据
    filtered_df = filter_ctd_by_cids(ctd_file, target_cids, output_file)

    if filtered_df is not None:
        # 验证筛选结果
        validate_filtered_data(filtered_df)

        print(f"\n" + "=" * 70)
        print("筛选完成!")
        print("=" * 70)
        print(f"输入文件: {ctd_file}")
        print(f"筛选条件: {len(target_cids)} 个目标CID")
        print(f"输出文件: {output_file}")
        print(f"筛选结果: {len(filtered_df)} 条记录")
        print("=" * 70)
    else:
        print("筛选失败")


if __name__ == "__main__":
    main()