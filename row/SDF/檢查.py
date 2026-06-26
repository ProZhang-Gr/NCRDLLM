import os
import pandas as pd
from pathlib import Path
import re


def analyze_sdf_excel_match(excel_file="ALLdrug-smiles.xlsx", sdf_dir="sdf_files"):
    """
    统计SDF文件与Excel文件的匹配情况

    Args:
        excel_file: Excel文件路径
        sdf_dir: SDF文件目录

    Returns:
        dict: 详细的匹配统计信息
    """
    print("开始统计SDF文件与Excel文件的匹配情况...")

    # 1. 读取Excel文件中的CID
    if not os.path.exists(excel_file):
        print(f"✗ Excel文件不存在: {excel_file}")
        return None

    df = pd.read_excel(excel_file)
    excel_cids = set(df['CID'].astype(int))
    print(f"Excel文件中的CID数量: {len(excel_cids)}")

    # 2. 扫描SDF文件目录
    if not os.path.exists(sdf_dir):
        print(f"✗ SDF目录不存在: {sdf_dir}")
        return None

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]
    print(f"SDF文件数量: {len(sdf_files)}")

    # 3. 从SDF文件名提取CID
    sdf_cids = set()
    invalid_sdf_files = []

    for sdf_file in sdf_files:
        # 提取文件名中的CID（去掉.sdf后缀）
        match = re.match(r'^(\d+)\.sdf$', sdf_file)
        if match:
            cid = int(match.group(1))
            sdf_cids.add(cid)
        else:
            invalid_sdf_files.append(sdf_file)

    print(f"有效的SDF文件CID数量: {len(sdf_cids)}")
    if invalid_sdf_files:
        print(f"无效的SDF文件名: {invalid_sdf_files}")

    # 4. 计算匹配情况
    matched_cids = excel_cids.intersection(sdf_cids)  # 既在Excel中又有SDF文件的CID
    excel_only_cids = excel_cids - sdf_cids  # 只在Excel中，没有SDF文件的CID
    sdf_only_cids = sdf_cids - excel_cids  # 只有SDF文件，不在Excel中的CID

    # 5. 详细统计
    result = {
        'excel_total': len(excel_cids),
        'sdf_total': len(sdf_cids),
        'matched_count': len(matched_cids),
        'excel_only_count': len(excel_only_cids),
        'sdf_only_count': len(sdf_only_cids),
        'match_rate': len(matched_cids) / len(excel_cids) * 100 if excel_cids else 0,
        'coverage_rate': len(matched_cids) / len(sdf_cids) * 100 if sdf_cids else 0,
        'matched_cids': sorted(list(matched_cids)),
        'excel_only_cids': sorted(list(excel_only_cids)),
        'sdf_only_cids': sorted(list(sdf_only_cids)),
        'invalid_sdf_files': invalid_sdf_files
    }

    return result


def print_match_summary(result):
    """
    打印匹配结果摘要
    """
    if result is None:
        return

    print("\n" + "=" * 70)
    print("SDF文件与Excel文件匹配统计摘要")
    print("=" * 70)

    print(f"Excel中的CID总数:        {result['excel_total']}")
    print(f"SDF文件总数:             {result['sdf_total']}")
    print(f"成功匹配的CID数量:       {result['matched_count']}")
    print(f"缺失SDF文件的CID数量:    {result['excel_only_count']}")
    print(f"多余SDF文件的CID数量:    {result['sdf_only_count']}")

    print(f"\n匹配率: {result['match_rate']:.1f}% (匹配数/Excel总数)")
    print(f"覆盖率: {result['coverage_rate']:.1f}% (匹配数/SDF总数)")

    # 状态评估
    if result['match_rate'] == 100:
        print(f"🎉 完美匹配！所有Excel中的CID都有对应的SDF文件！")
    elif result['match_rate'] >= 95:
        print(f"✅ 匹配情况很好！只有少数CID缺失SDF文件。")
    elif result['match_rate'] >= 80:
        print(f"⚠️  匹配情况一般，需要补充一些SDF文件。")
    else:
        print(f"❌ 匹配情况较差，大量CID缺失SDF文件。")


def print_detailed_analysis(result, show_lists=True, max_show=20):
    """
    打印详细分析结果
    """
    if result is None:
        return

    print("\n" + "-" * 70)
    print("详细分析")
    print("-" * 70)

    # 缺失SDF文件的CID
    if result['excel_only_cids']:
        print(f"\n📝 缺失SDF文件的CID ({len(result['excel_only_cids'])}个):")
        if show_lists:
            cids_to_show = result['excel_only_cids'][:max_show]
            print(f"   {cids_to_show}")
            if len(result['excel_only_cids']) > max_show:
                print(f"   ... 还有 {len(result['excel_only_cids']) - max_show} 个")
        print(f"   建议: 这些CID需要下载对应的SDF文件")

    # 多余的SDF文件
    if result['sdf_only_cids']:
        print(f"\n📁 多余的SDF文件 ({len(result['sdf_only_cids'])}个):")
        if show_lists:
            cids_to_show = result['sdf_only_cids'][:max_show]
            print(f"   {cids_to_show}")
            if len(result['sdf_only_cids']) > max_show:
                print(f"   ... 还有 {len(result['sdf_only_cids']) - max_show} 个")
        print(f"   建议: 这些SDF文件可能是多余的，或者Excel文件不完整")

    # 无效的SDF文件名
    if result['invalid_sdf_files']:
        print(f"\n❓ 无效的SDF文件名 ({len(result['invalid_sdf_files'])}个):")
        if show_lists:
            print(f"   {result['invalid_sdf_files']}")
        print(f"   建议: 检查这些文件的命名是否正确")


def generate_missing_cid_list(result, output_file="missing_cids.txt"):
    """
    生成缺失CID列表文件
    """
    if result is None or not result['excel_only_cids']:
        print("没有缺失的CID需要生成列表")
        return

    with open(output_file, 'w') as f:
        f.write("# 缺失SDF文件的CID列表\n")
        f.write(f"# 总数: {len(result['excel_only_cids'])}个\n")
        f.write("# 格式: 每行一个CID\n\n")

        for cid in result['excel_only_cids']:
            f.write(f"{cid}\n")

    print(f"\n📄 缺失CID列表已保存到: {output_file}")


def check_sdf_file_sizes(sdf_dir="sdf_files", min_size=100):
    """
    检查SDF文件大小，识别可能损坏的文件
    """
    print(f"\n检查SDF文件大小（最小大小: {min_size} bytes）...")

    if not os.path.exists(sdf_dir):
        print(f"SDF目录不存在: {sdf_dir}")
        return

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]
    small_files = []
    total_size = 0

    for sdf_file in sdf_files:
        file_path = os.path.join(sdf_dir, sdf_file)
        file_size = os.path.getsize(file_path)
        total_size += file_size

        if file_size < min_size:
            small_files.append((sdf_file, file_size))

    print(f"SDF文件总数: {len(sdf_files)}")
    print(f"总大小: {total_size / 1024 / 1024:.2f} MB")
    print(f"平均大小: {total_size / len(sdf_files):.0f} bytes" if sdf_files else "N/A")

    if small_files:
        print(f"\n⚠️  可能有问题的小文件 ({len(small_files)}个):")
        for filename, size in small_files[:10]:  # 只显示前10个
            print(f"   {filename}: {size} bytes")
        if len(small_files) > 10:
            print(f"   ... 还有 {len(small_files) - 10} 个小文件")
    else:
        print(f"✅ 所有SDF文件大小都正常")


# 使用示例
if __name__ == "__main__":
    # 分析匹配情况
    result = analyze_sdf_excel_match(
        excel_file="ALLdrug-smiles.xlsx",
        sdf_dir="sdf_files"
    )

    if result:
        # 打印摘要
        print_match_summary(result)

        # 打印详细分析
        print_detailed_analysis(result, show_lists=True, max_show=30)

        # 生成缺失CID列表
        if result['excel_only_cids']:
            generate_missing_cid_list(result, "missing_cids.txt")

        # 检查SDF文件大小
        check_sdf_file_sizes("sdf_files", min_size=100)

        print(f"\n" + "=" * 70)
        print("分析完成！")

        # 给出建议
        if result['excel_only_cids']:
            print(f"\n💡 下一步建议:")
            print(f"   1. 使用missing_cids.txt中的CID列表下载缺失的SDF文件")
            print(f"   2. 或者从Excel中删除这些CID的行")

        if result['match_rate'] == 100:
            print(f"\n🎉 数据集已完整！可以开始进行图神经网络的特征提取了！")