import os
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import time


def download_2d_sdf_pubchem(cid, output_dir="sdf_files"):
    """
    通过PubChem下载2D SDF文件（不指定record_type='3d'）

    Args:
        cid: 化合物的CID
        output_dir: SDF文件保存目录

    Returns:
        str: SDF文件路径，如果失败返回None
    """
    Path(output_dir).mkdir(exist_ok=True)
    sdf_path = os.path.join(output_dir, f"{cid}.sdf")

    try:
        # 下载2D SDF文件（去掉record_type='3d'参数）
        pcp.download('SDF', sdf_path, overwrite=True, identifier=cid)

        # 检查文件是否成功创建
        if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            file_size = os.path.getsize(sdf_path)
            print(f"  ✓ 成功下载2D SDF: {sdf_path} ({file_size} bytes)")
            return sdf_path
        else:
            print(f"  ✗ 2D SDF文件创建失败")
            return None

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return None


def generate_2d_sdf_from_smiles(smiles, cid, output_dir="sdf_files"):
    """
    从SMILES生成2D SDF文件（备选方案）

    Args:
        smiles: SMILES字符串
        cid: 化合物CID
        output_dir: 输出目录

    Returns:
        str: SDF文件路径，失败时返回None
    """
    try:
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"    无效的SMILES字符串")
            return None

        # 添加氢原子
        mol = Chem.AddHs(mol)

        # 生成2D坐标（不进行3D嵌入）
        AllChem.Compute2DCoords(mol)

        # 保存为SDF文件
        Path(output_dir).mkdir(exist_ok=True)
        sdf_path = os.path.join(output_dir, f"{cid}.sdf")

        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

        # 检查文件是否成功创建
        if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            file_size = os.path.getsize(sdf_path)
            print(f"  ✓ 从SMILES生成2D SDF: {sdf_path} ({file_size} bytes)")
            return sdf_path
        else:
            print(f"  ✗ SDF文件创建失败")
            return None

    except Exception as e:
        print(f"  ✗ 生成2D SDF失败: {e}")
        return None


def batch_download_2d_sdf(no_3d_cids, excel_file="ALLdrug-smiles.xlsx", output_dir="sdf_files", method="pubchem",
                          delay=0.3):
    """
    批量下载2D SDF文件

    Args:
        no_3d_cids: 需要下载2D SDF的CID列表
        excel_file: Excel文件路径（用于获取SMILES）
        output_dir: SDF文件保存目录
        method: 下载方法 ("pubchem" 或 "smiles" 或 "hybrid")
        delay: 请求间延迟（秒）

    Returns:
        dict: 下载结果统计
    """
    print(f"开始批量下载2D SDF文件...")
    print(f"方法: {method}")
    print(f"CID数量: {len(no_3d_cids)}")

    # 读取SMILES数据（如果需要的话）
    cid_smiles_map = {}
    if method in ["smiles", "hybrid"]:
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            cid_smiles_map = dict(zip(df['CID'], df['SMILES']))
            print(f"已读取SMILES数据: {len(cid_smiles_map)}条")
        else:
            print(f"Excel文件不存在: {excel_file}")
            return None

    # 统计变量
    success_count = 0
    failed_cids = []

    # 逐个处理CID
    for idx, cid in enumerate(no_3d_cids):
        print(f"\n处理 {idx + 1}/{len(no_3d_cids)}: CID {cid}")

        success = False

        if method == "pubchem":
            # 方法1: 直接从PubChem下载2D SDF
            result = download_2d_sdf_pubchem(cid, output_dir)
            if result:
                success = True

        elif method == "smiles":
            # 方法2: 从SMILES生成2D SDF
            if cid in cid_smiles_map:
                smiles = cid_smiles_map[cid]
                print(f"  SMILES: {smiles}")
                result = generate_2d_sdf_from_smiles(smiles, cid, output_dir)
                if result:
                    success = True
            else:
                print(f"  ✗ 找不到CID {cid}的SMILES")

        elif method == "hybrid":
            # 方法3: 混合策略 - 优先PubChem，失败时用SMILES
            result = download_2d_sdf_pubchem(cid, output_dir)
            if result:
                success = True
            else:
                print(f"  PubChem下载失败，尝试从SMILES生成...")
                if cid in cid_smiles_map:
                    smiles = cid_smiles_map[cid]
                    print(f"  SMILES: {smiles}")
                    result = generate_2d_sdf_from_smiles(smiles, cid, output_dir)
                    if result:
                        success = True
                        print(f"  ✓ 备用方法成功")
                else:
                    print(f"  ✗ 也找不到SMILES数据")

        if success:
            success_count += 1
        else:
            failed_cids.append(cid)

        # 添加延迟
        if delay > 0 and method in ["pubchem", "hybrid"]:
            time.sleep(delay)

    # 统计结果
    result = {
        'total': len(no_3d_cids),
        'success': success_count,
        'failed': len(failed_cids),
        'success_rate': success_count / len(no_3d_cids) * 100,
        'failed_cids': failed_cids,
        'method': method
    }

    print_summary(result)
    return result


def print_summary(result):
    """
    打印下载结果摘要
    """
    print("\n" + "=" * 60)
    print("2D SDF下载结果摘要:")
    print(f"使用方法: {result['method']}")
    print(f"总数量: {result['total']}")
    print(f"成功下载: {result['success']}")
    print(f"下载失败: {result['failed']}")
    print(f"成功率: {result['success_rate']:.1f}%")

    if result['failed_cids']:
        print(f"\n失败的CID: {result['failed_cids']}")
    else:
        print(f"\n✓ 所有CID都成功下载了2D SDF文件!")


def verify_sdf_files(cid_list, output_dir="sdf_files"):
    """
    验证SDF文件是否成功创建
    """
    print(f"\n验证SDF文件...")

    existing_files = []
    missing_files = []

    for cid in cid_list:
        sdf_path = os.path.join(output_dir, f"{cid}.sdf")
        if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
            file_size = os.path.getsize(sdf_path)
            existing_files.append((cid, file_size))
        else:
            missing_files.append(cid)

    print(f"✓ 存在的SDF文件: {len(existing_files)}个")
    print(f"✗ 缺失的SDF文件: {len(missing_files)}个")

    if missing_files:
        print(f"缺失的CID: {missing_files}")

    # 显示前几个文件的信息
    for i, (cid, size) in enumerate(existing_files[:5]):
        print(f"  {cid}.sdf ({size} bytes)")

    if len(existing_files) > 5:
        print(f"  ... 还有 {len(existing_files) - 5} 个文件")


# 使用示例
if __name__ == "__main__":
    # 需要下载2D SDF的CID列表
    no_3d_cids = [15, 544, 2767, 3101, 4158, 4943, 5291, 5361, 5950, 5978, 9444, 13342, 14888,
                  23924, 23939, 23950, 23973, 24288, 24497, 24759, 28486, 36314, 38904, 44164,
                  47725, 60699, 60780, 65348, 78260, 84046, 92727, 122634, 123596, 148124, 148177,
                  156413, 162282, 162859, 261004, 387447, 392622, 426756, 441203, 442111, 445434, 446378,
                  455658, 462382, 636397, 637511, 2733525, 3006531, 3028194, 5222465, 5280443, 5281691, 5284373,
                  5284616, 5310940, 5311497, 5352624, 5359596, 5360373, 5381226, 5460033, 5460341, 5462222, 6442177, 6857599,
                  6918289, 9800555, 9854073, 9887053, 9931953, 10339178, 11226684, 11228183, 11538455, 11556711, 13920603, 16131098,
                  16197265, 17754356, 23666112, 24978538, 25183872, 44424639, 46907787, 54608508, 56842234, 73425383, 76968809,
                  91820602, 92135919, 135565658, 138374198, 145994598, 157010069]

    # 选择下载方法：
    # "pubchem" - 直接从PubChem下载2D SDF
    # "smiles" - 从SMILES生成2D SDF
    # "hybrid" - 混合策略，优先PubChem，失败时用SMILES

    # 推荐使用hybrid方法，最全面
    result = batch_download_2d_sdf(
        no_3d_cids,
        excel_file="ALLdrug-smiles.xlsx",
        method="hybrid",  # 推荐使用混合策略
        delay=0.3
    )

    # 验证下载结果
    if result:
        print(f"\n开始验证...")
        verify_sdf_files(no_3d_cids)

        if result['success'] == len(no_3d_cids):
            print(f"\n🎉 完美！所有79个CID都成功获得了2D SDF文件！")
            print(f"现在你有：")
            print(f"  - 355个3D SDF文件（之前下载的）")
            print(f"  - 79个2D SDF文件（刚刚下载的）")
            print(f"  - 总共434个SDF文件！")
        else:
            print(f"\n还有 {result['failed']} 个CID需要处理")