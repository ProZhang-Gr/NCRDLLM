import os
import pandas as pd
import pubchempy as pcp
from pathlib import Path


def batch_download_sdf_pubchempy(excel_file, cid_column="CID", output_dir="sdf_files"):
    """
    使用pubchempy批量下载3D构象SDF文件

    Args:
        excel_file: Excel文件路径 (ALLdrug-smiles.xlsx)
        cid_column: CID列名
        output_dir: SDF文件保存目录

    Returns:
        dict: 下载结果统计
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    if cid_column not in df.columns:
        raise ValueError(f"列 '{cid_column}' 不存在于Excel文件中")

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    success_count = 0
    failed_cids = []
    no_3d_cids = []

    total_count = len(df)

    print(f"开始下载 {total_count} 个化合物的3D构象SDF文件...")

    for idx, cid in enumerate(df[cid_column]):
        print(f"正在处理 {idx + 1}/{total_count}: CID {cid}")

        # 构建SDF文件路径
        sdf_path = os.path.join(output_dir, f"{cid}.sdf")

        try:
            # 下载3D构象SDF文件
            pcp.download('SDF', sdf_path, overwrite=True, identifier=cid, record_type='3d')
            success_count += 1
            print(f"  ✓ 成功下载: {sdf_path}")

        except pcp.NotFoundError as e:
            print(f"  ✗ No 3d Conformer for CID {cid}")
            no_3d_cids.append(cid)

        except Exception as e:
            print(f"  ✗ 下载失败 CID {cid}: {e}")
            failed_cids.append(cid)

    # 统计结果
    result = {
        'total': total_count,
        'success': success_count,
        'no_3d': len(no_3d_cids),
        'failed': len(failed_cids),
        'no_3d_cids': no_3d_cids,
        'failed_cids': failed_cids
    }

    print("\n" + "=" * 50)
    print("下载完成统计:")
    print(f"总数量: {result['total']}")
    print(f"成功下载: {result['success']}")
    print(f"无3D构象: {result['no_3d']}")
    print(f"下载失败: {result['failed']}")

    if no_3d_cids:
        print(f"\n无3D构象的CID: {no_3d_cids}")

    if failed_cids:
        print(f"\n下载失败的CID: {failed_cids}")

    return result


def check_sdf_files(output_dir="sdf_files"):
    """
    检查已下载的SDF文件
    """
    if not os.path.exists(output_dir):
        print(f"目录不存在: {output_dir}")
        return

    sdf_files = [f for f in os.listdir(output_dir) if f.endswith('.sdf')]
    print(f"已下载的SDF文件数量: {len(sdf_files)}")

    # 显示前几个文件
    for i, filename in enumerate(sdf_files[:5]):
        file_path = os.path.join(output_dir, filename)
        file_size = os.path.getsize(file_path)
        print(f"  {filename} ({file_size} bytes)")

    if len(sdf_files) > 5:
        print(f"  ... 还有 {len(sdf_files) - 5} 个文件")


def download_single_cid(cid, output_dir="sdf_files"):
    """
    下载单个CID的SDF文件（用于测试）
    """
    Path(output_dir).mkdir(exist_ok=True)
    sdf_path = os.path.join(output_dir, f"{cid}.sdf")

    try:
        pcp.download('SDF', sdf_path, overwrite=True, identifier=cid, record_type='3d')
        print(f"成功下载 CID {cid} 的3D构象SDF文件")

        # 显示文件信息
        file_size = os.path.getsize(sdf_path)
        print(f"文件路径: {sdf_path}")
        print(f"文件大小: {file_size} bytes")

        return sdf_path

    except pcp.NotFoundError:
        print(f"CID {cid} 没有3D构象")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    # 先测试单个CID下载
    print("测试单个CID下载...")
    test_result = download_single_cid(529)  # 博客中的示例CID

    if test_result:
        print("\n单个下载测试成功，开始批量下载...")

        # 批量下载你的Excel文件中的所有CID
        excel_file = "ALLdrug-smiles.xlsx"
        result = batch_download_sdf_pubchempy(excel_file)

        # 检查下载结果
        print("\n检查下载的文件...")
        check_sdf_files()
    else:
        print("单个下载测试失败，请检查网络连接或CID")


# 如果只想下载你Excel中的前几个CID进行测试
def test_download_few(excel_file, n=5):
    """
    只下载前n个CID进行测试
    """
    df = pd.read_excel(excel_file)
    test_cids = df['CID'].head(n).tolist()

    print(f"测试下载前{n}个CID: {test_cids}")

    for cid in test_cids:
        download_single_cid(cid)

# 取消注释下面这行来测试前5个CID
# test_download_few("ALLdrug-smiles.xlsx", 5)y
