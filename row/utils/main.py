import pandas as pd
import os
from pathlib import Path


def find_missing_cid_files(xlsx_file_path, image_folder_path):
    """
    查找在图片文件夹中存在但在Excel文件CID列中不存在的文件

    参数:
    xlsx_file_path: Excel文件路径
    image_folder_path: 图片文件夹路径
    """

    # 读取Excel文件
    try:
        df = pd.read_excel(xlsx_file_path)
        print(f"成功读取Excel文件，共有 {len(df)} 行数据")
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    # 获取CID列的所有值，转换为字符串集合
    if 'CID' not in df.columns:
        print("错误：Excel文件中没有找到'CID'列")
        print(f"可用的列名: {list(df.columns)}")
        return

    cid_set = set(str(cid) for cid in df['CID'].dropna())
    print(f"Excel中的CID数量: {len(cid_set)}")

    # 获取图片文件夹中的所有png文件
    image_folder = Path(image_folder_path)
    if not image_folder.exists():
        print(f"错误：图片文件夹不存在: {image_folder_path}")
        return

    # 获取所有.png文件的文件名（不包含扩展名）
    png_files = list(image_folder.glob("*.png"))
    file_ids = set(file.stem for file in png_files)  # stem获取不含扩展名的文件名
    print(f"文件夹中的PNG文件数量: {len(png_files)}")

    # 找出在文件夹中存在但在CID列中不存在的文件
    missing_in_excel = file_ids - cid_set

    # 找出在CID列中存在但在文件夹中不存在的文件
    missing_in_folder = cid_set - file_ids

    # 输出结果
    print("\n" + "=" * 50)
    print("统计结果:")
    print("=" * 50)

    if missing_in_excel:
        print(f"\n在文件夹中存在但在Excel CID列中缺失的文件 ({len(missing_in_excel)} 个):")
        for file_id in sorted(missing_in_excel, key=lambda x: int(x) if x.isdigit() else float('inf')):
            print(f"  {file_id}.png")
    else:
        print("\n✅ 所有图片文件都在Excel的CID列中有对应记录")

    if missing_in_folder:
        print(f"\n在Excel CID列中存在但在文件夹中缺失的图片 ({len(missing_in_folder)} 个):")
        for cid in sorted(missing_in_folder, key=lambda x: int(x) if x.isdigit() else float('inf')):
            print(f"  {cid}.png")
    else:
        print("\n✅ Excel CID列中的所有记录都有对应的图片文件")

    print(f"\n总结:")
    print(f"  Excel中的CID记录: {len(cid_set)}")
    print(f"  文件夹中的PNG文件: {len(png_files)}")
    print(f"  缺失Excel记录的文件: {len(missing_in_excel)}")
    print(f"  缺失图片文件的记录: {len(missing_in_folder)}")

    return {
        'missing_in_excel': missing_in_excel,
        'missing_in_folder': missing_in_folder,
        'total_cids': len(cid_set),
        'total_files': len(png_files)
    }


# 使用示例
if __name__ == "__main__":
    # 请修改为你的实际文件路径
    xlsx_path = r"D:\Desktop\1.xlsx"  # 替换为你的Excel文件路径
    image_folder = r"D:\Desktop\CDLLM\ing\official\drugimages"

    result = find_missing_cid_files(xlsx_path, image_folder)