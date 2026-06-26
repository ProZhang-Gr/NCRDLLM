import requests
import pandas as pd
import os
from typing import List


def get_smiles_batch_api(cids: List[int], chunk_size: int = 100) -> str:
    """
    使用PubChem批量API获取SMILES数据

    Args:
        cids (List[int]): CID列表
        chunk_size (int): 每次请求的CID数量上限

    Returns:
        str: CSV格式的响应文本
    """
    # PubChem批量查询API
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES,title/CSV'
    headers = {"content-type": "application/x-www-form-urlencoded"}

    all_results = []
    header_written = False

    # 分批处理，避免URL过长或请求过大
    for i in range(0, len(cids), chunk_size):
        chunk = cids[i:i + chunk_size]
        cids_str = ','.join(map(str, chunk))

        print(f"正在处理第 {i + 1}-{min(i + chunk_size, len(cids))} 个CID...")

        data = {"cid": cids_str}

        try:
            response = requests.post(url, data=data, headers=headers, timeout=30)
            response.raise_for_status()

            # 处理响应文本
            lines = response.text.strip().split('\n')

            if not header_written:
                # 第一次添加标题行
                all_results.extend(lines)
                header_written = True
            else:
                # 后续只添加数据行（跳过标题行）
                all_results.extend(lines[1:])

        except requests.exceptions.RequestException as e:
            print(f"请求第 {i + 1}-{min(i + chunk_size, len(cids))} 批CID时出错: {e}")
            continue

    return '\n'.join(all_results)


def read_cids_from_excel(file_path: str, cid_column: str = 'CID', sheet_name=0) -> List[int]:
    """
    从Excel文件读取CID列表

    Args:
        file_path (str): Excel文件路径
        cid_column (str): CID列名
        sheet_name: 工作表名称或索引

    Returns:
        List[int]: CID列表
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"成功读取Excel文件: {file_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 检查CID列
        if cid_column not in df.columns:
            print(f"警告: 未找到列 '{cid_column}'，使用第一列")
            cid_column = df.columns[0]

        # 提取并清理CID数据
        cids = df[cid_column].dropna().astype(int).tolist()
        print(f"读取到 {len(cids)} 个CID")

        return cids

    except Exception as e:
        print(f"读取Excel文件出错: {e}")
        return []


def save_results(csv_text: str, output_file: str = 'cid_smiles_results.csv'):
    """
    保存结果到CSV文件并显示统计信息

    Args:
        csv_text (str): CSV格式的数据
        output_file (str): 输出文件名
    """
    # 保存CSV文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(csv_text)

    print(f"结果已保存到: {output_file}")

    # 分析结果
    try:
        df = pd.read_csv(output_file)
        print(f"\n结果统计:")
        print(f"总记录数: {len(df)}")

        if 'CanonicalSMILES' in df.columns:
            valid_smiles = df['CanonicalSMILES'].notna().sum()
            print(f"成功获取SMILES: {valid_smiles}")
            print(f"获取失败: {len(df) - valid_smiles}")

        # 显示前几条结果
        print(f"\n前5条结果预览:")
        print(df.head().to_string(index=False))

        # 同时保存Excel格式
        excel_file = output_file.replace('.csv', '.xlsx')
        df.to_excel(excel_file, index=False)
        print(f"同时保存Excel格式: {excel_file}")

    except Exception as e:
        print(f"分析结果时出错: {e}")


def main():
    """主函数 - 演示完整流程"""

    # 方法1: 直接使用CID列表（类似原始代码）
    print("方法1: 直接使用CID列表")
    print("=" * 50)

    # 示例CID列表
    cids_example = [44259, 65028, 2764, 15, 241, 544, 702, 712, 784, 790]

    print(f"处理CID: {cids_example}")
    csv_result = get_smiles_batch_api(cids_example)
    print("API响应:")
    print(csv_result)

    # 保存结果
    save_results(csv_result, 'method1_results.csv')

    print("\n" + "=" * 70 + "\n")

    # 方法2: 从Excel文件读取CID
    print("方法2: 从Excel文件读取CID")
    print("=" * 50)

    excel_file = r"D:\Desktop\CDLLM\项目进行时\row\各种工具\把drug的CID转化为smiles\ALLdrug-smiles.xlsx"  # 你的Excel文件路径

    if os.path.exists(excel_file):
        # 从Excel读取CID
        cids_from_excel = read_cids_from_excel(excel_file)

        if cids_from_excel:
            print(f"从Excel读取到 {len(cids_from_excel)} 个CID")

            # 批量获取SMILES
            csv_result = get_smiles_batch_api(cids_from_excel)

            # 保存结果
            save_results(csv_result, 'excel_results.csv')
        else:
            print("从Excel文件读取CID失败")
    else:
        print(f"Excel文件 '{excel_file}' 不存在")
        print("创建一个示例Excel文件进行演示...")

        # 创建示例Excel文件
        sample_df = pd.DataFrame({
            'CID': [216416, 65028, 2764, 15, 241, 544, 702, 712, 784, 790, 807, 896],
            'Note': ['示例化合物'] * 12
        })
        sample_df.to_excel('sample_compounds.xlsx', index=False)
        print("已创建示例文件: sample_compounds.xlsx")


# 原始代码的改进版本
def original_method_improved():
    """
    基于你提供的原始代码的改进版本
    """
    print("原始方法的改进版本:")
    print("=" * 30)

    # 多个CID (类似你的原始代码)
    cids = '216416,65028,2764,15,241'

    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES,title/CSV'
    headers = {"content-type": "application/x-www-form-urlencoded"}
    data = {"cid": cids}

    try:
        res = requests.post(url, data=data, headers=headers, timeout=30)
        res.raise_for_status()  # 检查HTTP错误

        print("API响应:")
        print(res.text)

        # 改进的文件写入 (使用with语句自动关闭文件)
        output_file = 'result_improved.csv'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(res.text)
        # 注意: 不需要显式调用f.close()，with语句会自动处理

        print(f"结果已保存到: {output_file}")

        # 额外的数据分析
        df = pd.read_csv(output_file)
        print(f"获取到 {len(df)} 条记录")

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
    except Exception as e:
        print(f"处理出错: {e}")


if __name__ == "__main__":
    # 运行改进的原始方法
    original_method_improved()
    print("\n" + "=" * 70 + "\n")

    # 运行完整的解决方案
    main()

# 安装依赖命令:
# pip install requests pandas openpyxl