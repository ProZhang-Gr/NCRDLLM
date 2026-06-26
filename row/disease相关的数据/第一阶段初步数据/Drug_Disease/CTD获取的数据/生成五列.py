import requests
import pandas as pd
import time
from urllib.parse import quote
import os


def get_cid_from_pubchem(chemical_name):
    """通过PubChem API根据化合物名称获取CID"""
    base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON'

    try:
        encoded_name = quote(chemical_name)
        url = base_url.format(encoded_name)
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                return data['IdentifierList']['CID'][0]
        return None

    except Exception as e:
        print(f"查询CID出错 {chemical_name}: {str(e)}")
        return None


def main():
    # 文件配置
    input_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\Drug_Disease\CTD获取的数据\failed_chemicals_records.csv'
    output_file = 'ctd_with_cid_results.xlsx'
    progress_file = 'cid_progress.txt'

    print("=" * 60)
    print("CTD数据处理 - 化合物名称转CID")
    print("=" * 60)

    # 检查文件
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    # 读取CTD数据
    print("读取CTD文件...")
    try:
        # 首先检查文件的实际分隔符
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"第一行内容: {first_line[:100]}...")
            print(f"Tab数量: {first_line.count(chr(9))}")
            print(f"逗号数量: {first_line.count(',')}")

            # 检测分隔符
            if first_line.count('\t') > 0:
                separator = '\t'
                print("使用Tab分隔符")
            elif first_line.count(',') > first_line.count('\t'):
                separator = ','
                print("使用逗号分隔符")
            else:
                separator = '\t'  # 强制使用Tab
                print("强制使用Tab分隔符")

        # 使用检测到的分隔符读取文件
        df = pd.read_csv(input_file, sep=separator, encoding='utf-8')
        print(f"成功读取 {len(df)} 条记录")
        print(f"列数: {len(df.columns)}")
        print(f"前5个列名: {list(df.columns)[:5]}")

        # 如果只有一列，说明分隔符错误，强制重试
        if len(df.columns) == 1:
            print("分隔符错误，重试使用Tab分隔符...")
            df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
            print(f"重试后列数: {len(df.columns)}")
            print(f"重试后前5个列名: {list(df.columns)[:5]}")

        # 检查必需的列是否存在
        required_cols = ['ChemicalName', 'ChemicalID', 'DiseaseName', 'DiseaseID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"错误: 缺少必需的列: {missing_cols}")
            print(f"实际列名: {list(df.columns)}")
            return

        # 显示前几行数据
        print("\n数据预览:")
        print(df[required_cols].head(3).to_string(index=False))

    except Exception as e:
        print(f"读取文件出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 提取唯一化合物
    print(f"\n提取唯一化合物...")
    unique_chemicals = df['ChemicalName'].unique()
    print(f"唯一化合物数量: {len(unique_chemicals)}")

    # 加载已完成的进度
    completed_cids = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if '\t' in line:
                        name, cid = line.strip().split('\t', 1)
                        completed_cids[name] = cid if cid != 'N/A' else None
            print(f"加载已完成的查询: {len(completed_cids)} 个")
        except Exception as e:
            print(f"读取进度文件出错: {e}")

    # 查询CID
    print(f"\n开始查询PubChem CID...")
    print("=" * 60)

    chemical_to_cid = {}
    success_count = 0

    for i, chemical_name in enumerate(unique_chemicals, 1):
        # 跳过已完成的
        if chemical_name in completed_cids:
            chemical_to_cid[chemical_name] = completed_cids[chemical_name]
            if completed_cids[chemical_name]:
                success_count += 1
            continue

        print(f"[{i}/{len(unique_chemicals)}] 查询: {chemical_name[:50]}{'...' if len(chemical_name) > 50 else ''}")

        cid = get_cid_from_pubchem(chemical_name)
        chemical_to_cid[chemical_name] = cid

        # 保存进度
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{chemical_name}\t{cid if cid else 'N/A'}\n")

        if cid:
            success_count += 1
            print(f"  -> 找到CID: {cid}")
        else:
            print(f"  -> 未找到")

        # 延迟和进度统计
        time.sleep(0.3)
        if i % 100 == 0:
            print(f"\n*** 进度统计 [{i}/{len(unique_chemicals)}] ***")
            print(f"成功: {success_count}, 成功率: {success_count / i * 100:.1f}%\n")

    # 构建最终结果
    print("\n生成最终结果...")

    # 为原始数据添加CID列
    df['CID'] = df['ChemicalName'].map(lambda x: chemical_to_cid.get(x, 'N/A'))

    # 选择需要的5列
    result_df = df[['CID', 'ChemicalName', 'ChemicalID', 'DiseaseName', 'DiseaseID']].copy()
    result_df.rename(columns={'ChemicalID': 'MESHID'}, inplace=True)

    # 保存Excel文件
    result_df.to_excel(output_file, index=False, sheet_name='CTD_with_CID')

    # 最终统计
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"输出文件: {output_file}")
    print(f"总记录数: {len(result_df)}")
    print(f"唯一化合物数: {len(unique_chemicals)}")
    print(f"成功获取CID: {success_count}")
    print(f"CID成功率: {success_count / len(unique_chemicals) * 100:.1f}%")

    valid_records = len(result_df[result_df['CID'] != 'N/A'])
    print(f"包含有效CID的记录数: {valid_records}")
    print(f"有效记录占比: {valid_records / len(result_df) * 100:.1f}%")

    # 预览结果
    print(f"\n最终结果预览:")
    print(result_df.head(3).to_string(index=False))

    # 保存失败列表
    failed_chemicals = [name for name, cid in chemical_to_cid.items() if cid is None]
    if failed_chemicals:
        with open('failed_chemicals.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed_chemicals))
        print(f"\n无法获取CID的化合物 ({len(failed_chemicals)}个) 已保存到: failed_chemicals.txt")


if __name__ == "__main__":
    main()