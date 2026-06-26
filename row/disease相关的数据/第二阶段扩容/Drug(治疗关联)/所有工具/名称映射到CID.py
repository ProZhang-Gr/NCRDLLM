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
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\Drug(治疗关联)\Drugbank收集的数据\DrugName_文本语言.xlsx" # 修改为你的xlsx文件路径
    output_file = 'drug_disease_with_cid_results.xlsx'
    progress_file = 'drug_cid_progress.txt'

    print("=" * 60)
    print("Drug-Disease数据处理 - 药物名称转CID")
    print("=" * 60)

    # 检查文件
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    # 读取Excel/CSV文件
    print("读取文件...")
    try:
        # 根据文件扩展名选择读取方式
        if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            df = pd.read_excel(input_file)
            print("使用Excel格式读取")
        else:
            # CSV文件处理逻辑保持不变
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.count('\t') > 0:
                    separator = '\t'
                elif first_line.count(',') > first_line.count('\t'):
                    separator = ','
                else:
                    separator = '\t'
            df = pd.read_csv(input_file, sep=separator, encoding='utf-8')
            print("使用CSV格式读取")

        print(f"成功读取 {len(df)} 条记录")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")

        # 检查必需的列是否存在
        required_cols = ['Drug name', 'Disease name']
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

    # 提取唯一药物
    print(f"\n提取唯一药物...")
    unique_drugs = df['Drug name'].dropna().unique()
    print(f"唯一药物数量: {len(unique_drugs)}")

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

    drug_to_cid = {}
    success_count = 0

    for i, drug_name in enumerate(unique_drugs, 1):
        # 跳过已完成的
        if drug_name in completed_cids:
            drug_to_cid[drug_name] = completed_cids[drug_name]
            if completed_cids[drug_name]:
                success_count += 1
            continue

        print(f"[{i}/{len(unique_drugs)}] 查询: {drug_name[:50]}{'...' if len(drug_name) > 50 else ''}")

        cid = get_cid_from_pubchem(drug_name)
        drug_to_cid[drug_name] = cid

        # 保存进度
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{drug_name}\t{cid if cid else 'N/A'}\n")

        if cid:
            success_count += 1
            print(f"  -> 找到CID: {cid}")
        else:
            print(f"  -> 未找到")

        # 延迟和进度统计
        time.sleep(0.3)
        if i % 50 == 0:
            print(f"\n*** 进度统计 [{i}/{len(unique_drugs)}] ***")
            print(f"成功: {success_count}, 成功率: {success_count / i * 100:.1f}%\n")

    # 构建最终结果
    print("\n生成最终结果...")

    # 为原始数据添加CID列
    df['CID'] = df['Drug name'].map(lambda x: drug_to_cid.get(x, 'N/A'))

    # 重新排列列的顺序
    result_df = df[['CID', 'Drug name', 'Disease name']].copy()

    # 保存Excel文件
    result_df.to_excel(output_file, index=False, sheet_name='Drug_Disease_with_CID')

    # 最终统计
    print("=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"输出文件: {output_file}")
    print(f"总记录数: {len(result_df)}")
    print(f"唯一药物数: {len(unique_drugs)}")
    print(f"成功获取CID: {success_count}")
    print(f"CID成功率: {success_count / len(unique_drugs) * 100:.1f}%")

    valid_records = len(result_df[result_df['CID'] != 'N/A'])
    print(f"包含有效CID的记录数: {valid_records}")
    print(f"有效记录占比: {valid_records / len(result_df) * 100:.1f}%")

    # 预览结果
    print(f"\n最终结果预览:")
    print(result_df.head(5).to_string(index=False))

    # 保存失败列表
    failed_drugs = [name for name, cid in drug_to_cid.items() if cid is None]
    if failed_drugs:
        with open('failed_drugs.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed_drugs))
        print(f"\n无法获取CID的药物 ({len(failed_drugs)}个) 已保存到: failed_drugs.txt")

    # 保存成功获取CID的药物列表
    success_drugs = {name: cid for name, cid in drug_to_cid.items() if cid is not None}
    if success_drugs:
        success_df = pd.DataFrame(list(success_drugs.items()), columns=['Drug_name', 'CID'])
        success_df.to_excel('successful_drugs_cid.xlsx', index=False)
        print(f"成功获取CID的药物 ({len(success_drugs)}个) 已保存到: successful_drugs_cid.xlsx")


if __name__ == "__main__":
    main()