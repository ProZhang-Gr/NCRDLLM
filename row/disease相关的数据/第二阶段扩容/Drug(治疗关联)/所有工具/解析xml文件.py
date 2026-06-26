import xml.etree.ElementTree as ET
import pandas as pd
import os
from pathlib import Path


def extract_drugbank_data(xml_file_path):
    """
    从DrugBank XML文件中提取药物信息
    """
    print("=" * 60)
    print("DrugBank XML数据提取")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(xml_file_path):
        print(f"错误: 找不到文件 {xml_file_path}")
        return None

    print(f"正在解析XML文件: {xml_file_path}")

    try:
        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 获取命名空间
        namespace = {'db': 'http://www.drugbank.ca'}

        # 存储提取的数据
        drug_data = []

        # 查找所有药物条目
        drugs = root.findall('.//db:drug', namespace)
        print(f"找到 {len(drugs)} 个药物条目")

        for i, drug in enumerate(drugs, 1):
            try:
                # 提取drugbank-id (primary=true的那个)
                drugbank_id_elem = drug.find('.//db:drugbank-id[@primary="true"]', namespace)
                drugbank_id = drugbank_id_elem.text if drugbank_id_elem is not None else 'N/A'

                # 提取药物名称
                name_elem = drug.find('db:name', namespace)
                name = name_elem.text if name_elem is not None else 'N/A'

                # 提取适应症
                indication_elem = drug.find('db:indication', namespace)
                indication = indication_elem.text if indication_elem is not None else 'N/A'

                # 清理indication文本（去除多余空白和换行）
                if indication != 'N/A' and indication:
                    # 去除HTML实体和多余空白
                    indication = indication.replace('&#13;', ' ').replace('\n', ' ')
                    indication = ' '.join(indication.split())

                drug_data.append({
                    'drugbank-id': drugbank_id,
                    'name': name,
                    'indication': indication
                })

                # 进度显示
                if i % 100 == 0:
                    print(f"已处理 {i}/{len(drugs)} 个药物...")

            except Exception as e:
                print(f"处理第 {i} 个药物时出错: {e}")
                continue

        print(f"成功提取 {len(drug_data)} 个药物的信息")

        # 转换为DataFrame
        df = pd.DataFrame(drug_data)

        # 生成输出文件名
        xml_file_name = Path(xml_file_path).stem
        output_file = f"drugbank_extracted_{xml_file_name}.xlsx"

        # 保存为Excel文件
        df.to_excel(output_file, index=False, sheet_name='DrugBank_Data')

        # 统计信息
        print("\n" + "=" * 60)
        print("提取完成!")
        print("=" * 60)
        print(f"输出文件: {output_file}")
        print(f"总药物数量: {len(df)}")
        print(f"有drugbank-id的药物: {len(df[df['drugbank-id'] != 'N/A'])}")
        print(f"有名称的药物: {len(df[df['name'] != 'N/A'])}")
        print(f"有适应症的药物: {len(df[df['indication'] != 'N/A'])}")

        # 显示前几行预览
        print(f"\n前5行数据预览:")
        for _, row in df.head().iterrows():
            indication_short = (row['indication'][:80] + '...') if len(str(row['indication'])) > 80 else row[
                'indication']
            print(f"ID: {row['drugbank-id']}")
            print(f"Name: {row['name']}")
            print(f"Indication: {indication_short}")
            print("-" * 40)

        # 统计indication的情况
        valid_indications = df[(df['indication'] != 'N/A') & (df['indication'].str.strip() != '')]
        print(f"\n适应症统计:")
        print(f"有效适应症数量: {len(valid_indications)}")
        print(f"适应症覆盖率: {len(valid_indications) / len(df) * 100:.1f}%")

        return output_file

    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
        return None
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数 - 在这里指定XML文件路径
    """
    # 在这里修改为你的XML文件路径
    xml_file_path = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\Drug(治疗关联)\Drugbank收集的数据\full database.xml"  # 修改这里！！！

    print(f"处理文件: {xml_file_path}")

    # 开始提取
    result_file = extract_drugbank_data(xml_file_path)

    if result_file:
        print(f"\n数据提取成功! 结果保存在: {result_file}")
    else:
        print("\n数据提取失败!")


if __name__ == "__main__":
    main()