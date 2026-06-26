import pandas as pd
import os


def load_mesh_doid_mapping(mapping_file):
    """加载MESH ID到DOID的映射表"""

    print(f"加载映射文件: {mapping_file}")

    try:
        df_mapping = pd.read_excel(mapping_file)
        print(f"映射文件列名: {list(df_mapping.columns)}")
        print(f"映射记录数: {len(df_mapping)}")

        # 创建MESH ID到DOID的字典
        mesh_to_doid = {}

        for index, row in df_mapping.iterrows():
            mesh_id = row['MESH_ID']
            doid = row['DOID']
            disease_name = row['Disease_Name']

            mesh_to_doid[mesh_id] = {
                'DOID': doid,
                'Disease_Name': disease_name
            }

        print(f"成功加载 {len(mesh_to_doid)} 个MESH-DOID映射")

        # 显示映射示例
        print("\n映射示例:")
        count = 0
        for mesh_id, mapping in mesh_to_doid.items():
            if count < 5:
                print(f"  {mesh_id} -> {mapping['DOID']} ({mapping['Disease_Name']})")
                count += 1
            else:
                break

        return mesh_to_doid

    except Exception as e:
        print(f"加载映射文件出错: {e}")
        return {}


def convert_ctd_mesh_to_doid(ctd_file, output_file, mapping_dict):
    """将CTD文件中的DiseaseID从MESH ID转换为DOID"""

    print(f"\n开始转换CTD文件: {ctd_file}")

    try:
        # 读取CTD数据
        df_ctd = pd.read_excel(ctd_file)
        print(f"CTD文件行数: {len(df_ctd)}")
        print(f"CTD文件列名: {list(df_ctd.columns)}")

        # 检查必需的列
        if 'DiseaseID' not in df_ctd.columns:
            print(f"错误: 缺少DiseaseID列")
            return

        # 显示原始数据预览
        print(f"\n原始数据预览:")
        print(df_ctd.head(3).to_string(index=False))

        # 统计转换情况
        converted_count = 0
        not_found_count = 0
        conversion_log = []

        # 创建新的DataFrame副本
        df_converted = df_ctd.copy()

        # 遍历每行进行转换
        for index, row in df_ctd.iterrows():
            original_disease_id = row['DiseaseID']
            original_disease_name = row.get('DiseaseName', '')
            chemical_name = row.get('ChemicalName', '')

            # 从DiseaseID中提取MESH ID (去掉MESH:前缀)
            if isinstance(original_disease_id, str) and original_disease_id.startswith('MESH:'):
                mesh_id = original_disease_id.replace('MESH:', '').strip()

                # 检查MESH ID是否在映射表中
                if mesh_id in mapping_dict:
                    new_doid = mapping_dict[mesh_id]['DOID']
                    new_disease_name = mapping_dict[mesh_id]['Disease_Name']

                    # 更新数据
                    df_converted.at[index, 'DiseaseID'] = new_doid
                    df_converted.at[index, 'DiseaseName'] = new_disease_name

                    converted_count += 1

                    conversion_log.append({
                        'ChemicalName': chemical_name,
                        'Original_DiseaseID': original_disease_id,
                        'MESH_ID': mesh_id,
                        'Original_DiseaseName': original_disease_name,
                        'New_DOID': new_doid,
                        'New_DiseaseName': new_disease_name
                    })

                    # 显示前几个转换示例
                    if converted_count <= 10:
                        print(f"转换 {converted_count}: {original_disease_id} -> {new_doid} ({new_disease_name})")
                else:
                    not_found_count += 1
                    # 显示前几个未找到的MESH ID
                    if not_found_count <= 5:
                        print(f"未找到映射: {mesh_id} (来自 {original_disease_id})")
            else:
                # DiseaseID格式不正确
                not_found_count += 1

            # 每处理10万行显示进度
            if (index + 1) % 100000 == 0:
                print(f"已处理 {index + 1} 行，转换 {converted_count} 个，未找到 {not_found_count} 个")

        # 保存转换后的文件
        df_converted.to_excel(output_file, index=False)
        print(f"\n转换后的文件已保存: {output_file}")

        # 保存转换日志
        if conversion_log:
            log_df = pd.DataFrame(conversion_log)
            log_file = output_file.replace('.xlsx', '_conversion_log.xlsx')
            log_df.to_excel(log_file, index=False)
            print(f"转换日志已保存: {log_file}")

        # 统计报告
        print(f"\n" + "=" * 60)
        print("转换完成!")
        print("=" * 60)
        print(f"总记录数: {len(df_ctd)}")
        print(f"成功转换: {converted_count}")
        print(f"未找到映射: {not_found_count}")
        print(f"转换率: {converted_count / len(df_ctd) * 100:.1f}%")

        # 显示转换后的数据预览
        print(f"\n转换后数据预览:")
        print(df_converted.head(3).to_string(index=False))

        # 显示转换统计
        if conversion_log:
            print(f"\n转换示例 (前5个):")
            for i, log in enumerate(conversion_log[:5]):
                print(f"{i + 1}. {log['Original_DiseaseID']} -> {log['New_DOID']}")
                print(f"   疾病: {log['Original_DiseaseName']} -> {log['New_DiseaseName']}")

        return df_converted

    except Exception as e:
        print(f"转换文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_conversion(converted_df):
    """验证转换结果"""

    print(f"\n验证转换结果...")

    # 检查DOID格式
    doid_pattern = converted_df['DiseaseID'].str.contains('DOID:', na=False)
    valid_doid_count = doid_pattern.sum()

    print(f"有效DOID格式的记录数: {valid_doid_count}")
    print(f"DOID格式正确率: {valid_doid_count / len(converted_df) * 100:.1f}%")

    # 统计唯一疾病数
    unique_diseases = converted_df['DiseaseID'].nunique()
    unique_chemicals = converted_df['ChemicalName'].nunique() if 'ChemicalName' in converted_df.columns else 0

    print(f"唯一疾病数 (DOID): {unique_diseases}")
    if unique_chemicals > 0:
        print(f"唯一化合物数: {unique_chemicals}")


def main():
    # 文件配置
    mapping_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\Drug_Disease\CTD获取的数据\MESH_ID_DOID_Disease_Name.xlsx'  # MESH-DOID映射文件
    ctd_file = 'ctd_with_cid_results.xlsx'  # 待转换的CTD文件
    output_file = 'ctd_with_doid_results.xlsx'  # 转换后的输出文件

    print("=" * 70)
    print("CTD数据MESH ID转DOID转换器")
    print("=" * 70)

    # 检查文件存在性
    if not os.path.exists(mapping_file):
        print(f"错误: 找不到映射文件 {mapping_file}")
        print("请先运行MESH-DOID映射提取器生成映射文件")
        return

    if not os.path.exists(ctd_file):
        print(f"错误: 找不到CTD文件 {ctd_file}")
        print("请将CTD数据文件重命名为此文件名")
        return

    # 加载映射表
    mapping_dict = load_mesh_doid_mapping(mapping_file)

    if not mapping_dict:
        print("无法加载映射表，程序终止")
        return

    # 转换CTD数据
    converted_df = convert_ctd_mesh_to_doid(ctd_file, output_file, mapping_dict)

    if converted_df is not None:
        # 验证转换结果
        validate_conversion(converted_df)

        print(f"\n" + "=" * 70)
        print("转换完成! 现在你的CTD数据使用标准DOID标识符")
        print(f"输出文件: {output_file}")
        print("=" * 70)
    else:
        print("转换失败")


if __name__ == "__main__":
    main()