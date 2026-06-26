import pandas as pd
import os


def load_id_mapping(mapping_file):
    """加载新旧ID映射表，创建旧ID到新ID的字典"""

    print(f"加载ID映射文件: {mapping_file}")

    try:
        # 读取映射Excel文件
        df_mapping = pd.read_excel(mapping_file)
        print(f"映射文件列名: {list(df_mapping.columns)}")
        print(f"映射条目数: {len(df_mapping)}")

        # 创建旧ID到新ID的映射字典
        old_to_new = {}

        for index, row in df_mapping.iterrows():
            new_id = row['newID']

            # 遍历所有oldID列
            for col in df_mapping.columns:
                if col.startswith('oldID') and pd.notna(row[col]) and row[col] != '':
                    old_id = row[col].strip()
                    old_to_new[old_id] = new_id

        print(f"总共找到 {len(old_to_new)} 个旧ID需要更新")

        # 显示一些映射示例
        print("\n映射示例:")
        count = 0
        for old_id, new_id in old_to_new.items():
            if count < 5:
                print(f"  {old_id} -> {new_id}")
                count += 1
            else:
                break

        return old_to_new

    except Exception as e:
        print(f"加载映射文件出错: {e}")
        return {}


def update_doid_file(input_file, output_file, mapping_dict):
    """更新疾病-DOID文件中的旧DOID为新DOID"""

    print(f"\n开始更新DOID文件: {input_file}")

    try:
        # 读取待更新的文件
        df = pd.read_excel(input_file)
        print(f"原始文件行数: {len(df)}")
        print(f"原始文件列名: {list(df.columns)}")

        # 检查DOID列是否存在
        if 'DOID' not in df.columns:
            print("错误: 文件中没有找到'DOID'列")
            return

        # 显示原始数据预览
        print(f"\n原始数据预览:")
        print(df.head(3).to_string(index=False))

        # 统计更新情况
        updated_count = 0
        not_found_count = 0

        # 创建更新记录
        update_log = []

        # 遍历每行，检查并更新DOID
        for index, row in df.iterrows():
            original_doid = row['DOID']
            disease_name = row.get('Disease_Name', 'Unknown')

            # 检查当前DOID是否在旧ID映射中
            if original_doid in mapping_dict:
                new_doid = mapping_dict[original_doid]
                df.at[index, 'DOID'] = new_doid
                updated_count += 1

                update_log.append({
                    'Disease_Name': disease_name,
                    'Old_DOID': original_doid,
                    'New_DOID': new_doid
                })

                print(f"更新: {disease_name} | {original_doid} -> {new_doid}")
            else:
                not_found_count += 1

        # 保存更新后的文件
        df.to_excel(output_file, index=False)
        print(f"\n更新后的文件已保存: {output_file}")

        # 保存更新日志
        if update_log:
            log_df = pd.DataFrame(update_log)
            log_file = output_file.replace('.xlsx', '_update_log.xlsx')
            log_df.to_excel(log_file, index=False)
            print(f"更新日志已保存: {log_file}")

        # 统计报告
        print(f"\n" + "=" * 50)
        print("更新完成!")
        print("=" * 50)
        print(f"总记录数: {len(df)}")
        print(f"更新的DOID数: {updated_count}")
        print(f"未找到映射的DOID数: {not_found_count}")
        print(f"更新率: {updated_count / len(df) * 100:.1f}%")

        # 显示更新后的数据预览
        print(f"\n更新后数据预览:")
        print(df.head(3).to_string(index=False))

        # 显示更新的条目
        if update_log:
            print(f"\n更新的条目 (前10个):")
            for i, log in enumerate(update_log[:10]):
                print(f"{i + 1}. {log['Disease_Name']}: {log['Old_DOID']} -> {log['New_DOID']}")

        return df

    except Exception as e:
        print(f"更新文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # 文件配置
    mapping_file = 'doid_id_mapping.xlsx'  # 新旧ID映射文件
    input_file = r'D:\Desktop\CDLLM\ing\row\disease相关的数据\Disease_DOID_Map\Doid新旧map\DIsease_DOID.xlsx'  # 待更新的疾病-DOID文件
    output_file = 'disease_doid_list_updated.xlsx'  # 更新后的文件

    print("=" * 60)
    print("DOID更新器 - 将旧DOID替换为标准新DOID")
    print("=" * 60)

    # 检查文件存在性
    if not os.path.exists(mapping_file):
        print(f"错误: 找不到映射文件 {mapping_file}")
        print("请先运行DOID映射提取器生成映射文件")
        return

    if not os.path.exists(input_file):
        print(f"错误: 找不到待更新文件 {input_file}")
        print("请将你的疾病-DOID Excel文件重命名为此文件名")
        return

    # 加载新旧ID映射
    mapping_dict = load_id_mapping(mapping_file)

    if not mapping_dict:
        print("无法加载ID映射，程序终止")
        return

    # 更新DOID文件
    updated_df = update_doid_file(input_file, output_file, mapping_dict)

    if updated_df is not None:
        print(f"\n处理完成! 请查看输出文件: {output_file}")
    else:
        print("更新失败")


if __name__ == "__main__":
    main()