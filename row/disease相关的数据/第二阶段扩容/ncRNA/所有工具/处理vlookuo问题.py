import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import sys


def copy_excel_file(input_file, output_file):
    """
    复制Excel文件的所有内容到新文件
    保持原有格式和数值（不保留公式）

    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    """
    try:
        # 读取原始Excel文件的所有工作表
        excel_file = pd.ExcelFile(input_file)
        sheet_names = excel_file.sheet_names

        # 创建新的工作簿
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name in sheet_names:
                # 读取每个工作表（包括所有数据类型）
                df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)

                # 写入新文件
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        print(f"✅ 成功复制文件!")
        print(f"   输入文件: {input_file}")
        print(f"   输出文件: {output_file}")
        print(f"   复制了 {len(sheet_names)} 个工作表: {', '.join(sheet_names)}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


def copy_excel_with_formatting(input_file, output_file):
    """
    使用openpyxl复制Excel文件（保持更多格式）
    """
    try:
        # 打开原始工作簿
        wb = openpyxl.load_workbook(input_file, data_only=True)

        # 保存到新文件
        wb.save(output_file)

        print(f"✅ 成功复制文件（保持格式）!")
        print(f"   输入文件: {input_file}")
        print(f"   输出文件: {output_file}")
        print(f"   工作表: {', '.join(wb.sheetnames)}")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{input_file}'")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 方法1: 基本复制（推荐用于解决VLOOKUP问题）
    input_file = r"D:\Desktop\CDLLM\ing\row\disease相关的数据\第二阶段扩容\Drug\所有工具\successful_drugs_cid.xlsx" # 替换为你的输入文件路径
    output_file = "复制文件.xlsx"  # 替换为你的输出文件路径

    print("方法1: 基本复制（将公式转换为数值）")
    copy_excel_file(input_file, output_file)

    print("\n" + "=" * 50 + "\n")

    # 方法2: 保持格式复制
    output_file2 = "复制文件_保持格式.xlsx"
    print("方法2: 保持格式复制")
    copy_excel_with_formatting(input_file, output_file2)

# 命令行使用示例:
# python copy_excel.py 输入文件.xlsx 输出文件.xlsx