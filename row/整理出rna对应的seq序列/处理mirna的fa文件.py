def process_fasta_file(input_file_path, output_file_path):
    """
    处理FASTA文件，格式化序列标识符和保留序列数据

    参数:
    input_file_path: 输入的.fa文件路径
    output_file_path: 输出的处理后文件路径
    """

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:

            for line in input_file:
                line = line.strip()  # 去除行首行尾空白字符

                if line.startswith('>'):  # 处理标识符行
                    # 分割标识符，获取第一部分
                    identifier = line.split()[0]  # 获取>cel-miR-59-3p部分

                    # 找到第三个"-"的位置并截取
                    parts = identifier.split('-')
                    if len(parts) >= 3:
                        # 保留前两个部分，如>cel-miR-59
                        processed_identifier = '-'.join(parts[:3])
                        output_file.write(processed_identifier + '\n')
                    else:
                        # 如果没有足够的"-"，保留原标识符
                        output_file.write(identifier + '\n')

                elif line:  # 处理序列行（非空行且不以>开头）
                    output_file.write(line + '\n')
                # 空行直接跳过


def process_fasta_string(fasta_content):
    """
    处理FASTA格式的字符串内容

    参数:
    fasta_content: FASTA格式的字符串

    返回:
    处理后的FASTA格式字符串
    """

    lines = fasta_content.strip().split('\n')
    processed_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith('>'):  # 处理标识符行
            identifier = line.split()[0]  # 获取>cel-miR-59-3p部分

            # 找到第三个"-"的位置并截取
            parts = identifier.split('-')
            if len(parts) >= 3:
                # 保留前三个部分，如>cel-miR-59
                processed_identifier = '-'.join(parts[:3])
                processed_lines.append(processed_identifier)
            else:
                processed_lines.append(identifier)

        elif line:  # 处理序列行
            processed_lines.append(line)

    return '\n'.join(processed_lines)


# 使用示例
if __name__ == "__main__":

    process_fasta_file(r'D:\Desktop\CDLLM\ing\row\整理出rna对应的seq序列\mature.fa', 'output.fa')

    # 示例2: 处理字符串内容
    sample_fasta = """
>cel-miR-59-3p MIMAT0000031 Caenorhabditis elegans miR-59-3p
UCGAAUCGUUUAUCAGGAUGAUG
>cel-miR-60-5p MIMAT0015102 Caenorhabditis elegans miR-60-5p
AACUGGAAGAGUGCCAUAAAAUC
>cel-miR-60-3p MIMAT0000032 Caenorhabditis elegans miR-60-3p
UAUUAUGCACAUUUUCUAGUUCA
    """

    result = process_fasta_string(sample_fasta)
    print("处理结果:")
    print(result)

    print("\n" + "=" * 50 + "\n")

    # 逐行展示处理过程
    lines = sample_fasta.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            identifier = line.split()[0]
            parts = identifier.split('-')
            if len(parts) >= 3:
                processed = '-'.join(parts[:3])
                print(f"原始: {line}")
                print(f"处理后: {processed}")
                print()