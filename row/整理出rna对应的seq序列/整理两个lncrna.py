import re
from collections import defaultdict


def parse_fasta_files(ncrna_file, cdna_file, output_file):
    """
    解析ncRNA和cDNA FASTA文件，按ENSG ID合并序列

    参数:
    ncrna_file: ncRNA FASTA文件路径
    cdna_file: cDNA FASTA文件路径
    output_file: 输出文件路径
    """

    # 存储每个ENSG对应的所有序列
    ensg_sequences = defaultdict(list)

    def process_fasta_file(file_path, file_type):
        """处理单个FASTA文件"""
        print(f"正在处理 {file_type} 文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            current_ensg = None
            current_sequence = ""

            for line in f:
                line = line.strip()

                if line.startswith('>'):
                    # 如果之前有序列，先保存
                    if current_ensg and current_sequence:
                        ensg_sequences[current_ensg].append(current_sequence)

                    # 提取ENSG ID
                    match = re.search(r'gene:(ENSG\d+)', line)
                    if match:
                        current_ensg = match.group(1)
                        current_sequence = ""
                    else:
                        current_ensg = None
                        current_sequence = ""

                elif current_ensg:  # 如果找到了ENSG，收集序列
                    current_sequence += line

            # 处理最后一个序列
            if current_ensg and current_sequence:
                ensg_sequences[current_ensg].append(current_sequence)

    # 处理两个文件
    process_fasta_file(ncrna_file, "ncRNA")
    process_fasta_file(cdna_file, "cDNA")

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        print(f"正在写入输出文件: {output_file}")

        # 按ENSG ID排序
        for ensg_id in sorted(ensg_sequences.keys()):
            sequences = ensg_sequences[ensg_id]

            # 写入ENSG标题
            f.write(f">{ensg_id}\n")

            # 合并所有转录本的序列
            merged_sequence = "".join(sequences)

            # 按行写入序列（每行60个字符，FASTA标准格式）
            for i in range(0, len(merged_sequence), 60):
                f.write(merged_sequence[i:i + 60] + "\n")

    # 统计信息
    total_genes = len(ensg_sequences)
    total_transcripts = sum(len(seqs) for seqs in ensg_sequences.values())

    print(f"处理完成！")
    print(f"共找到 {total_genes} 个ENSG基因")
    print(f"共处理 {total_transcripts} 个转录本")
    print(f"结果已保存到: {output_file}")

    return ensg_sequences


def analyze_results(ensg_sequences):
    """分析处理结果"""
    print("\n=== 处理结果分析 ===")

    # 统计每个基因的转录本数量
    transcript_counts = {ensg: len(seqs) for ensg, seqs in ensg_sequences.items()}

    # 显示有多个转录本的基因
    multi_transcript_genes = {k: v for k, v in transcript_counts.items() if v > 1}

    if multi_transcript_genes:
        print(f"有多个转录本的基因 ({len(multi_transcript_genes)} 个):")
        for ensg, count in sorted(multi_transcript_genes.items()):
            print(f"  {ensg}: {count} 个转录本")

    # 序列长度统计
    sequence_lengths = {}
    for ensg, sequences in ensg_sequences.items():
        total_length = sum(len(seq) for seq in sequences)
        sequence_lengths[ensg] = total_length

    if sequence_lengths:
        max_length = max(sequence_lengths.values())
        min_length = min(sequence_lengths.values())
        avg_length = sum(sequence_lengths.values()) / len(sequence_lengths)

        print(f"\n序列长度统计:")
        print(f"  最长序列: {max_length} bp")
        print(f"  最短序列: {min_length} bp")
        print(f"  平均长度: {avg_length:.1f} bp")


# 使用示例
if __name__ == "__main__":
    # 指定文件路径
    ncrna_file = "Homo_sapiens.GRCh38.ncrna.fa"  # ncRNA文件路径
    cdna_file = "Homo_sapiens.GRCh38.cdna.all.fa"  # cDNA文件路径
    output_file = "merged_sequences_by_ENSG.fa"  # 输出文件路径

    try:
        # 执行合并处理
        ensg_sequences = parse_fasta_files(ncrna_file, cdna_file, output_file)

        # 分析结果
        analyze_results(ensg_sequences)

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


# 如果只想处理特定的ENSG列表
def process_specific_genes(ncrna_file, cdna_file, target_genes, output_file):
    """只处理指定的ENSG基因列表"""
    target_set = set(target_genes)

    ensg_sequences = defaultdict(list)

    def process_file_selective(file_path):
        with open(file_path, 'r') as f:
            current_ensg = None
            current_sequence = ""

            for line in f:
                line = line.strip()

                if line.startswith('>'):
                    if current_ensg and current_ensg in target_set and current_sequence:
                        ensg_sequences[current_ensg].append(current_sequence)

                    match = re.search(r'gene:(ENSG\d+)', line)
                    if match and match.group(1) in target_set:
                        current_ensg = match.group(1)
                        current_sequence = ""
                    else:
                        current_ensg = None

                elif current_ensg:
                    current_sequence += line

            if current_ensg and current_ensg in target_set and current_sequence:
                ensg_sequences[current_ensg].append(current_sequence)

    process_file_selective(ncrna_file)
    process_file_selective(cdna_file)

    # 写入结果
    with open(output_file, 'w') as f:
        for ensg_id in sorted(ensg_sequences.keys()):
            f.write(f">{ensg_id}\n")
            merged_sequence = "".join(ensg_sequences[ensg_id])
            for i in range(0, len(merged_sequence), 60):
                f.write(merged_sequence[i:i + 60] + "\n")

    print(f"处理了 {len(ensg_sequences)} 个指定基因")
    return ensg_sequences

# 使用特定基因列表的示例：
# target_genes = ['ENSG00000148346', 'ENSG00000232987']  # 你想要的基因列表
# process_specific_genes(ncrna_file, cdna_file, target_genes, "specific_genes.fa")