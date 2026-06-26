import requests
import json
import time
from typing import Dict, List, Optional


def get_gene_transcripts(gene_id: str) -> Optional[List[str]]:
    """
    获取基因对应的所有转录本ID
    """
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{gene_id}?content-type=application/json;expand=1"

    try:
        response = requests.get(server + ext, headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            data = response.json()
            transcripts = []

            if 'Transcript' in data:
                for transcript in data['Transcript']:
                    transcripts.append(transcript['id'])

            return transcripts
        else:
            print(f"获取 {gene_id} 的转录本失败: {response.status_code}")
            return None

    except Exception as e:
        print(f"请求 {gene_id} 时出错: {e}")
        return None


def get_transcript_sequence(transcript_id: str) -> Optional[str]:
    """
    获取转录本的序列
    """
    server = "https://rest.ensembl.org"
    ext = f"/sequence/id/{transcript_id}?content-type=text/plain"

    try:
        response = requests.get(server + ext, headers={"Content-Type": "text/plain"})

        if response.status_code == 200:
            return response.text.strip()
        else:
            print(f"获取 {transcript_id} 序列失败: {response.status_code}")
            return None

    except Exception as e:
        print(f"请求 {transcript_id} 序列时出错: {e}")
        return None


def get_gene_info(gene_id: str) -> Optional[Dict]:
    """
    获取基因的详细信息（用于检查biotype）
    """
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{gene_id}?content-type=application/json"

    try:
        response = requests.get(server + ext, headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        print(f"获取 {gene_id} 信息时出错: {e}")
        return None


def fetch_missing_sequences_from_api(missing_genes: List[str], output_file: str = "api_sequences.fa"):
    """
    通过API获取缺失基因的序列
    """
    print(f"开始通过API获取 {len(missing_genes)} 个缺失基因的序列...")

    found_sequences = {}
    failed_genes = []
    lncrna_genes = []
    non_lncrna_genes = []

    for i, gene_id in enumerate(missing_genes):
        print(f"处理进度: {i + 1}/{len(missing_genes)} - {gene_id}")

        # 获取基因信息
        gene_info = get_gene_info(gene_id)
        if gene_info:
            biotype = gene_info.get('biotype', 'unknown')
            print(f"  基因类型: {biotype}")

            # 记录基因类型
            if 'lnc' in biotype.lower() or 'linc' in biotype.lower():
                lncrna_genes.append((gene_id, biotype))
            else:
                non_lncrna_genes.append((gene_id, biotype))

        # 获取转录本列表
        transcripts = get_gene_transcripts(gene_id)

        if transcripts:
            print(f"  找到 {len(transcripts)} 个转录本")
            combined_sequence = ""

            # 获取所有转录本的序列
            for transcript_id in transcripts:
                sequence = get_transcript_sequence(transcript_id)
                if sequence:
                    combined_sequence += sequence
                    print(f"    {transcript_id}: {len(sequence)} bp")

                # 添加延时避免请求过快
                time.sleep(0.1)

            if combined_sequence:
                found_sequences[gene_id] = combined_sequence
                print(f"  总序列长度: {len(combined_sequence)} bp")
            else:
                failed_genes.append(gene_id)
                print(f"  未能获取到序列")
        else:
            failed_genes.append(gene_id)
            print(f"  未找到转录本")

        # 每10个基因添加一个较长的延时
        if (i + 1) % 10 == 0:
            print("  暂停1秒...")
            time.sleep(1)

    # 保存结果
    if found_sequences:
        with open(output_file, 'w', encoding='utf-8') as f:
            for gene_id in sorted(found_sequences.keys()):
                sequence = found_sequences[gene_id]
                f.write(f">{gene_id}\n")

                # 每行60字符
                for j in range(0, len(sequence), 60):
                    f.write(sequence[j:j + 60] + "\n")

        print(f"\n通过API获取的序列已保存到: {output_file}")

    # 生成统计报告
    print(f"\n=== API获取结果统计 ===")
    print(f"成功获取序列: {len(found_sequences)} 个")
    print(f"获取失败: {len(failed_genes)} 个")
    print(f"lncRNA相关基因: {len(lncrna_genes)} 个")
    print(f"非lncRNA基因: {len(non_lncrna_genes)} 个")

    if lncrna_genes:
        print(f"\nlncRNA相关基因:")
        for gene_id, biotype in lncrna_genes[:10]:  # 只显示前10个
            print(f"  {gene_id}: {biotype}")
        if len(lncrna_genes) > 10:
            print(f"  ... 还有 {len(lncrna_genes) - 10} 个")

    if failed_genes:
        print(f"\n获取失败的基因:")
        for gene_id in failed_genes[:10]:  # 只显示前10个
            print(f"  {gene_id}")
        if len(failed_genes) > 10:
            print(f"  ... 还有 {len(failed_genes) - 10} 个")

    return found_sequences, failed_genes


def merge_sequences_files(original_file: str, api_file: str, merged_file: str):
    """
    合并原始序列文件和API获取的序列文件
    """
    print(f"合并序列文件...")

    with open(merged_file, 'w', encoding='utf-8') as outf:
        # 先复制原始文件
        try:
            with open(original_file, 'r', encoding='utf-8') as inf:
                outf.write(inf.read())
            print(f"已复制原始文件: {original_file}")
        except FileNotFoundError:
            print(f"原始文件 {original_file} 未找到")

        # 再添加API获取的序列
        try:
            with open(api_file, 'r', encoding='utf-8') as inf:
                content = inf.read()
                if content.strip():  # 确保文件不为空
                    outf.write("\n" + content)
                    print(f"已添加API序列: {api_file}")
        except FileNotFoundError:
            print(f"API序列文件 {api_file} 未找到")

    print(f"合并完成，保存到: {merged_file}")


# 使用示例
if __name__ == "__main__":
    # 从你的运行结果中提取缺失的基因列表
    missing_genes = [
        "ENSG00000002079", "ENSG00000116883", "ENSG00000131484", "ENSG00000132832",
        "ENSG00000146521", "ENSG00000163009", "ENSG00000175911", "ENSG00000178863",
        "ENSG00000182912", "ENSG00000183653", "ENSG00000185162", "ENSG00000203594",
        # ... 这里可以添加更多缺失的基因ID
        # 或者从文件中读取完整列表
    ]


    # 你也可以从文件中读取缺失基因列表
    def load_missing_genes_from_report(report_file: str) -> List[str]:
        """从报告文件中提取缺失的基因ID"""
        missing_genes = []
        with open(report_file, 'r', encoding='utf-8') as f:
            in_missing_section = False
            for line in f:
                line = line.strip()
                if "缺失的基因ID:" in line:
                    in_missing_section = True
                    continue
                elif in_missing_section and line.startswith("ENSG"):
                    missing_genes.append(line)
                elif in_missing_section and not line:
                    break
        return missing_genes


    try:
        # 选择一种方式获取缺失基因列表

        # 方式1: 直接使用列表（测试少量基因）
        test_genes = missing_genes[:5]  # 先测试前5个

        # 方式2: 从报告文件读取（如果有报告文件的话）
        # missing_genes = load_missing_genes_from_report("extraction_report.txt")

        # 通过API获取序列
        found_sequences, failed_genes = fetch_missing_sequences_from_api(
            test_genes,
            "api_sequences.fa"
        )

        # 合并原始序列和API获取的序列
        merge_sequences_files(
            "extracted_sequences.fa",  # 原始提取的序列
            "api_sequences.fa",  # API获取的序列
            "complete_sequences.fa"  # 合并后的完整序列
        )

    except Exception as e:
        print(f"执行过程中出现错误: {e}")


# 批量处理所有缺失基因的函数
def process_all_missing_genes():
    """处理所有236个缺失的基因"""

    # 完整的缺失基因列表（从你的输出中复制）
    all_missing_genes = [
        "ENSG00000002079", "ENSG00000116883", "ENSG00000131484", "ENSG00000132832",
        "ENSG00000146521", "ENSG00000163009", "ENSG00000175911", "ENSG00000178863",
        "ENSG00000182912", "ENSG00000183653", "ENSG00000185162", "ENSG00000203594",
        "ENSG00000204282", "ENSG00000206044", "ENSG00000207846", "ENSG00000207857",
        # ... 这里需要添加完整的236个基因ID列表
    ]

    print(f"准备处理 {len(all_missing_genes)} 个缺失基因...")
    print("注意: 这将需要较长时间，建议分批处理")

    # 分批处理，每批20个
    batch_size = 20
    for i in range(0, len(all_missing_genes), batch_size):
        batch = all_missing_genes[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"\n=== 处理第 {batch_num} 批 ({len(batch)} 个基因) ===")

        found_sequences, failed_genes = fetch_missing_sequences_from_api(
            batch,
            f"api_sequences_batch_{batch_num}.fa"
        )

        print(f"第 {batch_num} 批完成，休息10秒...")
        time.sleep(10)