import pandas as pd
import os
import subprocess
import tempfile
from tqdm import tqdm
import RNA  # ViennaRNA Python interface


def rnafold_predict(sequence, temperature=37.0):
    """
    使用RNAfold预测RNA二级结构

    Parameters:
    sequence (str): RNA序列
    temperature (float): 温度，默认37°C

    Returns:
    tuple: (structure, mfe) - 二级结构和最小自由能
    """
    try:
        # 设置温度
        RNA.cvar.temperature = temperature

        # 预测二级结构
        structure, mfe = RNA.fold(sequence)

        return structure, mfe
    except Exception as e:
        print(f"Error predicting structure for sequence: {e}")
        return None, None


def calculate_structure_features(sequence, structure):
    """
    计算RNA二级结构特征

    Parameters:
    sequence (str): RNA序列
    structure (str): 二级结构点括号表示

    Returns:
    dict: 结构特征字典
    """
    if not structure:
        return {}

    try:
        # 计算各种结构特征
        features = {}

        # 基本长度特征
        features['sequence_length'] = len(sequence)
        features['structure_length'] = len(structure)

        # 配对统计
        paired_positions = structure.count('(') + structure.count(')')
        unpaired_positions = structure.count('.')
        features['paired_positions'] = paired_positions
        features['unpaired_positions'] = unpaired_positions
        features['pairing_ratio'] = paired_positions / len(structure) if len(structure) > 0 else 0

        # 茎环结构统计
        features['num_stems'] = structure.count('(')  # 近似茎数量

        # 使用ViennaRNA计算更多特征
        features['centroid_distance'] = RNA.bp_distance(structure, RNA.centroid(sequence)[0])

        # 计算配对概率矩阵相关特征（可选）
        fc = RNA.fold_compound(sequence)
        fc.pf()  # 计算配分函数

        # 集合多样性
        features['ensemble_diversity'] = fc.mean_bp_distance()

        return features

    except Exception as e:
        print(f"Error calculating structure features: {e}")
        return {}


def process_excel_file_rnafold(input_file, output_file, id_column, seq_column, temperature=37.0):
    """处理单个Excel文件进行RNAfold预测"""
    print(f"\n处理文件: {input_file}")

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在!")
        return

    # 读取Excel文件
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取文件，共 {len(df)} 条记录")
    except Exception as e:
        print(f"读取文件错误: {e}")
        return

    # 检查必需的列是否存在
    if id_column not in df.columns or seq_column not in df.columns:
        print(f"错误: 文件中缺少必需的列 {id_column} 或 {seq_column}")
        print(f"实际列名: {list(df.columns)}")
        return

    # 提取ID和序列
    ids = df[id_column].tolist()
    sequences = df[seq_column].tolist()

    # 过滤掉空序列和无效序列
    valid_data = []
    for id_val, seq in zip(ids, sequences):
        if pd.notna(seq) and str(seq).strip() != '':
            # 将序列转换为大写并去除非AUGC字符（适应RNA序列）
            clean_seq = ''.join(c for c in str(seq).upper() if c in 'AUGC')
            if clean_seq:  # 确保清理后还有有效序列
                valid_data.append((id_val, clean_seq))

    if not valid_data:
        print("没有找到有效的序列数据")
        return

    valid_ids, valid_sequences = zip(*valid_data)
    print(f"有效序列数量: {len(valid_sequences)}")

    # 预测二级结构
    print("开始预测RNA二级结构...")
    results = []

    for i, (id_val, sequence) in enumerate(tqdm(zip(valid_ids, valid_sequences),
                                                total=len(valid_sequences),
                                                desc="Predicting structures")):
        # RNAfold预测
        structure, mfe = rnafold_predict(sequence, temperature)

        # 计算结构特征
        struct_features = calculate_structure_features(sequence, structure)

        # 组合结果
        result = {
            id_column: id_val,
            'sequence': sequence,
            'structure': structure if structure else '',
            'mfe': mfe if mfe is not None else 0.0,
        }

        # 添加结构特征
        result.update(struct_features)
        results.append(result)

    # 创建DataFrame
    result_df = pd.DataFrame(results)

    # 保存到Excel文件
    try:
        result_df.to_excel(output_file, index=False)
        print(f"结构预测结果已保存到: {output_file}")
        print(f"输出文件包含 {len(result_df)} 行, {len(result_df.columns)} 列")
        print(f"输出列包括: {list(result_df.columns)}")
    except Exception as e:
        print(f"保存文件错误: {e}")


def batch_process_rnafold():
    """批量处理三个RNA文件进行RNAfold预测"""

    # 文件配置
    files_config = [
        {
            'input_file': 'ALLcircRNA-seq.xlsx',
            'output_file': 'circRNA_structures.xlsx',
            'id_column': 'circBase_ID',
            'seq_column': 'seq'
        },
        {
            'input_file': 'ALLlncRNA-seq.xlsx',
            'output_file': 'lncRNA_structures.xlsx',
            'id_column': 'ENSEMBL_ID',
            'seq_column': 'seq'
        },
        {
            'input_file': 'ALLmiRNA-seq.xlsx',
            'output_file': 'miRNA_structures.xlsx',
            'id_column': 'miRBase_ID',
            'seq_column': 'seq'
        }
    ]

    print("开始批量处理RNA序列文件进行二级结构预测...")
    print("=" * 60)

    temperature = 37.0  # 预测温度
    print(f"使用温度: {temperature}°C")

    for config in files_config:
        process_excel_file_rnafold(
            config['input_file'],
            config['output_file'],
            config['id_column'],
            config['seq_column'],
            temperature
        )
        print("=" * 60)

    print("所有文件处理完成！")


def create_fasta_and_predict(input_file, id_column, seq_column, output_prefix):
    """
    可选方法：创建FASTA文件并使用命令行RNAfold
    这种方法适合处理大量序列
    """
    print(f"使用命令行RNAfold处理: {input_file}")

    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 创建临时FASTA文件
    fasta_file = f"{output_prefix}.fasta"
    with open(fasta_file, 'w') as f:
        for _, row in df.iterrows():
            if pd.notna(row[seq_column]):
                seq = ''.join(c for c in str(row[seq_column]).upper() if c in 'AUGC')
                if seq:
                    f.write(f">{row[id_column]}\n{seq}\n")

    # 运行RNAfold命令
    output_file = f"{output_prefix}_rnafold_output.txt"
    try:
        cmd = f"RNAfold -i {fasta_file} > {output_file}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"RNAfold输出保存到: {output_file}")

        # 清理临时文件
        os.remove(fasta_file)

    except subprocess.CalledProcessError as e:
        print(f"RNAfold命令执行失败: {e}")
    except FileNotFoundError:
        print("错误: 未找到RNAfold命令，请确保已安装ViennaRNA包")


if __name__ == "__main__":
    # 检查ViennaRNA是否已安装
    try:
        import RNA

        print("ViennaRNA Python接口已安装")

        # 使用Python接口进行批量处理
        batch_process_rnafold()

    except ImportError:
        print("未找到ViennaRNA Python接口")
        print("请安装ViennaRNA包:")
        print("conda install -c bioconda viennarna")
        print("或")
        print("pip install ViennaRNA")

        # 可以尝试使用命令行版本（如果可用）
        print("\n尝试使用命令行RNAfold...")
        files_config = [
            ('ALLcircRNA-seq.xlsx', 'circBase_ID', 'seq', 'circRNA'),
            ('ALLlncRNA-seq.xlsx', 'ENSEMBL_ID', 'seq', 'lncRNA'),
            ('ALLmiRNA-seq.xlsx', 'miRBase_ID', 'seq', 'miRNA')
        ]

        for input_file, id_col, seq_col, prefix in files_config:
            if os.path.exists(input_file):
                create_fasta_and_predict(input_file, id_col, seq_col, prefix)