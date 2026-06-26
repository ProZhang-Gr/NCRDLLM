import pandas as pd
import torch
import numpy as np
import os
from multimolecule import RnaTokenizer, RnaFmModel
from tqdm import tqdm


def extract_bert_features(sequences, batch_size=16):
    """使用 RNA-FM 提取序列特征，支持批量处理"""
    print("Loading RNA-FM model...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    all_features = []

    # 批量处理序列
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing sequences"):
        batch_sequences = sequences[i:i + batch_size]
        batch_features = []

        for sequence in batch_sequences:
            # 处理每个序列
            max_length = min(len(sequence), tokenizer.model_max_length)
            inputs = tokenizer(
                sequence,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # 提取CLS token的特征（第一个token）
                cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                batch_features.append(cls_features[0])

        all_features.extend(batch_features)

    return all_features


def process_excel_file(input_file, output_file, id_column, seq_column):
    """处理单个Excel文件"""
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

    # 过滤掉空序列
    valid_data = [(id_val, seq) for id_val, seq in zip(ids, sequences)
                  if pd.notna(seq) and str(seq).strip() != '']

    if not valid_data:
        print("没有找到有效的序列数据")
        return

    valid_ids, valid_sequences = zip(*valid_data)
    print(f"有效序列数量: {len(valid_sequences)}")

    # 提取特征
    print("开始提取RNA-FM特征...")
    features = extract_bert_features(list(valid_sequences))

    # 创建特征DataFrame
    feature_df = pd.DataFrame(features)

    # 为特征列添加列名
    feature_columns = [f'feature{i}' for i in range(feature_df.shape[1])]
    feature_df.columns = feature_columns

    # 插入ID列
    feature_df.insert(0, id_column, valid_ids)

    # 保存到Excel文件
    try:
        feature_df.to_excel(output_file, index=False)
        print(f"特征矩阵已保存到: {output_file}")
        print(f"输出文件包含 {len(feature_df)} 行, {len(feature_df.columns)} 列")
    except Exception as e:
        print(f"保存文件错误: {e}")


def batch_process_rna_files():
    """批量处理三个RNA文件"""

    # 文件配置
    files_config = [
        {
            'input_file': 'ALLcircRNA-seq.xlsx',
            'output_file': 'circRNA_features.xlsx',
            'id_column': 'circBase_ID',
            'seq_column': 'seq'
        },
        {
            'input_file': 'ALLlncRNA-seq.xlsx',
            'output_file': 'lncRNA_features.xlsx',
            'id_column': 'ENSEMBL_ID',
            'seq_column': 'seq'
        },
        {
            'input_file': 'ALLmiRNA-seq.xlsx',
            'output_file': 'miRNA_features.xlsx',
            'id_column': 'miRBase_ID',
            'seq_column': 'seq'
        }
    ]

    print("开始批量处理RNA序列文件...")
    print("=" * 50)

    for config in files_config:
        process_excel_file(
            config['input_file'],
            config['output_file'],
            config['id_column'],
            config['seq_column']
        )
        print("=" * 50)

    print("所有文件处理完成！")


if __name__ == "__main__":

    os.chdir(r"D:\Desktop\CDLLM\ing\row\RNAFM处理出序列预处理特征\row")

    batch_process_rna_files()