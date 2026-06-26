import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def generate_ecfp(smiles, radius=2, n_bits=512):
    """
    生成ECFP指纹

    参数:
        smiles: SMILES字符串
        radius: 指纹半径 (radius=2对应ECFP4, radius=3对应ECFP6)
        n_bits: 指纹维度

    返回:
        numpy数组 (n_bits维的二进制向量)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 生成Morgan指纹 (ECFP的RDKit实现)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits
        )
        # 转为numpy数组
        arr = np.zeros((n_bits,), dtype=np.int8)
        AllChem.DataStructs.ConvertToNumpyArray(ecfp, arr)
        return arr

    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return None


def process_drug_smiles(input_file, output_file, radius=2, n_bits=512):
    """
    批量处理SMILES文件生成ECFP指纹

    参数:
        input_file: 输入Excel文件路径 (ALLdrug-smiles.xlsx)
        output_file: 输出Excel文件路径
        radius: ECFP半径
        n_bits: 指纹维度
    """
    # 读取数据
    print(f"读取文件: {input_file}")
    df = pd.read_excel(input_file)

    # 检查必要的列
    if 'CID' not in df.columns or 'SMILES' not in df.columns:
        raise ValueError("文件必须包含 'CID' 和 'SMILES' 列")

    print(f"共有 {len(df)} 个药物需要处理")
    print(f"ECFP参数: radius={radius}, n_bits={n_bits}")

    # 生成指纹
    ecfp_list = []
    failed_cids = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成ECFP指纹"):
        cid = row['CID']
        smiles = row['SMILES']

        ecfp = generate_ecfp(smiles, radius=radius, n_bits=n_bits)

        if ecfp is not None:
            ecfp_list.append(ecfp)
        else:
            ecfp_list.append(np.zeros(n_bits, dtype=np.int8))  # 失败用零向量
            failed_cids.append(cid)

    # 构建结果DataFrame
    ecfp_array = np.array(ecfp_list)

    # 创建列名: ECFP_dim_0, ECFP_dim_1, ...
    column_names = [f'ECFP_dim_{i}' for i in range(n_bits)]

    result_df = pd.DataFrame(ecfp_array, columns=column_names)
    result_df.insert(0, 'CID', df['CID'].values)

    # 保存结果
    print(f"\n保存到: {output_file}")
    result_df.to_excel(output_file, index=False)

    # 统计信息
    print(f"\n处理完成!")
    print(f"成功: {len(df) - len(failed_cids)} / {len(df)}")
    if failed_cids:
        print(f"失败的CID ({len(failed_cids)}个): {failed_cids[:10]}...")  # 只显示前10个

    return result_df


# ====== 使用示例 ======
if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "ALLdrug-smiles.xlsx"
    OUTPUT_FILE = "Features_ECFP_Drug_128D.xlsx"

    # 可选参数
    RADIUS = 2  # ECFP4 (常用)
    N_BITS = 512  # 推荐先用128维,可改为256/512

    # 执行处理
    result = process_drug_smiles(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        radius=RADIUS,
        n_bits=N_BITS
    )

    # 查看结果
    print("\n结果预览:")
    print(result.head())
    print(f"\n输出维度: {result.shape}")