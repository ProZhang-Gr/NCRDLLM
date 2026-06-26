import os
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np


def get_atom_features(atom):
    """
    提取原子特征，基于DeepChem特征表
    """
    # 原子类型 one-hot编码 (43维)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                  'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                  'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
                  'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']

    atom_symbol = atom.GetSymbol()
    atom_type_feature = [1 if atom_symbol == t else 0 for t in atom_types]

    # 原子度数 one-hot编码 (0-10)
    degree = atom.GetDegree()
    degree_feature = [1 if degree == i else 0 for i in range(11)]

    # 隐式化合价 one-hot编码 (0-6)
    implicit_valence = atom.GetImplicitValence()
    valence_feature = [1 if implicit_valence == i else 0 for i in range(7)]

    # 原子形式电荷
    formal_charge = atom.GetFormalCharge()

    # 自由基电子数
    radical_electrons = atom.GetNumRadicalElectrons()

    # 是否芳香性
    is_aromatic = 1 if atom.GetIsAromatic() else 0

    # 原子杂化类型 one-hot编码
    hybridization_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                           Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                           Chem.HybridizationType.SP3D2]
    hybridization = atom.GetHybridization()
    hybridization_feature = [1 if hybridization == h else 0 for h in hybridization_types]

    # 连接的氢原子数 one-hot编码 (0-4)
    total_hydrogens = atom.GetTotalNumHs()
    hydrogen_feature = [1 if total_hydrogens == i else 0 for i in range(5)]

    # 合并所有特征
    features = (atom_type_feature + degree_feature + valence_feature +
                [formal_charge, radical_electrons, is_aromatic] +
                hybridization_feature + hydrogen_feature)

    return features


def get_bond_features(bond):
    """
    提取化学键特征
    """
    # 化学键类型 one-hot编码
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                  Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    bond_type = bond.GetBondType()
    bond_type_feature = [1 if bond_type == bt else 0 for bt in bond_types]

    # 是否共轭
    is_conjugated = 1 if bond.GetIsConjugated() else 0

    # 是否在环中
    is_in_ring = 1 if bond.IsInRing() else 0

    # 立体构型 one-hot编码
    stereo_types = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY,
                    Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
                    Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]
    stereo = bond.GetStereo()
    stereo_feature = [1 if stereo == st else 0 for st in stereo_types]

    # 合并所有特征
    features = bond_type_feature + [is_conjugated, is_in_ring] + stereo_feature

    return features


def get_bond_type_name(bond):
    """
    获取化学键类型名称
    """
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return "SINGLE"
    elif bond_type == Chem.BondType.DOUBLE:
        return "DOUBLE"
    elif bond_type == Chem.BondType.TRIPLE:
        return "TRIPLE"
    elif bond_type == Chem.BondType.AROMATIC:
        return "AROMATIC"
    else:
        return "OTHER"


def sdf_to_graph_json(sdf_path, cid):
    """
    将SDF文件转换为图结构JSON格式

    Args:
        sdf_path: SDF文件路径
        cid: 化合物CID

    Returns:
        dict: 图结构数据
    """
    try:
        # 读取SDF文件
        mol = Chem.MolFromMolFile(sdf_path)
        if mol is None:
            print(f"  ✗ 无法读取SDF文件: {sdf_path}")
            return None

        # 添加氢原子以确保完整性
        mol = Chem.AddHs(mol)

        # 提取节点（原子）信息
        nodes = []
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features = get_atom_features(atom)
            node = {
                "id": i,
                "element": atom.GetSymbol(),
                "features": atom_features
            }
            nodes.append(node)

        # 提取边（化学键）信息
        edges = []
        for bond in mol.GetBonds():
            bond_features = get_bond_features(bond)
            bond_type_name = get_bond_type_name(bond)

            edge = {
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx(),
                "bond_type": bond_type_name,
                "features": bond_features
            }
            edges.append(edge)

        # 构建图结构数据
        graph_data = {
            "molecule_id": str(cid),
            "nodes": nodes,
            "edges": edges
        }

        return graph_data

    except Exception as e:
        print(f"  ✗ 处理SDF文件时出错: {e}")
        return None


def batch_process_sdf_files(sdf_dir, output_dir="druggraphs"):
    """
    批量处理SDF文件并生成JSON格式的图结构数据

    Args:
        sdf_dir: SDF文件目录
        output_dir: 输出JSON文件的目录

    Returns:
        dict: 处理结果统计
    """
    print("开始批量处理SDF文件...")

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)

    # 获取所有SDF文件
    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]

    if not sdf_files:
        print(f"在目录 {sdf_dir} 中未找到SDF文件")
        return None

    print(f"找到 {len(sdf_files)} 个SDF文件")

    # 统计变量
    success_count = 0
    failed_files = []

    # 逐个处理SDF文件
    for idx, sdf_file in enumerate(sdf_files):
        # 从文件名提取CID
        cid = sdf_file.replace('.sdf', '')

        print(f"\n处理 {idx + 1}/{len(sdf_files)}: {sdf_file} (CID: {cid})")

        # SDF文件路径
        sdf_path = os.path.join(sdf_dir, sdf_file)

        # 转换为图结构数据
        graph_data = sdf_to_graph_json(sdf_path, cid)

        if graph_data is not None:
            # 保存为JSON文件
            json_path = os.path.join(output_dir, f"{cid}.json")

            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False)

                success_count += 1
                print(f"  ✓ 成功生成: {json_path}")
                print(f"    节点数: {len(graph_data['nodes'])}")
                print(f"    边数: {len(graph_data['edges'])}")

            except Exception as e:
                print(f"  ✗ 保存JSON文件失败: {e}")
                failed_files.append(sdf_file)
        else:
            failed_files.append(sdf_file)

    # 统计结果
    result = {
        'total_files': len(sdf_files),
        'success_count': success_count,
        'failed_count': len(failed_files),
        'success_rate': success_count / len(sdf_files) * 100,
        'failed_files': failed_files
    }

    return result


def print_processing_summary(result):
    """
    打印处理结果摘要
    """
    if result is None:
        return

    print("\n" + "=" * 60)
    print("SDF文件处理结果摘要")
    print("=" * 60)

    print(f"总SDF文件数: {result['total_files']}")
    print(f"成功转换: {result['success_count']}")
    print(f"转换失败: {result['failed_count']}")
    print(f"成功率: {result['success_rate']:.1f}%")

    if result['failed_files']:
        print(f"\n转换失败的文件:")
        for filename in result['failed_files']:
            print(f"  - {filename}")
    else:
        print(f"\n🎉 所有SDF文件都成功转换为图结构JSON!")


def verify_json_files(output_dir="druggraphs", sample_size=3):
    """
    验证生成的JSON文件
    """
    print(f"\n验证生成的JSON文件...")

    if not os.path.exists(output_dir):
        print(f"输出目录不存在: {output_dir}")
        return

    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个JSON文件")

    # 检查前几个文件的内容
    for i, json_file in enumerate(json_files[:sample_size]):
        json_path = os.path.join(output_dir, json_file)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"\n文件: {json_file}")
            print(f"  分子ID: {data['molecule_id']}")
            print(f"  节点数: {len(data['nodes'])}")
            print(f"  边数: {len(data['edges'])}")

            # 检查特征维度
            if data['nodes']:
                node_feature_dim = len(data['nodes'][0]['features'])
                print(f"  节点特征维度: {node_feature_dim}")

            if data['edges']:
                edge_feature_dim = len(data['edges'][0]['features'])
                print(f"  边特征维度: {edge_feature_dim}")

        except Exception as e:
            print(f"  ✗ 读取JSON文件失败: {e}")


# 使用示例
if __name__ == "__main__":
    # SDF文件目录
    sdf_dir = r"D:\Desktop\CDLLM\ing\row\SDF\sdf_files"

    # 输出目录
    output_dir = "druggraphs"

    print("开始SDF文件到图结构JSON的转换...")

    # 批量处理SDF文件
    result = batch_process_sdf_files(sdf_dir, output_dir)

    if result:
        # 打印处理摘要
        print_processing_summary(result)

        # 验证生成的JSON文件
        verify_json_files(output_dir, sample_size=5)

        print(f"\n处理完成！")
        print(f"JSON文件已保存到: {output_dir}")
        print(f"可以开始进行图神经网络训练了!")