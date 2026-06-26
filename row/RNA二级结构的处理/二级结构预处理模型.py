import pandas as pd
import networkx as nx
import numpy as np
import pickle
from typing import List, Optional
import os
from tqdm import tqdm

# 安装依赖：pip install karateclub pandas networkx tqdm openpyxl

try:
    from karateclub import Graph2Vec
except ImportError:
    print("请先安装karateclub: pip install karateclub")
    exit(1)


class RNAGraph2VecProcessor:
    """
    RNA二级结构的Graph2Vec预处理器
    将RNA二级结构转换为图，然后用Graph2Vec生成固定维度的特征向量
    """

    def __init__(self, dimensions=128, epochs=10, workers=4, learning_rate=0.025):
        """
        初始化Graph2Vec模型

        Args:
            dimensions: embedding维度，默认128
            epochs: 训练轮数，默认10
            workers: 并行线程数，默认4
            learning_rate: 学习率，默认0.025
        """
        self.model = Graph2Vec(
            dimensions=dimensions,
            workers=workers,
            epochs=epochs,
            learning_rate=learning_rate,
            down_sampling=0.0001,
            min_count=1  # 设为1确保所有子图都被考虑
        )
        self.dimensions = dimensions
        self.fitted = False

    def structure_to_graph(self, structure: str) -> nx.Graph:
        """
        将RNA二级结构字符串转换为NetworkX图

        Args:
            structure: RNA二级结构，如 ".(((..)))."

        Returns:
            NetworkX图对象
        """
        G = nx.Graph()
        stack = []

        # 添加所有节点（每个位置一个节点）
        for i in range(len(structure)):
            G.add_node(i, position=i, structure_char=structure[i])

        # 添加主链连接（相邻碱基之间）
        for i in range(len(structure) - 1):
            G.add_edge(i, i + 1, edge_type='backbone')

        # 添加配对连接（根据括号匹配）
        for i, char in enumerate(structure):
            if char == '(':
                stack.append(i)
            elif char == ')' and stack:
                j = stack.pop()
                G.add_edge(i, j, edge_type='pairing')

        return G

    def load_rna_data(self, filepath: str) -> pd.DataFrame:
        """
        读取RNA数据文件

        Args:
            filepath: Excel文件路径

        Returns:
            pandas DataFrame
        """
        print(f"正在读取数据文件: {filepath}")

        # 根据文件扩展名选择读取方式
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("支持的文件格式：.xlsx, .xls, .csv")

        print(f"数据加载完成，共 {len(df)} 条记录")
        print(f"数据列: {list(df.columns)}")

        # 检查必要的列
        required_cols = ['structure']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")

        return df

    def validate_structures(self, structures: List[str]) -> List[str]:
        """
        验证和清理RNA结构数据

        Args:
            structures: RNA结构字符串列表

        Returns:
            清理后的结构列表
        """
        print("正在验证RNA结构数据...")

        valid_structures = []
        invalid_count = 0

        for i, struct in enumerate(structures):
            if pd.isna(struct) or not isinstance(struct, str):
                print(f"警告: 第{i + 1}行结构为空或非字符串，跳过")
                invalid_count += 1
                continue

            # 检查字符是否合法
            valid_chars = set('.()[]{}')
            if not all(c in valid_chars for c in struct):
                print(f"警告: 第{i + 1}行结构包含非法字符，跳过")
                invalid_count += 1
                continue

            # 检查括号是否匹配
            if not self._check_bracket_balance(struct):
                print(f"警告: 第{i + 1}行结构括号不匹配，跳过")
                invalid_count += 1
                continue

            valid_structures.append(struct)

        print(f"数据验证完成：有效 {len(valid_structures)} 条，无效 {invalid_count} 条")
        return valid_structures

    def _check_bracket_balance(self, structure: str) -> bool:
        """检查括号是否平衡"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}

        for char in structure:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                last = stack.pop()
                if pairs[last] != char:
                    return False

        return len(stack) == 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        训练Graph2Vec并转换数据

        Args:
            df: 包含RNA数据的DataFrame

        Returns:
            添加了Graph2Vec特征的DataFrame
        """
        # 验证结构数据
        structures = self.validate_structures(df['structure'].tolist())

        if len(structures) == 0:
            raise ValueError("没有有效的RNA结构数据")

        # 转换为图
        print("正在将RNA结构转换为图...")
        graphs = []
        valid_indices = []

        for i, structure in enumerate(tqdm(df['structure'])):
            if pd.isna(structure) or not isinstance(structure, str):
                continue
            if not self._check_bracket_balance(structure):
                continue

            try:
                graph = self.structure_to_graph(structure)
                if len(graph.nodes()) > 0:  # 确保图不为空
                    graphs.append(graph)
                    valid_indices.append(i)
            except Exception as e:
                print(f"警告: 转换第{i + 1}行结构时出错: {e}")
                continue

        print(f"成功转换 {len(graphs)} 个图结构")

        if len(graphs) == 0:
            raise ValueError("没有成功转换的图结构")

        # 训练Graph2Vec
        print("正在训练Graph2Vec模型...")
        self.model.fit(graphs)
        self.fitted = True

        # 获取embeddings
        print("正在获取图嵌入向量...")
        embeddings = self.model.get_embedding()

        # 创建特征DataFrame
        feature_columns = [f'g2v_dim_{i}' for i in range(self.dimensions)]
        embedding_df = pd.DataFrame(embeddings, columns=feature_columns)

        # 只保留有效的行
        valid_df = df.iloc[valid_indices].reset_index(drop=True)

        # 合并原始数据和Graph2Vec特征
        result_df = pd.concat([valid_df, embedding_df], axis=1)

        print(f"Graph2Vec特征生成完成！")
        print(f"原始特征数: {len(df.columns)}")
        print(f"添加Graph2Vec后特征数: {len(result_df.columns)}")
        print(f"有效样本数: {len(result_df)}")

        return result_df

    def save_model(self, filepath: str):
        """保存训练好的模型"""
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit_transform")

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载训练好的模型"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.fitted = True
        print(f"模型已从 {filepath} 加载")

    def get_feature_names(self) -> List[str]:
        """获取特征列名"""
        return [f'g2v_dim_{i}' for i in range(self.dimensions)]


def main():
    """主函数 - 使用示例"""

    # 配置参数
    input_file = r"D:\Desktop\CDLLM\ing\row\RNA二级结构的处理\lncRNA_structures_simple.xlsx"  # 你的输入文件名
    output_file = "rna_with_graph2vec_features.xlsx"
    model_file = "rna_graph2vec_model.pkl"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        print("请确保文件存在，或修改 input_file 变量为正确的路径")
        return

    try:
        # 初始化处理器
        processor = RNAGraph2VecProcessor(
            dimensions=128,  # embedding维度
            epochs=10,  # 训练轮数
            workers=4,  # 并行线程数
            learning_rate=0.025
        )

        # 加载数据
        df = processor.load_rna_data(input_file)

        # 显示原始数据信息
        print("\n=== 原始数据信息 ===")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print("\n前5行数据:")
        print(df.head())

        # 训练并转换
        print("\n=== 开始Graph2Vec预处理 ===")
        result_df = processor.fit_transform(df)

        # 保存结果
        result_df.to_excel(output_file, index=False)
        print(f"结果已保存到: {output_file}")

        # 保存模型
        processor.save_model(model_file)

        # 显示最终结果信息
        print("\n=== 预处理完成 ===")
        print(f"最终数据形状: {result_df.shape}")
        print(f"Graph2Vec特征列: {processor.get_feature_names()[:5]}... (共{processor.dimensions}个)")

        # 显示结果统计
        g2v_cols = [col for col in result_df.columns if col.startswith('g2v_')]
        print(f"\nGraph2Vec特征统计:")
        print(result_df[g2v_cols].describe())

        print("\n现在可以使用这些特征进行机器学习了！")
        print(f"特征文件: {output_file}")
        print(f"模型文件: {model_file}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()