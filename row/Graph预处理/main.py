import json
import torch
import dgl
from dgllife.model import AttentiveFPPredictor
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class DrugGraphFeatureExtractor:
    """使用AttentiveFP提取药物分子图特征并输出为XLSX"""

    def __init__(self, graph_feat_size=200, model_path=None, device='cpu'):
        """
        初始化特征提取器

        Args:
            graph_feat_size: 输出特征维度（默认200，可自定义）
            model_path: 预训练模型路径（可选）
            device: 'cpu' 或 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.graph_feat_size = graph_feat_size

        # 初始化AttentiveFP模型
        self.model = AttentiveFPPredictor(
            node_feat_size=74,
            edge_feat_size=12,
            num_layers=2,
            num_timesteps=2,
            graph_feat_size=graph_feat_size,
            n_tasks=1,
            dropout=0.2
        ).to(self.device)

        # 如果提供了预训练模型，加载权重
        if model_path:
            self.load_pretrained_model(model_path)

        self.model.eval()
        print(f"模型初始化完成，输出特征维度: {graph_feat_size}")

    def load_pretrained_model(self, model_path):
        """加载预训练模型权重"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"✓ 成功加载预训练模型: {model_path}")
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")

    def json_to_dgl_graph(self, drug_json):
        """
        将JSON格式的药物图转换为DGL图

        Args:
            drug_json: 字典或JSON文件路径

        Returns:
            DGL图对象
        """
        # 如果是文件路径，先加载
        if isinstance(drug_json, (str, Path)):
            with open(drug_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = drug_json

        num_nodes = len(data['nodes'])

        # 提取边信息
        src_nodes = [edge['source'] for edge in data['edges']]
        dst_nodes = [edge['target'] for edge in data['edges']]

        # 创建DGL图（无向图，添加双向边）
        g = dgl.graph((src_nodes + dst_nodes, dst_nodes + src_nodes),
                      num_nodes=num_nodes)

        # 添加节点特征
        node_features = torch.tensor(
            [node['features'] for node in data['nodes']],
            dtype=torch.float32
        )
        g.ndata['h'] = node_features

        # 添加边特征（双向复制）
        edge_features = [edge['features'] for edge in data['edges']]
        edge_features = edge_features + edge_features
        g.edata['e'] = torch.tensor(edge_features, dtype=torch.float32)

        return g, data['molecule_id']

    def extract_features(self, drug_json):
        """
        提取单个药物的特征

        Args:
            drug_json: 药物图的JSON数据或文件路径

        Returns:
            (molecule_id, features): 分子ID和特征向量
        """
        # 转换为DGL图
        g, molecule_id = self.json_to_dgl_graph(drug_json)
        g = g.to(self.device)

        # 提取特征
        with torch.no_grad():
            node_feats = g.ndata['h']
            edge_feats = g.edata['e']

            # 使用AttentiveFP的gnn部分提取图级特征
            graph_feats = self.model.gnn(g, node_feats, edge_feats)

        return molecule_id, graph_feats.cpu().numpy().flatten()

    def process_directory(self, input_dir, output_xlsx):
        """
        批量处理目录下所有JSON文件并输出XLSX

        Args:
            input_dir: 输入目录路径（包含JSON文件）
            output_xlsx: 输出XLSX文件路径
        """
        input_path = Path(input_dir)
        json_files = list(input_path.glob('*.json'))

        if not json_files:
            print(f"✗ 在 {input_dir} 中未找到JSON文件！")
            return

        print(f"找到 {len(json_files)} 个JSON文件")
        print(f"开始提取特征...")

        results = []
        failed_files = []

        # 使用tqdm显示进度条
        for json_file in tqdm(json_files, desc="处理进度"):
            try:
                molecule_id, features = self.extract_features(json_file)

                # 构建一行数据
                row = {'molecule_id': molecule_id}
                for i, feat_val in enumerate(features):
                    row[f'ATFP_dim_{i}'] = feat_val

                results.append(row)

            except Exception as e:
                failed_files.append((json_file.name, str(e)))
                print(f"\n✗ 处理失败: {json_file.name} - {e}")

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 确保列顺序：molecule_id在第一列
        cols = ['molecule_id'] + [f'ATFP_dim_{i}' for i in range(self.graph_feat_size)]
        df = df[cols]

        # 保存为XLSX
        df.to_excel(output_xlsx, index=False, engine='openpyxl')

        # 输出统计信息
        print(f"\n{'=' * 60}")
        print(f"✓ 处理完成！")
        print(f"  - 成功处理: {len(results)} 个药物")
        print(f"  - 失败数量: {len(failed_files)} 个")
        print(f"  - 特征维度: {self.graph_feat_size}")
        print(f"  - 输出文件: {output_xlsx}")
        print(f"{'=' * 60}")

        if failed_files:
            print("\n失败文件列表:")
            for fname, error in failed_files:
                print(f"  - {fname}: {error}")

        return df


# ==================== 主程序 ====================

def main():
    # 配置参数
    INPUT_DIR = r"D:\Desktop\CDLLM\ing\official\druggraphs"
    OUTPUT_XLSX = r"D:\Desktop\CDLLM\ing\official\drug_features_attentivefp.xlsx"
    FEATURE_DIM = 200  # 可自定义输出维度

    # 初始化特征提取器
    extractor = DrugGraphFeatureExtractor(
        graph_feat_size=FEATURE_DIM,
        model_path=None,  # 如果有预训练模型，填入路径
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 批量处理
    df = extractor.process_directory(INPUT_DIR, OUTPUT_XLSX)

    # 显示前几行预览
    if df is not None and len(df) > 0:
        print("\n数据预览（前5行）:")
        print(df.head())
        print(f"\nDataFrame形状: {df.shape}")


if __name__ == "__main__":
    main()