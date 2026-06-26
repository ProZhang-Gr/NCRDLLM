
# PyTorch Geometric 使用示例
import torch
import numpy as np
from torch_geometric.data import Data

# 加载特征数据
node_features = np.load('node_features.npy')
edge_index = np.load('edge_index.npy')
edge_features = np.load('edge_features.npy')

# 转换为PyTorch张量
x = torch.tensor(node_features, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attr = torch.tensor(edge_features, dtype=torch.float)

# 创建PyG数据对象
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

print(f"节点特征: {data.x.shape}")  # torch.Size([21, 74])
print(f"边索引: {data.edge_index.shape}")  # torch.Size([2, 24])
print(f"边特征: {data.edge_attr.shape}")  # torch.Size([24, 12])

# 使用图神经网络进行处理
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class MolecularGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 图级别池化
        return self.classifier(x)

# 初始化模型
model = MolecularGNN(
    node_dim=74,
    edge_dim=12, 
    hidden_dim=64,
    output_dim=1  # 如用于回归任务
)
