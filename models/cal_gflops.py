import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from ptflops import get_model_complexity_info


class DynamicMaskDisentangle(nn.Module):
    """动态掩码特征解耦模块"""

    def __init__(self, embed_dim=256, hidden_dim=128):
        super().__init__()
        # 两层FC网络生成动态掩码
        self.mask_generator = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        """输入x: [seq_len, batch, embed_dim]"""
        # 生成动态掩码
        z = self.mask_generator(x)  # [seq_len, batch, embed_dim]
        confounder_mask = torch.sigmoid(z)
        adjustment_mask = torch.sigmoid(-z)  # 互补掩码

        # 特征解耦
        x_confounder = confounder_mask * x
        x_adjustment = adjustment_mask * x

        return x_adjustment, x_confounder


class MaskGCNDisentangle(nn.Module):
    """基于掩码的GCN解耦模块"""

    def __init__(self, embed_dim=256):
        super().__init__()
        # 动态掩码解耦
        self.dynamic_mask = DynamicMaskDisentangle(embed_dim)

        # 分离的GCN聚合器
        self.gcn_adjustment = GCNConv(embed_dim, embed_dim)
        self.gcn_confounder = GCNConv(embed_dim, embed_dim)

    def forward(self, x, edge_index):
        """输入x: [seq_len, batch, embed_dim], edge_index: [2, num_edges]"""
        seq_len, batch_size, embed_dim = x.size()

        # 1. 动态特征解耦
        x_adj, x_conf = self.dynamic_mask(x)  # 各[seq_len, batch, embed_dim]

        # 2. 对batch中的每个样本独立处理图卷积
        adj_feats, conf_feats = [], []
        for b in range(batch_size):
            # 获取当前batch的节点特征 [seq_len, embed_dim]
            nodes_adj = x_adj[:, b, :]
            nodes_conf = x_conf[:, b, :]

            # 分别聚合调整因子和混淆因子
            adj_feat = self.gcn_adjustment(nodes_adj, edge_index)
            conf_feat = self.gcn_confounder(nodes_conf, edge_index)

            adj_feats.append(adj_feat)
            conf_feats.append(conf_feat)

        # [seq_len, batch, embed_dim]
        adj_feats = torch.stack(adj_feats, dim=1)
        conf_feats = torch.stack(conf_feats, dim=1)

        return adj_feats, conf_feats


class CausalMultiHypothesisGraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, ff_dim=512, num_hypotheses=3, num_gat_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hypotheses = num_hypotheses
        self.num_gat_layers = num_gat_layers

        self.feature_disentangle = MaskGCNDisentangle(embed_dim)

        self.causal_intervention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            ) for _ in range(num_hypotheses)
        ])

        # 用多层次 GATConv 生成不同粒度的假设
        self.hypothesis_generators = nn.ModuleList([
            nn.ModuleList([
                GATv2Conv(embed_dim, embed_dim, heads=num_heads, concat=False, dropout=dropout)
                for _ in range(num_gat_layers)  # Each layer has own GAT
            ])
            for _ in range(num_hypotheses)
        ])

        self.node_projection = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def create_graph_edges(self, H, W, device):
        # 4邻域格点图
        idx = torch.arange(H * W, device=device).reshape(H, W)
        edges = []
        for i in range(H):
            for j in range(W):
                src = idx[i, j]
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        tgt = idx[ni, nj]
                        edges.append([src, tgt])
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

    def forward(self, x, edge_index=None):
        """
        x: [batch, C, H, W]
        edge_index: [2, num_edges] or None
        """
        batch, C, H, W = x.size()
        seq_len = H * W
        device = x.device

        x_seq = x.view(batch, C, seq_len).permute(2, 0, 1)  # [seq_len, batch, C]

        if edge_index is None:
            edge_index = self.create_graph_edges(H, W, device)

        node_features = self.node_projection(x_seq)  # [seq_len, batch, embed_dim]

        # 解耦
        adj_feat, conf_feat = self.feature_disentangle(node_features, edge_index)

        # LayerNorm
        adj_feat = self.norms[0](adj_feat)
        conf_feat = self.norms[1](conf_feat)

        hypotheses = []
        for i in range(self.num_hypotheses):
            # 干预混淆因子
            intervention = self.causal_intervention[i](conf_feat)
            intervened_feat = adj_feat + (conf_feat + intervention)

            # 对每个batch分别做GAT，并使用不同层次的GAT
            h_batch = []
            for b in range(batch):
                nodes = intervened_feat[:, b, :]  # [seq_len, embed_dim]

                # 选择不同的graph attention layer
                gat_layer_index = i % self.num_gat_layers  # 保证每个假设使用不同的层次
                h_nodes = self.hypothesis_generators[i][gat_layer_index](nodes, edge_index)  # [seq_len, embed_dim]

                h_batch.append(h_nodes)
            h = torch.stack(h_batch, dim=1)  # [seq_len, batch, embed_dim]
            hypotheses.append(h)

        # 原始特征（调整因子和混淆因子的组合）
        original_feat = adj_feat + conf_feat

        # FFN
        original_feat = self.norms[2](original_feat + self.dropout(self.ffn(original_feat)))

        # 转换回原始形状 [batch, C, H, W]
        hypotheses_out = [h.permute(1, 2, 0).contiguous().view(batch, C, H, W)
                          for h in hypotheses]
        original_feat_out = original_feat.permute(1, 2, 0).contiguous().view(batch, C, H, W)

        return (*hypotheses_out, original_feat_out)


class CausalMultiHypothesisGraphTransformer(nn.Module):
    """顶层模型"""

    def __init__(self, num_layers=2, num_gat_layers=3, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalMultiHypothesisGraphTransformerLayer(num_gat_layers=num_gat_layers, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x, edge_index=None):
        for layer in self.layers:
            *h, original_feat = layer(x, edge_index)
            x = original_feat  # 将原始特征传入下一层
        return (*h, original_feat)


# 计算GFLOPs的函数
def calculate_gflops(model, input_shape=(2, 256, 8, 8)):
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    gflops = macs / 1e9  # 转换为GFLOPs
    return gflops, params


# 使用示例
if __name__ == "__main__":
    num_gat_layers = 3
    model = CausalMultiHypothesisGraphTransformer(
        num_layers=3,
        embed_dim=256,
        num_heads=8,
        dropout=0.1,
        ff_dim=512,
        num_hypotheses=3,
        num_gat_layers=num_gat_layers
    )

    input_shape = (2, 256, 8, 8)  # (batch, channels, height, width)
    gflops, params = calculate_gflops(model, input_shape)

    print(f"Model GFLOPs: {gflops:.2f} G")
    print(f"Number of parameters: {params / 1e6:.2f} M")

    x = torch.randn(input_shape)
    h1, h2, h3, original = model(x)
    print(f"Output shapes - h1: {h1.shape}, h2: {h2.shape}, h3: {h3.shape}, original: {original.shape}")