
import torch
import torch.nn as nn
import torch.nn.functional as F


# 两个模块类和一个混合层类：

class FCBlock(nn.Module):
    # 这个类定义了一个全连接块，其中包含一个线性层、LayerNorm 和 ReLU 激活函数。
    def __init__(self, dim, out_dim):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ff(x)


class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)


class MixerLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 hidden_inter_dim,
                 token_dim,
                 token_inter_dim,
                 dropout_ratio):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Points_MLP(nn.Module):
    def __init__(self, input_dim=3 * 17 * 2, output_dim=1 * 17 * 2, hidden_dims=[102, 34]):
        super(Points_MLP, self).__init__()

        layers = []
        prev_dim = input_dim
        # 创建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        # 如果需要，可以添加激活函数，例如 ReLU 或 Sigmoid
        # layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x 的形状假设为 (batch_size, 3, 17, 2)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平成 (batch_size, 102)
        x = self.network(x)  # 通过 MLP 得到 (batch_size,34)
        x = x.view(batch_size, 1, 17, 2)  # 重新调整为 (batch_size, 1, 17, 2)
        return x


class GraphAttentionLayer(nn.Module):
    """图注意力层 (基于GAT改进，支持多关系交互)"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性变换矩阵
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.W_key = nn.Linear(embed_dim, embed_dim)
        self.W_value = nn.Linear(embed_dim, embed_dim)

        # 注意力偏置（可学习的关系权重）
        self.relation_bias = nn.Parameter(torch.randn(num_heads))

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, nodes, edges=None):
        """
        nodes: [num_nodes, batch, embed_dim]
        edges: [num_nodes, num_nodes] 可选的关系矩阵
        """
        num_nodes, batch_size, _ = nodes.shape

        # 线性变换
        q = self.W_query(nodes).view(num_nodes, batch_size, self.num_heads, self.head_dim)
        k = self.W_key(nodes).view(num_nodes, batch_size, self.num_heads, self.head_dim)
        v = self.W_value(nodes).view(num_nodes, batch_size, self.num_heads, self.head_dim)

        # 计算注意力分数 [num_nodes, num_nodes, num_heads, batch]
        attn_scores = torch.einsum('ibhd,jbhd->hbij', q, k) / (self.head_dim ** 0.5)

        # 添加关系偏置
        if edges is not None:
            attn_scores += edges.unsqueeze(0).unsqueeze(-1) * self.relation_bias.view(1, 1, -1, 1)

        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)

        # 聚合特征 [num_nodes, batch, num_heads, head_dim]
        out = torch.einsum('hbij,jbhd->ibhd', attn_weights, v)
        out = out.contiguous().view(num_nodes, batch_size, self.embed_dim)

        return self.out_proj(out)


class HierarchicalGraphAttention(nn.Module):
    """层次化图注意力模块"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # 全局图注意力（粗粒度）
        self.global_gat = GraphAttentionLayer(embed_dim, num_heads, dropout)

        # 局部图注意力（细粒度）
        self.local_gat = GraphAttentionLayer(embed_dim, num_heads, dropout)

        # 跨层次注意力
        self.cross_level_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # 归一化层
        self.norm_global = nn.LayerNorm(embed_dim)
        self.norm_local = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def build_hierarchical_graph(self, x, patch_size=4):
        """
        构建层次化图结构：
        1. 全局图：将特征图划分为粗粒度的超级节点
        2. 局部图：在每个超级节点内部构建细粒度连接
        """
        batch, C, H, W = x.shape
        device = x.device

        # --- 全局图构建 ---
        # 将特征图划分为 PxP 的超级节点
        P = H // patch_size
        global_nodes = F.avg_pool2d(x, kernel_size=patch_size)  # [batch, C, P, P]
        global_nodes = global_nodes.view(batch, C, -1).permute(2, 0, 1)  # [P*P, batch, C]

        # 全连接的全局边（可替换为稀疏连接）
        global_edges = torch.ones(P * P, P * P, device=device)

        # --- 局部图构建 ---
        # 每个patch内部的局部连接
        local_nodes = x.view(batch, C, H * W).permute(2, 0, 1)  # [H*W, batch, C]

        # 构建局部邻接矩阵（4连通）
        local_edges = torch.zeros(H * W, H * W, device=device)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                # 连接上下左右邻居
                if i > 0: local_edges[idx, idx - W] = 1
                if i < H - 1: local_edges[idx, idx + W] = 1
                if j > 0: local_edges[idx, idx - 1] = 1
                if j < W - 1: local_edges[idx, idx + 1] = 1

        return global_nodes, global_edges, local_nodes, local_edges

    def forward(self, x):
        # 输入x: [batch, C, H, W]
        batch, C, H, W = x.shape

        # 构建层次化图
        global_nodes, global_edges, local_nodes, local_edges = self.build_hierarchical_graph(x)

        # --- 全局图注意力 ---
        global_out = self.global_gat(global_nodes, global_edges)
        global_out = self.norm_global(global_out + global_nodes)

        # --- 局部图注意力 ---
        local_out = self.local_gat(local_nodes, local_edges)
        local_out = self.norm_local(local_out + local_nodes)

        # --- 跨层次交互 ---
        # 上采样全局特征到局部分辨率
        global_up = F.interpolate(
            global_out.permute(1, 2, 0).view(batch, C, H // 4, W // 4),
            size=(H, W), mode='nearest'
        ).view(batch, C, H * W).permute(2, 0, 1)

        cross_out, _ = self.cross_level_attn(
            query=local_out,
            key=global_up,
            value=global_up
        )
        cross_out = self.norm_cross(cross_out + local_out)

        # 恢复原始形状
        output = cross_out.permute(1, 2, 0).view(batch, C, H, W)
        return output


class MultiHypothesisGraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, ff_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # 假设独立的层次化图注意力
        self.hga_x1 = HierarchicalGraphAttention(embed_dim, num_heads, dropout)
        self.hga_x2 = HierarchicalGraphAttention(embed_dim, num_heads, dropout)
        self.hga_x3 = HierarchicalGraphAttention(embed_dim, num_heads, dropout)

        # 假设间图注意力交互
        self.inter_hypothesis_gat = GraphAttentionLayer(embed_dim, num_heads, dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # 归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, x3):
        batch, C, H, W = x1.shape

        # --- 假设内层次化图注意力 ---
        # 转换为序列形式 [H*W, batch, C]
        x1_seq = x1.view(batch, C, H * W).permute(2, 0, 1)
        x2_seq = x2.view(batch, C, H * W).permute(2, 0, 1)
        x3_seq = x3.view(batch, C, H * W).permute(2, 0, 1)

        # 层次化图处理
        hga_x1 = self.hga_x1(x1)
        hga_x2 = self.hga_x2(x2)
        hga_x3 = self.hga_x3(x3)

        # 残差连接
        x1 = x1 + self.dropout(hga_x1)
        x2 = x2 + self.dropout(hga_x2)
        x3 = x3 + self.dropout(hga_x3)
        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        x3 = self.norm1(x3)

        # --- 假设间图注意力 ---
        # 合并所有假设节点 [3*H*W, batch, C]
        all_nodes = torch.cat([
            x1.view(batch, C, H * W).permute(2, 0, 1),
            x2.view(batch, C, H * W).permute(2, 0, 1),
            x3.view(batch, C, H * W).permute(2, 0, 1)
        ], dim=0)

        # 构建假设间全连接边
        inter_edges = torch.ones(3 * H * W, 3 * H * W, device=x1.device)

        # 图注意力交互
        inter_out = self.inter_hypothesis_gat(all_nodes, inter_edges)

        # 分割回各假设
        inter_out = inter_out.split(H * W, dim=0)
        x1 = x1 + self.dropout(inter_out[0].permute(1, 2, 0).view(batch, C, H, W))
        x2 = x2 + self.dropout(inter_out[1].permute(1, 2, 0).view(batch, C, H, W))
        x3 = x3 + self.dropout(inter_out[2].permute(1, 2, 0).view(batch, C, H, W))
        x1 = self.norm2(x1)
        x2 = self.norm2(x2)
        x3 = self.norm2(x3)

        # --- 前馈网络 ---
        x1 = x1 + self.dropout(self.ffn(x1.view(batch, C, H * W).permute(2, 0, 1))
                               .permute(1, 2, 0).view(batch, C, H, W))
        x2 = x2 + self.dropout(self.ffn(x2.view(batch, C, H * W).permute(2, 0, 1))
                               .permute(1, 2, 0).view(batch, C, H, W))
        x3 = x3 + self.dropout(self.ffn(x3.view(batch, C, H * W).permute(2, 0, 1))
                               .permute(1, 2, 0).view(batch, C, H, W))
        x1 = self.norm3(x1)
        x2 = self.norm3(x2)
        x3 = self.norm3(x3)

        return x1, x2, x3


class MultiHypothesisGraphTransformer(nn.Module):
    def __init__(self, num_layers=2, embed_dim=256, num_heads=8, dropout=0.1, ff_dim=512):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHypothesisGraphTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_dim=ff_dim
            ) for _ in range(num_layers)
        ])

    def forward(self, x1, x2, x3):
        for layer in self.layers:
            x1, x2, x3 = layer(x1, x2, x3)
        return x1, x2, x3


