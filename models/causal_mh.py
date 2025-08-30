import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHypothesisTransformerLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, ff_dim=512, num_hypotheses=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hypotheses = num_hypotheses

        # 因果干预模块
        self.causal_intervention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            ) for _ in range(num_hypotheses)
        ])

        # 假设生成器（使用共享权重）
        self.hypothesis_generators = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_hypotheses)
        ])

        # 反事实注意力机制
        self.counterfactual_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=dropout)
            for _ in range(num_hypotheses)
        ])

        # 因果特征解耦模块
        self.feature_disentanglers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            ) for _ in range(2)  # 因果特征和非因果特征
        ])

        # 共享的前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # 规范化层
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

        # 假设融合权重预测
        self.fusion_weight_predictor = nn.Sequential(
            nn.Linear(embed_dim * num_hypotheses, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_hypotheses),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        参数:
            x: Tensor, 形状为 [batch, C, H, W]
        返回:
            Tuple[Tensor]: 多个假设特征和融合结果
        """
        batch, C, H, W = x.size()
        seq_len = H * W

        # 转换为序列格式 [seq_len, batch, embed_dim]
        x_seq = x.view(batch, C, seq_len).permute(2, 0, 1)

        # 1. 因果特征解耦
        causal_feat = self.feature_disentanglers[0](x_seq)
        non_causal_feat = self.feature_disentanglers[1](x_seq)

        # 2. 多假设生成（基于因果干预）
        hypotheses = []
        for i in range(self.num_hypotheses):
            # 对因果特征进行干预
            intervention = self.causal_intervention[i](causal_feat)
            intervened_feat = causal_feat + intervention

            # 生成假设
            h, _ = self.hypothesis_generators[i](
                intervened_feat,
                intervened_feat,
                intervened_feat
            )
            hypotheses.append(h)

        # 3. 反事实推理
        refined_hypotheses = []
        for i, h in enumerate(hypotheses):
            # 使用其他假设作为反事实参考
            other_hypotheses = torch.stack([hypotheses[j] for j in range(self.num_hypotheses) if j != i])
            ref = other_hypotheses.mean(dim=0)

            # 反事实注意力
            cf, _ = self.counterfactual_attention[i](h, ref, ref)
            refined_h = self.norms[0](h + self.dropout(cf))
            refined_hypotheses.append(refined_h)

        # 4. 假设融合
        stacked_hypotheses = torch.stack(refined_hypotheses, dim=-2)  # [seq_len, batch, num_hyp, embed_dim]
        flat_hypotheses = stacked_hypotheses.flatten(-2)  # [seq_len, batch, num_hyp*embed_dim]

        # 预测融合权重
        weights = self.fusion_weight_predictor(flat_hypotheses)  # [seq_len, batch, num_hyp]
        weights = weights.unsqueeze(-1)  # [seq_len, batch, num_hyp, 1]

        # 加权融合
        fused = (stacked_hypotheses * weights).sum(dim=-2)
        fused = self.norms[1](fused + x_seq)

        # 5. 前馈网络
        ff_out = self.ffn(self.norms[2](fused))
        fused = self.norms[3](fused + self.dropout(ff_out))

        # 转换回原始形状 [batch, C, H, W]
        fused = fused.permute(1, 2, 0).contiguous().view(batch, C, H, W)
        hypotheses_out = [h.permute(1, 2, 0).contiguous().view(batch, C, H, W)
                          for h in refined_hypotheses]

        return (*hypotheses_out, fused)


class CausalMultiHypothesisGraphTransformer(nn.Module):
    def __init__(self, num_layers=2, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalMultiHypothesisTransformerLayer(**kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        hypotheses = []
        for layer in self.layers:
            *h, x = layer(x)
            hypotheses.append(h)

        # 返回最后一层的多个假设和融合结果
        return (*hypotheses[-1], x)


model = CausalMultiHypothesisGraphTransformer(
    num_layers=3,
    embed_dim=256,
    num_heads=8,
    dropout=0.1,
    ff_dim=512,
    num_hypotheses=3
)

# 输入形状 [batch, 256, 8, 8]
x = torch.randn(2, 256, 8, 8)

# 输出为 (hypothesis1, hypothesis2, hypothesis3, fused_output)
h1, h2, h3, out = model(x)
print(h1.size())