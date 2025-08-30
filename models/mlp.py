import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim=192 * 17 * 2, output_dim=64 * 17 * 2, hidden_dims=[2048, 1024]):
        super(MLP, self).__init__()

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
        # x 的形状假设为 (batch_size, 192, 17, 2)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # 展平成 (batch_size, 6528)
        x = self.network(x)  # 通过 MLP 得到 (batch_size, 2176)
        x = x.view(batch_size, 64, 17, 2)  # 重新调整为 (batch_size, 64, 17, 2)
        return x


# 示例用法
if __name__ == "__main__":
    # 创建一个 MLP 实例
    model = MLP()
    print(model)

    # 创建一个示例输入张量，batch_size=8
    input_tensor = torch.randn(8, 192, 17, 2)

    # 通过模型进行前向传播
    output = model(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("输出形状:", output.shape)