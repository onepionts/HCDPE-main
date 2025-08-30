import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

def sd(x):
    """
    计算输入数组x在第一个轴（通常是样本维度）上的标准差。
    """
    return np.std(x, axis=0, ddof=1)

def normalize(x):
    """
    对输入数组x进行标准化处理，通过减去均值并除以标准差。
    如果标准差为零，则将其设置为1，以避免除以零的情况。
    """
    mean = np.mean(x, axis=0)
    std = sd(x)
    std[std == 0] = 1
    x = (x - mean) / std
    return x


def log_adjust(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if np.any(x <= 0):
        raise ValueError("所有输入数必须大于0。")

    y = (-np.log(x)) / np.sum(-np.log(x))
    return y


def softmax_inverse_temp_adjust(x, lambda_=1.0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if np.any(x <= 0):
        raise ValueError("所有输入数必须大于0。")

    transformed = np.exp(-lambda_ * x)
    y = transformed / np.sum(transformed)
    return y


def random_fourier_features_gpu(x, w=None, b=None, num_f=1, sum_agg=True, sigma=1, seed=None, device='cpu'):
    """
    在GPU上生成随机傅里叶特征，用于近似核函数。
    """
    '''
    seed = 19980125  # my birthday, :)
    random.seed(seed)
    '''

    if seed is not None:
        torch.manual_seed(seed)
    n, r = x.size(0), x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma == 0:
        sigma = 1
    if w is None:
        w = (1 / sigma) * torch.randn(num_f, c, device=device)
    if b is None:
        b = 2 * np.pi * torch.rand(r, num_f, device=device)
        b = b.unsqueeze(0).repeat(n, 1, 1)  # 形状为 (n, r, num_f)

    # 确保 sqrt 的输入是张量
    Z = torch.sqrt(torch.tensor(2.0 / num_f, device=device, dtype=x.dtype))

    mid = torch.matmul(x, w.t())  # 形状为 (n, r, num_f)
    mid = mid + b
    mid = mid - mid.min(dim=1, keepdim=True)[0]
    mid = mid / mid.max(dim=1, keepdim=True)[0]
    mid = mid * (np.pi / 2.0)

    if sum_agg:
        Z = Z * (torch.cos(mid) + torch.sin(mid))
    else:
        Z = Z * torch.cat((torch.cos(mid), torch.sin(mid)), dim=-1)

    return Z

def cov(x, w=None):
    """
    计算输入张量x的协方差矩阵。如果提供了权重w，则计算加权协方差矩阵。
    """
    if w is None:
        n = x.shape[0]
        cov_matrix = (x.t() @ x) / n
        e = torch.mean(x, dim=0).view(-1, 1)  # 使用 dim 而不是 axis
        res = cov_matrix - e @ e.t()
    else:
        if w.size(0) != x.size(0):
            raise ValueError(f"Weight tensor size {w.size(0)} does not match x tensor size {x.size(0)}")
        w = w.view(-1, 1)
        cov_matrix = (w * x).t() @ x
        e = torch.sum(w * x, dim=0).view(-1, 1)  # 使用 dim 而不是 axis
        res = cov_matrix - e @ e.t()
    return res

def lossb_expect_func(cfeaturec, weight, num_f=1, sum_agg=True, device='cpu'):
    """
    计算基于协方差的自定义损失函数。
    """
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum_agg=sum_agg, device=device)
    loss = torch.zeros(1, device=device)
    weight = weight.to(device)
    for i in range(cfeaturecs.size(-1)):
        cfeaturec_i = cfeaturecs[:, :, i]
        cov1 = cov(cfeaturec_i, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)
    return loss

def weight_learner(cfeatures1, features1, global_epoch=0, iter=0, device='cuda'):
    """
    学习每个样本的权重，以最小化自定义的损失函数。
    """
    cfeatures = cfeatures1.clone().detach()
    features = features1.clone().detach()
    n = cfeatures.size(0)
    cfeatures_flat = cfeatures.view(n, -1)
    features_flat = features.view(n, -1)

    # 可选：标准化特征
    cfeatures_flat_np = normalize(cfeatures_flat.cpu().detach().numpy())
    features_flat_np = normalize(features_flat.cpu().detach().numpy())
    cfeatures_flat = torch.from_numpy(cfeatures_flat_np).to(device).float().requires_grad_(True)
    features_flat = torch.from_numpy(features_flat_np).to(device).float().requires_grad_(True)
    cfeatures = cfeatures_flat
    features = features_flat
    softmax = nn.Softmax(dim=0)
    weight = torch.ones(cfeatures.size(0), 1, device=device, requires_grad=True)
    # 拼接特征，修正为沿特征维度拼接
    all_feature = torch.cat([cfeatures, features.detach()], dim=1)
    optimizerbl = torch.optim.SGD([weight], lr=3.0e-3, momentum=0.9)

    # 超参数
    num_f = 1
    sum_agg = True
    decay_pow = 2
    lambda_decay_rate = 1
    lambdap = 3
    lambda_decay_epoch = 5
    min_lambda_times = 0.01
    optimizerbl.zero_grad()
    for epoch in range(20):
        current_softmax = softmax(weight)
        lossb = lossb_expect_func(all_feature, current_softmax, num_f, sum_agg, device=device)
        lossp = (current_softmax.pow(decay_pow)).sum()
        lambdap = lambdap * max(
            (lambda_decay_rate ** (global_epoch // lambda_decay_epoch)),
            min_lambda_times
        )
        lossg = lossb / lambdap + lossp
        lossg.backward()
        optimizerbl.step()
        '''
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {lossg.item()}")
        '''
    softmax_weight = softmax(weight).detach()
    softmax_weight_adj = log_adjust(softmax_weight.cpu().numpy())
    softmax_weight_adj = torch.from_numpy(softmax_weight_adj)
    softmax_weight_adj = softmax_weight_adj.to(device)
    return softmax_weight_adj

if __name__ == '__main__':
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    npz_file_1 = 'D:\\Research\\HCDPE\\models\\cls1.npy'
    cls_1 = np.load(npz_file_1)
    npz_file_h1 = 'D:\\Research\\HCDPE\\models\\cls_h1.npy'
    cls_h1 = np.load(npz_file_h1)
    npz_file_h2 = 'D:\\Research\\HCDPE\\models\\cls_h2.npy'
    cls_h2 = np.load(npz_file_h2)
    n = 10 # 样本数量
    cls_1 = cls_1[:n]
    cls_h1 = cls_h1[:n]
    cls_h2 = cls_h2[:n]
    # 定义样本数量和特征维度
    cfeatures = torch.from_numpy(cls_1)
    features = torch.from_numpy((cls_h1+cls_h2)/2)
    c = 34  # 通道数
    h = 64
    w = 8  # 高和宽

    # 创建随机特征数据，形状为 (n, 34, 64, 8)
    #cfeatures = torch.randn(n, c, h, device=device)
    #features = torch.randn(n, c, h, device=device)

    # 将特征展平成 (n, 34*64*8) = (n, 17408)
    cfeatures_flat = cfeatures.view(n, -1)
    features_flat = features.view(n, -1)

    # 可选：标准化特征
    cfeatures_flat_np = normalize(cfeatures_flat.cpu().detach().numpy())
    features_flat_np = normalize(features_flat.cpu().detach().numpy())
    cfeatures_flat = torch.from_numpy(cfeatures_flat_np).to(device).float()
    features_flat = torch.from_numpy(features_flat_np).to(device).float()

    # 调用权重学习函数
    learned_weights ,learned_weights_adj= weight_learner(cfeatures_flat, features_flat, global_epoch=0, iter=0, device=device)
    #learned_weights_adj = log_adjust(learned_weights.cpu().numpy())
    #learned_weights_adj_sf = softmax_inverse_temp_adjust(learned_weights.cpu().numpy())
    #learned_weights_f = torch.from_numpy(learned_weights_adj)
    #learned_weights_f = learned_weights_f.to(device)
    print("学习到的权重：", learned_weights,learned_weights_adj)