
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import LOSSES

#实现了平滑的 L1 损失函数，用于关键点的损失计算。包括可见性加权。
@LOSSES.register_module()
class JointS1Loss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    #用于关键点的损失计算
    def smooth_l1_loss(self, pred, gt):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < self.beta
        loss = torch.where(cond, 0.5*l1_loss**2/self.beta, l1_loss-0.5*self.beta)
        return loss

    def forward(self, pred, gt):

        joint_dim = gt.shape[2] - 1#确定关节维度的大小，gt 是真实值张量，其形状为 [batch_size, num_joints, num_dimensions]，其中 num_dimensions 包括了关节位置和可见性标志。
        visible = gt[..., joint_dim:]#提取出 gt 张量中的可见性标志，即最后一个维度。这个标志用于确定每个关节是否可见，如果可见则为1，不可见则为0。
        pred, gt = pred[..., :joint_dim], gt[..., :joint_dim]#从 pred 和 gt 中分别提取出关节位置信息，即前 joint_dim 维度，以便计算损失。
 
        loss = self.smooth_l1_loss(pred, gt) * visible#使用平滑的 L1 损失函数 smooth_l1_loss 计算关节位置的损失，然后乘以可见性标志，将不可见的关节位置的损失置为0。
        loss = loss.mean(dim=2).mean(dim=1).mean(dim=0)#对损失张量进行维度的缩减操作，先沿着关节维度（dim=2）求均值，然后沿着批次维度（dim=1）求均值，最后沿着批次维度（dim=0）求均值，得到最终的损失值。

        return loss

#定义了 Tokenizer 阶段的损失函数，由关键点损失和编码损失组成。
@LOSSES.register_module()
class Tokenizer_loss(nn.Module):
    def __init__(self, joint_loss_w, e_loss_w, beta=0.05):#接受三个参数：joint_loss_w（关节损失权重）、e_loss_w（编码器潜在损失权重）和可选参数 beta（平滑 L1 损失函数的参数）。
        super().__init__()

        self.joint_loss = JointS1Loss(beta)#创建一个 JointS1Loss 的实例，用于计算关节损失。
        self.joint_loss_w = joint_loss_w#将关节损失权重存储在类中，以便在前向传播方法中使用。

        self.e_loss_w = e_loss_w#将编码器潜在损失权重存储在类中，以便在前向传播方法中使用。
    #接受三个参数：output_joints（模型输出的关节位置）、joints（真实的关节位置）和 e_latent_loss（编码器潜在损失）。
    def forward(self, output_joints, joints, e_latent_loss):

        losses = []
        joint_loss = self.joint_loss(output_joints, joints)
        joint_loss *= self.joint_loss_w
        losses.append(joint_loss)

        e_latent_loss *= self.e_loss_w
        losses.append(e_latent_loss)

        return losses

#定义了 Classifier 阶段的损失函数，由分类损失和关键点损失组成。
@LOSSES.register_module()
class Classifer_loss(nn.Module):
    def __init__(self, token_loss=1.0, joint_loss=1.0, beta=0.05):
        super().__init__()

        self.token_loss = nn.CrossEntropyLoss()#初始化交叉熵损失函数，用于计算分类损失。
        self.token_loss_w = token_loss

        self.joint_loss = JointS1Loss(beta=beta)#创建一个 JointS1Loss 的实例，用于计算关节损失。
        self.joint_loss_w = joint_loss
    #接受四个参数：p_logits（模型预测的类别标签）、p_joints（模型预测的关节位置）、g_logits（真实的类别标签）和 joints（真实的关节位置）。
    def forward(self, p_logits, p_joints, g_logits, joints):

        losses = []
        if self.token_loss_w > 0:
            token_loss = self.token_loss(p_logits, g_logits)
            token_loss *= self.token_loss_w
            losses.append(token_loss)
        else:
            losses.append(None)
        
        if self.joint_loss_w > 0:
            joint_loss = self.joint_loss(p_joints, joints)
            joint_loss *= self.joint_loss_w
            losses.append(joint_loss)
        else:
            losses.append(None)
            
        return losses