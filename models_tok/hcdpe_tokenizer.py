# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from mmpose.models.builder import build_loss
from mmpose.models.builder import HEADS
from timm.models.layers import trunc_normal_
import numpy as np
from .modules import MixerLayer
    

@HEADS.register_module()
class HCDPE_Tokenizer(nn.Module):

    def __init__(self,
                 stage_hcdpe,
                 tokenizer=None,
                 num_joints=17,
                 guide_ratio=0,
                 guide_channels=0):
        super().__init__()

        self.stage_hcdpe = stage_hcdpe
        self.guide_ratio = guide_ratio
        self.num_joints = num_joints

        self.drop_rate = tokenizer['encoder']['drop_rate']     
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']#提取令牌类别数量。
        self.token_dim = tokenizer['codebook']['token_dim']#提取令牌维度。
        self.decay = tokenizer['codebook']['ema_decay']#提取指数移动平均（EMA）的衰减率。
        #
        #定义了一个可学习的参数，用于表示不可见的令牌。这个参数的维度是 (1, 1, self.enc_hidden_dim)，并且通过截断的正态分布进行初始化。
        self.invisible_token = nn.Parameter(
            torch.zeros(1, 1, self.enc_hidden_dim))
        trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)
        #函数对张量进行截断的正态分布初始化。这个函数通常用于初始化模型的参数，特别是在需要避免梯度爆炸或梯度消失的情况下。
        if self.guide_ratio > 0:
            self.start_img_embed = nn.Linear(guide_channels, int(self.enc_hidden_dim*self.guide_ratio))
        self.start_embed = nn.Linear(2, int(self.enc_hidden_dim*(1-self.guide_ratio)))#输入节点的位置，输出为节点的的隐藏表示，隐藏表示可以更好地发现坐标之间的关系，构成子结构
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(
            self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)

        self.register_buffer('codebook', 
            torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', 
            torch.zeros(self.token_class_num))
        self.register_buffer('ema_w', 
            torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_()

        #decoder_token_mlp 是一个线性层，用于将 token 解码为关键点坐标。
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)

        #decoder_start 是一个线性层，用于将 token 解码的起始点映射到解码器的隐藏维度。
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)
        #decoder 是一个由多个 MixerLayer 组成的解码器模块列表。
        # 每个 MixerLayer 的输入维度是 dec_hidden_dim，输出维度是 dec_hidden_dim，
        # 用于对 token 进行解码。

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        #decoder_layer_norm 是用于对解码器输出进行层归一化的层
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        self.loss = build_loss(tokenizer['loss_keypoint'])

    def forward(self, joints, joints_feature, cls_logits, train=True):
        """Forward function. """
        #如果模型处于训练模式或者当前阶段是 Tokenizer 阶段，
        # 则首先从输入 joints 中提取关节坐标和关节可见性信息，并获取批量大小。
        if train or self.stage_hcdpe == "tokenizer":
            joints_coord, joints_visible, bs \
                = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]#

            encode_feat = self.start_embed(joints_coord)
            #如果指定了图像引导比例 guide_ratio 大于0，则使用线性层
            #start_img_embed 对图像引导特征进行编码。这可以用于将图像特征引导到 HCDPE 模型中，以提高关节预测的准确性。
            if self.guide_ratio > 0:
                encode_img_feat = self.start_img_embed(joints_feature)
                encode_feat = torch.cat((encode_feat, encode_img_feat), dim=2)
            # 如果模型处于训练模式且当前阶段是 Tokenizer 阶段，
            # 则以 drop_rate 的概率随机屏蔽一部分关节的可见性信息，这有助于模型学习鲁棒性。
            if train and self.stage_hcdpe == "tokenizer":
                rand_mask_ind = torch.rand(
                    joints_visible.shape, device=joints.device) > self.drop_rate
                joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 
            # 通过一系列 MixerLayer 编码器块对特征进行编码，
            # 通过线性层 token_mlp 将编码后的特征映射到 token_num 大小的空间，并应用维度变换。
            mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1)
            w = joints_visible.unsqueeze(-1).type_as(mask_tokens)#w=？
            encode_feat = encode_feat * w + mask_tokens * (1 - w)
                    
            for num_layer in self.encoder:
                encode_feat = num_layer(encode_feat)
            encode_feat = self.encoder_layer_norm(encode_feat)
            
            encode_feat = encode_feat.transpose(2, 1)
            encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
            #通过线性层 feature_embed 将 token 表示映射到 token_dim 大小的特征空间。
            encode_feat = self.feature_embed(encode_feat).flatten(0,1)
            #计算编码后的特征与 codebook 中所有 token 的距离，并根据最近的距离获取对应的 token 编码。
            distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
                + torch.sum(self.codebook**2, dim=1) \
                - 2 * torch.matmul(encode_feat, self.codebook.t())
                
            encoding_indices = torch.argmin(distances, dim=1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self.token_class_num, device=joints.device)
            encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        else:
            bs = cls_logits.shape[0] // self.token_num
            print("bs",bs)
            encoding_indices = None
        #如果当前阶段是分类器阶段，则使用分类器输出 cls_logits 与 codebook 中的 token 来获取部分 token 特征表示。
        if self.stage_hcdpe == "classifier":
            part_token_feat = torch.matmul(cls_logits, self.codebook)
            print('部分token特征表示',part_token_feat.shape)
        else:
            part_token_feat = torch.matmul(encodings, self.codebook)
        #则通过指数移动平均（EMA）更新 codebook，并计算 token 特征的编码损失（e_latent_loss）。
        if train and self.stage_hcdpe == "tokenizer":
            # Updating Codebook using EMA
            #使用指数移动平均（EMA）方法更新 codebook。
            # 通过将 token 的编码（encodings）与编码特征（encode_feat）的点积作为更新值（dw）。
            dw = torch.matmul(encodings.t(), encode_feat.detach())
            # sync
            n_encodings, n_dw = encodings.numel(), dw.numel()
            #将编码和更新值转换为适当的形状，并合并成一个张量 combined。
            encodings_shape, dw_shape = encodings.shape, dw.shape
            combined = torch.cat((encodings.flatten(), dw.flatten()))
            #----------
            #在实际训练中，通常使用分布式计算来进行梯度同步，此处通过 dist.all_reduce 进行梯度同步，以确保不同设备上的计算结果一致。
            # dist.all_reduce(combined) # math sum
            sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
            sync_encodings, sync_dw = \
                sync_encodings.view(encodings_shape), sync_dw.view(dw_shape)

            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(sync_encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.token_class_num * 1e-5) * n)
            #根据同步后的编码信息和更新值，更新 codebook 中的聚类大小（ema_cluster_size）和权重矩阵（ema_w），并重新计算 codebook。
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
            self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
            print("************")
            #计算 token 特征的编码损失（e_latent_loss），用于模型训练过程中的优化。
            e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat)
            #将部分 token 特征表示（part_token_feat）调整为适当的形状，并送入解码器中进行解码。
            part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()
        else:
            e_latent_loss = None
        
        # Decoder of Tokenizer, Recover the joints.
        #将部分 token 特征表示（part_token_feat）转换为解码器的输入格式，
        # 并通过一系列 MixerLayer 解码器块对部分 token 特征进行解码，应用层归一化。

        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)

        #print(part_token_feat.size())
        print('1222222222222222')
        part_token_feat = part_token_feat.transpose(2,1)

        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)

        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)
        #最终，将解码特征映射回关节坐标空间，并返回重建的关节坐标、token 编码以及编码损失。

        return recoverd_joints, encoding_indices, e_latent_loss

    def get_loss(self, output_joints, joints, e_latent_loss):
        """Calculate loss for training tokenizer.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """

        losses = dict()

        kpt_loss, e_latent_loss = self.loss(output_joints, joints, e_latent_loss)

        losses['joint_loss'] = kpt_loss
        losses['e_latent_loss'] = e_latent_loss

        return losses

    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_hcdpe == "classifier"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_hcdpe == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')
