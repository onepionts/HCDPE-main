

#实现了 HCDPE的头部模块 HCDPE_Head，用于姿态估计。
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (constant_init, normal_init)
from mmpose.models.builder import build_loss
from mmpose.models.heads.topdown_heatmap_base_head import TopdownHeatmapBaseHead
from mmpose.models.builder import HEADS

from .hcdpe_tokenizer import HCDPE_Tokenizer
from .modules import MixerLayer, FCBlock, BasicBlock


@HEADS.register_module()
class HCDPE_Head(TopdownHeatmapBaseHead):


    def __init__(self,stage_hcdpe,in_channels,image_size,num_joints,cls_head=None,tokenizer=None,loss_keypoint=None,):
        super().__init__()
        self.stage_hcdpe = stage_hcdpe
        self.image_size = image_size


        self.guide_ratio = tokenizer['guide_ratio']
        self.img_guide = self.guide_ratio > 0.0

        self.conv_channels = cls_head['conv_channels']
        self.hidden_dim = cls_head['hidden_dim']

        self.num_blocks = cls_head['num_blocks']
        self.hidden_inter_dim = cls_head['hidden_inter_dim']
        self.token_inter_dim = cls_head['token_inter_dim']
        self.dropout = cls_head['dropout']

        self.token_num = tokenizer['codebook']['token_num']  # token数量34
        print('token_num',self.token_num)
        self.token_class_num = tokenizer['codebook']['token_class_num']  # token类别数量2048
        print('token_class_num',self.token_class_num)

        if stage_hcdpe == "classifier":
            self.conv_trans = self._make_transition_for_head(in_channels, self.conv_channels)#卷积特征转换层。
            self.conv_head = self._make_cls_head(cls_head)#分类器头部卷积层

            input_size = (image_size[0]//32)*(image_size[1]//32)
            # 特征转换层。
            self.mixer_trans = FCBlock(self.conv_channels * input_size,self.token_num * self.hidden_dim)
            #混合层。
            self.mixer_head = nn.ModuleList([MixerLayer(self.hidden_dim, self.hidden_inter_dim,self.token_num, self.token_inter_dim,
                                                        self.dropout) for _ in range(self.num_blocks)])
            # 归一化层。
            self.mixer_norm_layer = FCBlock(self.hidden_dim, self.hidden_dim)
            # 线性分类层。
            self.cls_pred_layer = nn.Linear(self.hidden_dim, self.token_class_num)
        
        self.tokenizer = HCDPE_Tokenizer(
            stage_hcdpe=stage_hcdpe, tokenizer=tokenizer, num_joints=num_joints,
            guide_ratio=self.guide_ratio, guide_channels=in_channels,)


        self.loss = build_loss(loss_keypoint)

    #计算分类器阶段的损失，包括分类损失和关键点损失。
    def get_loss(self, p_logits, p_joints, g_logits, joints):
        """Calculate loss for training classifier.

        Note:
            batch_size: N
            num_keypoints: K
            num_token: M
            num_token_class: V

        Args:
            p_logits (torch.Tensor[NxMxV]): Predicted class logits.
            p_joints(torch.Tensor[NxKx3]): Predicted joints 
                recovered from the predicted class.
            g_logits(torch.Tensor[NxM]): Groundtruth class labels
                calculated by the well-trained tokenizer encoder 
                and groundtruth joints.
            joints(torch.Tensor[NxKx3]): Groundtruth joints.
        """

        losses = dict()
        
        losses['token_loss'], losses['kpt_loss'] = self.loss(\
            p_logits, p_joints, g_logits, joints)

        unused_losses = []
        for name, loss in losses.items():
            if loss == None:
                unused_losses.append(name)
        for unused_loss in unused_losses:
            losses.pop(unused_loss)
                
        return losses

    def forward(self, x, extra_x, joints=None, train=True):
        """Forward function."""
        
        if self.stage_hcdpe == "classifier":
            batch_size = x[-1].shape[0]

            cls_feat = self.conv_head[0](self.conv_trans(x[-1]))#change!

            print('M',cls_feat.shape)
            #二维特征图拉直成一维
            cls_feat = cls_feat.flatten(2).transpose(2,1).flatten(1)
            cls_feat = self.mixer_trans(cls_feat)
            cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)

            for mixer_layer in self.mixer_head:
                cls_feat = mixer_layer(cls_feat)
            cls_feat = self.mixer_norm_layer(cls_feat)
            print('cls_feat',cls_feat.shape)
            #分类层计算 MxV 的 logits
            cls_logits = self.cls_pred_layer(cls_feat)#这个层是一个全连接层，用于将 M 个特征转换为 MxV 的 logits：
            print('cls_logits', cls_logits.shape)
            encoding_scores = cls_logits.topk(1, dim=2)[0]
            cls_logits = cls_logits.flatten(0,1)
            #在 [V] 维度取 softmax:
            cls_logits_softmax = cls_logits.clone().softmax(1)
            print('cls_logits_softmax',cls_logits_softmax.shape)
        else:
            encoding_scores = None
            cls_logits = None
            cls_logits_softmax = None

        if not self.img_guide or \
            (self.stage_hcdpe == "classifier" and not train):
            joints_feat = None
        else:
            joints_feat = self.extract_joints_feat(extra_x[-1], joints)
        #用 softmax 出来的结果乘上 codebook 里的特征矩阵
        output_joints, cls_label, e_latent_loss = \
            self.tokenizer(joints, joints_feat, cls_logits_softmax, train=train)

        if train:
            return cls_logits, output_joints, cls_label, e_latent_loss
        else:
            return output_joints, encoding_scores
    #构建用于分类器阶段的特征转换层
    def _make_transition_for_head(self, inplanes, outplanes):
        # 定义一个包含卷积层、批量归一化层和ReLU激活函数的列表 第二阶段
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        # 使用nn.Sequential将上述列表中的层组合成一个序列
        return nn.Sequential(*transition_layer)
    #构建分类器阶段的卷积特征提取层
    def _make_cls_head(self, layer_config):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            layer_config['conv_channels'],
            layer_config['conv_channels'],
            layer_config['conv_num_blocks'],
            dilation=layer_config['dilation']
        )
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)
    #构建特征提取层中的基本块。
    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    #：从特征图中提取关键点的特征
    def extract_joints_feat(self, feature_map, joint_coords):
        assert self.image_size[1] == self.image_size[0], \
            'If you want to use a rectangle input, ' \
            'please carefully check the length and width below.'
        batch_size, _, _, height = feature_map.shape
        stride = self.image_size[0] / feature_map.shape[-1]
        joint_x = (joint_coords[:,:,0] / stride + 0.5).int()
        joint_y = (joint_coords[:,:,1] / stride + 0.5).int()
        joint_x = joint_x.clamp(0, feature_map.shape[-1] - 1)
        joint_y = joint_y.clamp(0, feature_map.shape[-2] - 1)
        joint_indices = (joint_y * height + joint_x).long()

        flattened_feature_map = feature_map.clone().flatten(2)
        joint_features = flattened_feature_map[
            torch.arange(batch_size).unsqueeze(1), :, joint_indices]

        return joint_features
    #初始化模型的权重。
    def init_weights(self):
        print("head")
        if self.stage_hcdpe == "classifier":
            self.tokenizer.eval()
            for name, params in self.tokenizer.named_parameters():
                params.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

