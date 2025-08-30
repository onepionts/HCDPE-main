_base_ = ['./coco.py']  # 基础配置文件
log_level = 'INFO'  # 日志级别
load_from = None  # 不从预训练模型加载
resume_from = None  # 不从之前的检查点恢复
dist_params = dict(backend='nccl')  # 分布式训练参数
workflow = [('train', 1)]  # 工作流程：训练一个周期
find_unused_parameters = False  # 不查找未使用的参数
checkpoint_config = dict(interval=5, create_symlink=False)  # 检查点配置，每5个周期保存一次检查点
evaluation = dict(interval=10, metric='mAP', save_best='AP')  # 评估配置，每个周期进行一次评估，使用mAP（平均精度）指标，保存最佳AP模型


optimizer = dict(
    type='AdamW',  # 优化器类型为AdamW
    lr=8e-4,  # 学习率
    betas=(0.9, 0.999),  # AdamW参数
    weight_decay=0.05,  # 权重衰减
    constructor='SwinLayerDecayOptimizerConstructor',  # 构造器
    paramwise_cfg=dict(
        num_layers=[2, 2, 18, 2],  # 每层的参数
        layer_decay_rate=0.9,  # 层衰减率
        no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale']  # 不进行衰减的参数名
    )
)

optimizer_config = dict(grad_clip=None)  # 优化器配置，无梯度裁剪

lr_config = dict(
    policy='CosineAnnealing',  # 学习率策略为余弦退火
    warmup='linear',  # 预热策略为线性
    warmup_iters=500,  # 预热迭代次数
    warmup_ratio=0.00001,  # 预热初始学习率
    min_lr_ratio=1e-5  # 最小学习率
)
total_epochs = 210  # 总训练周期数


log_config = dict(
    interval=2,  # 日志间隔
    hooks=[
        dict(type='TextLoggerHook')  # 文本日志挂钩
    ]
)


channel_cfg = dict(
    num_output_channels=17,  # 输出通道数
    dataset_joints=17,  # 数据集中的关键点数
    dataset_channel=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],  # 数据集通道
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 推理通道
)

data_cfg = dict(
    image_size=[256, 256],  # 输入图像尺寸
    heatmap_size=[64, 64],  # 热图尺寸
    num_output_channels=channel_cfg['num_output_channels'],  # 输出通道数
    num_joints=channel_cfg['dataset_joints'],  # 关键点数
    dataset_channel=channel_cfg['dataset_channel'],  # 数据集通道
    inference_channel=channel_cfg['inference_channel'],  # 推理通道
    soft_nms=False,  # 是否使用软NMS
    nms_thr=1.0,  # NMS阈值
    oks_thr=0.9,  # OKS阈值
    vis_thr=0.2,  # 可视化阈值
    use_gt_bbox=True,  # 是否使用GT边界框
    det_bbox_thr=0.0,  # 检测边界框阈值
    bbox_file=None  # 边界框文件
)

# model settings
model = dict(
    type='HCDPE',  # 模型类型
    #pretrained="weights/swin_base.pth",  # 检查点路径
    #pretrained="utils/weight/best_AP_epoch_137.pth",  # 检查点路径
    backbone=dict(
        type='SwinV2TransformerRPE2FC',  # 主干网络类型
        embed_dim=128,  # 嵌入维度
        depths=[2, 2, 18, 2],  # 每层深度
        num_heads=[4, 8, 16, 32],  # 每层头数
        window_size=[16, 16, 16, 8],  # 窗口大小
        pretrain_window_size=[12, 12, 12, 6],  # 预训练窗口大小
        ape=False,  # 是否使用绝对位置编码
        drop_path_rate=0.3,  # 丢弃路径率
        patch_norm=True,  # 是否使用补丁归一化
        use_checkpoint=True,  # 是否使用检查点
        rpe_interpolation='geo',  # RPE插值方式
        use_shift=[True, True, False, False],  # 是否使用shift
        relative_coords_table_type='norm8_log_bylayer',  # 相对坐标表类型
        attn_type='cosine_mh',  # 注意力类型
        rpe_output_type='sigmoid',  # RPE输出类型
        postnorm=True,  # 是否使用后规范化
        mlp_type='normal',  # MLP类型
        out_indices=(3,),  # 输出索引
        patch_embed_type='normal',  # 补丁嵌入类型
        patch_merge_type='normal',  # 补丁合并类型
        strid16=False,  # 是否使用stride 16
        frozen_stages=5,  # 冻结阶段
    ),
    keypoint_head=dict(
        type='HCDPE_Head',  # 关键点头部类型
        stage_hcdpe='classifier',  # HCDPE阶段
        in_channels=1024,  # 输入通道数
        image_size=data_cfg['image_size'],  # 图像尺寸
        num_joints=channel_cfg['num_output_channels'],  # 关键点数
        loss_keypoint=dict(
            type='Classifer_loss',  # 损失类型
            token_loss=1.0,  # token损失
            joint_loss=1.0  # 关键点损失
        ),
        cls_head=dict(
            conv_num_blocks=2,  # 卷积块数量
            conv_channels=256,  # 卷积通道数
            dilation=1,  # 膨胀率
            num_blocks=4,  # 块数量
            hidden_dim=64,  # 隐藏层维度
            token_inter_dim=64,  # token中间维度
            hidden_inter_dim=256,  # 隐藏中间维度
            dropout=0.0  # Dropout率
        ),
        tokenizer=dict(
            guide_ratio=0.5,  # 引导比例
            ckpt=r"D:\\code\\HCDPE\\utils\\weight\\tokenizer\\swin_base.pth",  # 检查点路径
            encoder=dict(
                drop_rate=0.2,  # 丢弃率
                num_blocks=4,  # 块数量
                hidden_dim=512,  # 隐藏层维度
                token_inter_dim=64,  # token中间维度
                hidden_inter_dim=512,  # 隐藏中间维度
                dropout=0.0  # Dropout率
            ),
            decoder=dict(
                num_blocks=1,  # 块数量
                hidden_dim=32,  # 隐藏层维度
                token_inter_dim=64,  # token中间维度
                hidden_inter_dim=64,  # 隐藏中间维度
                dropout=0.0  # Dropout率
            ),
            codebook=dict(
                token_num=34,  # token数量
                token_dim=512,  # token维度
                token_class_num=2048,  # token类别数量
                ema_decay=0.9  # EMA衰减
            ),
            loss_keypoint=dict(
                type='Tokenizer_loss',  # 损失类型
                joint_loss_w=1.0,  # 关键点损失权重
                e_loss_w=15.0,  # e损失权重
                beta=0.05  # beta值
            )
        )
    ),
    test_cfg=dict(
        flip_test=True,  # 是否进行翻转测试
        dataset_name='COCO'  # 数据集名称
    )
)

#
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='TopDownGetBboxCenterScale', padding=1.25),  # 获取边界框中心尺度
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),  # 随机平移边界框中心
    dict(type='TopDownRandomFlip', flip_prob=0.5),  # 随机翻转
    dict(
        type='TopDownHalfBodyTransform',  # 半身变换
        num_joints_half_body=8,
        prob_half_body=0.3
    ),
    dict(
        type='TopDownGetRandomScaleRotation',
        rot_factor=40,
        scale_factor=0.5
    ),
    dict(type='TopDownAffine'),  # 仿射变换
    dict(
        type='Albumentation',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2,
                p=1.0
            ),
            dict(
                type='GridDropout',
                unit_size_min=10,
                unit_size_max=40,
                random_offset=True,
                p=0.5
            )
        ]
    ),
    dict(type='ToTensor'),  # 转换为张量
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(
        type='Collect',
        keys=['img', 'joints_3d', 'joints_3d_visible'],
        meta_keys=[
             'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]
    )
]

val_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='TopDownGetBboxCenterScale', padding=1.12),  # 获取边界框中心尺度
    dict(type='TopDownAffine'),  # 仿射变换
    dict(type='ToTensor'),  # 转换为张量
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
             'image_file','center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]
    )
]

test_pipeline = val_pipeline  # 测试管道与验证管道相同

data_root = r"D:\\code\\HCDPE\\data\\v7_ex\\"  # 数据根目录
data = dict(
    samples_per_gpu=256,  # 每个GPU的样本数
    workers_per_gpu=2,  # 每个GPU的工作线程数
    val_dataloader=dict(samples_per_gpu=32),  # 验证数据加载器
    test_dataloader=dict(samples_per_gpu=32),  # 测试数据加载器
    train=dict(
        type='TopDownCocoDataset',  # 数据集类型
        ann_file=f'{data_root}/annotations/train//train.json',  # 训练注释文件
        img_prefix=f'{data_root}/images/train/',  # 训练图像前缀
        data_cfg=data_cfg,  # 数据配置
        pipeline=train_pipeline,  # 训练管道
        dataset_info={{_base_.dataset_info}}  # 数据集信息
    ),
    val=dict(
        type='TopDownCocoDataset',  # 数据集类型
        ann_file=f'{data_root}/annotations/val/val.json',  # 验证注释文件
        img_prefix=f'{data_root}/images/val/',  # 验证图像前缀
        data_cfg=data_cfg,  # 数据配置
        pipeline=val_pipeline,  # 验证管道
        dataset_info={{_base_.dataset_info}}  # 数据集信息
    ),
    test=dict(
        type='TopDownCocoDataset',  # 数据集类型
        ann_file=f'{data_root}/annotations/val/val.json',  # 验证注释文件
        img_prefix=f'{data_root}/images/val/',  # 验证图像前缀
        data_cfg=data_cfg,  # 数据配置
        pipeline=val_pipeline,  # 测试管道
        dataset_info={{_base_.dataset_info}}  # 数据集信息
    )

)
'''
data_root = 'data/coco/v7_ex'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/train//train.json',
        img_prefix=f'{data_root}/images/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/val/val.json',
        img_prefix=f'{data_root}/images/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/val/val.json',
        img_prefix=f'{data_root}/images/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}})
)
'''
