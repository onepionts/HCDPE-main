_base_ = ['./coco.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
find_unused_parameters=False
checkpoint_config = dict(interval=5, create_symlink=False)
evaluation = dict(interval=1, metric='mAP', save_best='AP')

optimizer = dict(type='AdamW', lr=1e-2, betas=(0.9, 0.999), weight_decay=0.15,
                 constructor='SwinLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=[2, 2, 18, 2], layer_decay_rate=0.9,
                                    no_decay_names=['relative_position_bias_table',
                                                    'rpe_mlp',
                                                    'logit_scale']))

optimizer_config = dict(grad_clip=None)

# optim_wrapper = dict(
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone.layer0': dict(lr_mult=0, decay_mult=0),
#             'backbone.layer0': dict(lr_mult=0, decay_mult=0),
#         }))

# learning policy
lr_config = dict(  # 学习率的设置
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)

total_epochs = 150

log_config = dict(   # 日志的配置
    # interval=100,
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(  # 定义了通道的配置，包括输出通道的数量、数据集中关节点的数量，以及数据集通道和推断通道的配置
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

data_cfg = dict( # 是数据的配置，主要用于姿态估计模型中的数据处理部分。它包含了图像和热图的尺寸信息，输出通道和关节点的数量，数据集通道和推断通道的配置，以及一些其他的参数，如非极大值抑制（NMS）的阈值、使用真实边界框还是检测边界框等
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    # bbox_file='data/coco/person_detection_results/'
    # 'COCO_val2017_detections_AP_H_56_person.json',
    # bbox_file='data/coco/v1/annotations/train/'
    # 'train.json',
    bbox_file=None
)

# model settings
model = dict(
    type='HCDPE',
    #pretrained='./utils/weight/tokenizer/swin_base.pth',
    pretrained=r'D:\\code\\HCDPE\\utils\\weight\\tokenizer\\swin_base.pth',
    backbone=dict(
        type='SwinV2TransformerRPE2FC',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[16, 16, 16, 8],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        rpe_interpolation='geo',
        use_shift=[True, True, False, False],
        relative_coords_table_type='norm8_log_bylayer',
        attn_type='cosine_mh',
        rpe_output_type='sigmoid',
        postnorm=True,
        mlp_type='normal',
        out_indices=(3,),
        patch_embed_type='normal',
        patch_merge_type='normal',
        strid16=False,
        frozen_stages=5,
        # frozen_stages=-1,
        # init_cfg=dict(
        #     type='Pretrained', # 预训练参数，只加载backbone权重用于迁移学习
        #     prefix='backbone.',
        #     checkpoint='weights/tokenizer/swin_base.pth'),
    ),
    keypoint_head=dict(
        type='HCDPE_Head',
        stage_hcdpe='tokenizer',
        in_channels=1024,
        image_size=data_cfg['image_size'],
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(
            type='Classifer_loss',
            token_loss=1.0,
            joint_loss=1.0),
        cls_head=dict(
            conv_num_blocks=2,
            conv_channels=256,
            dilation=1,
            num_blocks=4,
            hidden_dim=64,
            token_inter_dim=64,
            hidden_inter_dim=256,
            dropout=0.0),
        tokenizer=dict(
            guide_ratio=0.5,
            ckpt="",
            encoder=dict(
                drop_rate=0.2,
                num_blocks=4,
                hidden_dim=512,
                token_inter_dim=64,
                hidden_inter_dim=512,
                dropout=0.0,
            ),
            decoder=dict(
                num_blocks=1,
                hidden_dim=32,
                token_inter_dim=64,
                hidden_inter_dim=64,
                dropout=0.0,
            ),
            codebook=dict(
                token_num=40,  # Token数量为34
                token_dim=512, # Token维度为512
                token_class_num=4096, # Token类别数量为2048
                ema_decay=0.9, # 指数移动平均的衰减率为0.9
            ),

            loss_keypoint=dict(
                type='Tokenizer_loss',
                joint_loss_w=1.0, 
                e_loss_w=15.0,
                beta=0.05,)
            )),

    test_cfg=dict(
        flip_test=True,
        dataset_name='COCO'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(
        type='Albumentation',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2,
                p=1.0),
            dict(
                type='GridDropout',
                unit_size_min=10,
                unit_size_max=40,
                random_offset=True,
                p=0.5),
        ]),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img', 'joints_3d', 'joints_3d_visible'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.12),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img', 'joints_3d', 'joints_3d_visible'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = r'D:\\code\\HCDPE\\data\\'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='TopDownCocoDataset',
        # ann_file=f'{data_root}/v5/v5_1/train/output_coco/train.json',
        # img_prefix=f'{data_root}/v5/v5_1/train/',
        #ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        #img_prefix=f'{data_root}/images/train2017/train2017/',
        ann_file=f'{data_root}/v7_ex/annotations/train/train.json',
        img_prefix=f'{data_root}/v7_ex/images/train/',
        # ann_file=f'{data_root}/image3/train\output_coco/coco_sample1.json',
        # img_prefix=f'{data_root}/image3/train/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/v7_ex/annotations/val/val.json',
        img_prefix=f'{data_root}/v7_ex/images/val/',
        # ann_file=f'{data_root}/v5/v5_1/test/output_coco/val.json',
        # img_prefix=f'{data_root}/v5/v5_1/test/',
        #ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        #img_prefix=f'{data_root}/images/val2017/val2017/',
        # ann_file=f'{data_root}/image3/val\output_coco/coco_sample1.json',
        # img_prefix=f'{data_root}/image3/val/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='TopDownCocoDataset',
        # ann_file=f'{data_root}/v4/image1/output_coco/val.json',
        # img_prefix=f'{data_root}/v4/image1/',
        ann_file=f'{data_root}/v7_ex/annotations/val/val.json',
        img_prefix=f'{data_root}/v7_ex/images/val/',
        # ann_file=f'{data_root}/v4/image1/output_coco/coco_sample1.json',
        # img_prefix=f'{data_root}/v4/image1/',
        #ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        #img_prefix=f'{data_root}/images/val2017/val2017/',
        # ann_file=f'{data_root}/v5/v5_1/test/output_coco/val.json',
        # img_prefix=f'{data_root}/v5/v5_1/test/',
        # ann_file=f'{data_root}/image3_1/output_coco/coco_sample1.json',
        # img_prefix=f'{data_root}/image3_1/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}})
)