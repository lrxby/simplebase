# ---------------------------------------------------------------
# 1. 引入自定义 Pipeline (必须，否则找不到 LoadPseudoAnnotations)
# ---------------------------------------------------------------
custom_imports = dict(
    imports=['mmrotate.datasets.transforms.loading_pseudo'], 
    allow_failed_imports=False
)

_base_ = [
    '../_base_/datasets/dotav15.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

randomness = dict(seed=0, deterministic=False)

# model settings
model = dict(
    type='Point2RBoxV2',
    ss_prob=[0.68, 0.07, 0.25],
    copy_paste_start_epoch=6,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        strides=[8],
        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                    ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        post_process={11: 1.2},
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
            
        # ------------------ Loss 独立开关控制区 ------------------
        use_loss_bbox=True,       # 基础 GWD 回归 Loss
        use_loss_overlap=True,    # 高斯重叠 Loss
        use_loss_voronoi=True,    # 原始的在线注水池 Loss
        use_loss_bbox_edg=True,   # 边缘 Loss
        use_loss_ss=True,         # 自监督一致性 Loss
        use_loss_size=True,       # [新增] 透视感知尺寸 Loss
        use_loss_angle=True,      # [新增] 角度一致性 Loss
        use_loss_ourwater=True,   # [新增] 离线伪标签 OurWater Loss
        # --------------------------------------------------------

        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
            
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),
        loss_voronoi=dict(type='VoronoiWatershedLoss', loss_weight=5.0),
        loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),
        loss_ss=dict(type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),
        
        # === [修改] 将原来的 loss_naoa 重命名并更新类型为 AngleLoss ===
        loss_angle=dict(
            type='AngleLoss',
            loss_weight=1.0,     # 继承原 naoa 配置，若效果好可尝试调大至 2.0
            k_radius=2.0,        # 高斯邻域半径系数
            score_alpha=1.0,     # 置信度权重幂次
            target_classes=[0,4,5,6,7,8,10,12,14,15], # 指定计算类别。None 表示全类别开启
            topk=0.95            # 补充 TopK 松弛参数
        ),
        
        # === [修改] 将原来的 loss_perspective 重命名并更新类型为 SizeLoss ===
        loss_size=dict(
            type='SizeLoss', 
            loss_weight=1.0,
            beta=1.0,
            target_classes=[0,1,4,7,8,9,10,11,14,15],
            topk=0.95
        ),
        
        # === [新增] OurWaterLoss ===
        loss_ourwater=dict(
            type='OurWaterLoss',
            loss_weight=5.0      # 参考你之前的设置
        )
    ),
    
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# ---------------------------------------------------------------
# 2. 修改 Pipeline (打通离线伪标签数据流)
# ---------------------------------------------------------------
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    
    # === [关键插入] 加载离线生成的伪标签 ===
    dict(type='LoadPseudoAnnotations', 
         pkl_path='/mnt/data/xiekaikai/split_ss_dotav1.5/trainval/trainval.pkl'),
        # pkl_path='/mnt/data/liurunxiang/workplace/point2rbox-v2-baseline/dataset/dota.5/dota1.5.pkl'),
    # ========================================
    
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
        
    # === [修改] 将 pseudo_boxes 显式加入 meta_keys ===
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'scale_factor', 'flip', 'flip_direction', 'pseudo_boxes')
    )
]

train_dataloader = dict(batch_size=2,
                        dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]

# =======================================================
# [新增] 切换测试阶段为 Validation 集，并要求直接计算 mAP
# =======================================================
test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/mnt/data/xiekaikai/split_ss_dotav1.5/',
        # 👇 关键：把路径指向你的验证集！
        ann_file='trainval/annfiles/',         # 验证集的标签路径
        data_prefix=dict(img_path='trainval/images/'), # 验证集的图片路径
        test_mode=True,
        pipeline=_base_.test_pipeline))

# 👇 关键：关掉 format_only，要求直接计算并打印 mAP
test_evaluator = dict(type='DOTAMetric', metric='mAP')