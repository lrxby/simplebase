# ---------------------------------------------------------------
# 1. 引入自定义 Pipeline (必须，否则找不到 LoadPseudoAnnotations)
# ---------------------------------------------------------------
custom_imports = dict(
    imports=['mmrotate.datasets.transforms.loading_pseudo'], 
    allow_failed_imports=False
)

_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

randomness = dict(seed=0, deterministic=False)

# model settings
model = dict(
    type='Point2RBoxV2',
    
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
        
        # ------------------ Loss 开关控制 (全家桶) ------------------
        use_bbox_loss=True,      # 基础回归 GDLoss
        use_size_loss=True,     # SizeLoss
        use_angle_loss=True,    # AngleLoss
        use_voronoi_loss=False,   # [新增] 控制 VoronoiLoss 是否生效
        use_ourwater_loss=True,  # [新增] 控制 OurWaterLoss 是否生效
        # -----------------------------------------------------------
        
        # 几何参数 (Voronoi & Coder)
        voronoi_type='gaussian-orientation', 
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11], 
        
        # post_process={11: 1.2}, 
        
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        
        # ------------------ Loss 实例配置 ------------------
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
            
        # 1. 基础回归 (GDLoss)
        loss_bbox=dict(
            type='GDLoss',
            loss_type='gwd',
            loss_weight=5.0),

        # 2. 下界约束 (Voronoi)
        loss_voronoi=dict(
            type='VoronoiWatershedLoss', 
            loss_weight=5.0), 
            
        # 3. 离线伪标签监督 (OurWater - Row 5/6) [新增]
        loss_ourwater=dict(
            type='OurWaterLoss',
            loss_weight=3.0
        ),

        # 4. 上界/透视约束 (Size)
        loss_size=dict(
            type='SizeLoss', 
            loss_weight=1.0, 
            beta=1.0,
            topk=0.95,
            target_classes=[0,1,4,7,8,9,10,11,14]
            ),
            
        # 5. 局部一致性约束 (Angle)
        loss_angle=dict(
            type='AngleLoss',
            loss_weight=1.0, 
            k_radius=2.0,
            score_alpha=1.0,
            topk=0.95,
            target_classes=[0,4,5,6,7,8,10,12,14]
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
# 2. 修改 Pipeline (加载离线数据)
# ---------------------------------------------------------------
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    
    # === [关键插入] 加载离线生成的伪标签 ===
    # 请务必修改 pkl_path 为您实际生成的路径
    dict(type='LoadPseudoAnnotations', 
        #  pkl_path='/mnt/data/xiekaikai/split_ss_dota/trainval/trainval-ori-pca.pkl'),
        pkl_path='/mnt/data/liurunxiang/dataset/split_ss_dota/trainval/dota1-rect.pkl'),
    # ========================================

    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # 转换为点监督模式
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    # 注意：这里 PackDetInputs 需要确保能打包伪标签。
    # 如果伪标签已加入 gt_instances，则不需要额外操作。
    # 否则可能需要自定义 PackDetInputs 或在 loading_pseudo 中妥善处理。
    dict(
        type='mmdet.PackDetInputs',
        # [关键] 将 pseudo_boxes 加入 meta_keys，否则会被过滤掉
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'scale_factor', 'flip', 'flip_direction', 'pseudo_boxes')
    )
]

train_dataloader = dict(
    batch_size=2, 
    dataset=dict(pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

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
        data_root='/mnt/data/liurunxiang/dataset/split_ss_dota/',
        # 👇 关键：把路径指向你的验证集！
        ann_file='trainval/labelTxt/',         # 验证集的标签路径
        data_prefix=dict(img_path='trainval/images/'), # 验证集的图片路径
        test_mode=True,
        pipeline=_base_.test_pipeline))

# 👇 关键：关掉 format_only，要求直接计算并打印 mAP
test_evaluator = dict(type='DOTAMetric', metric='mAP')