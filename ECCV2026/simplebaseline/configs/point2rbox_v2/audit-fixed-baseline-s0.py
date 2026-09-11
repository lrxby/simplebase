# ---------------------------------------------------------------
# audit-fixed-baseline-s0.py
# 修复后基线 (伪标签几何同步 + 缺失伪标签处理 + 角度评估指标)
# 仅修改数据管线相关 bug，不改变任何损失数学目标/权重/优化器/训练长度/lr schedule
# seed=0, 12epoch, lr=5e-5, batch=2, LinearLR 500iter + MultiStepLR[8,11]
# ---------------------------------------------------------------
custom_imports = dict(
    imports=['mmrotate.datasets.transforms.loading_pseudo',
             'mmrotate.datasets.transforms.pseudo_box_sync'],
    allow_failed_imports=False)

_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

randomness = dict(seed=0, deterministic=False)

# model settings (与 point2rbox_v2-1x-dota.py 完全一致)
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

        # ------------------ Loss 开关控制 (与基线一致) ------------------
        use_bbox_loss=True,
        use_size_loss=True,
        use_angle_loss=True,
        use_voronoi_loss=False,
        use_ourwater_loss=True,
        # ---------------------------------------------------------------

        voronoi_type='gaussian-orientation',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],

        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),

        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        loss_bbox=dict(
            type='GDLoss',
            loss_type='gwd',
            loss_weight=5.0),

        loss_voronoi=dict(
            type='VoronoiWatershedLoss',
            loss_weight=5.0),

        loss_ourwater=dict(
            type='OurWaterLoss',
            loss_weight=1.0),

        loss_size=dict(
            type='SizeLoss',
            loss_weight=1.0,
            beta=1.0,
            topk=0.95,
            target_classes=[0, 1, 4, 7, 8, 9, 10, 11, 14]),

        loss_angle=dict(
            type='AngleLoss',
            loss_weight=1.0,
            k_radius=2.0,
            score_alpha=1.0,
            topk=0.95,
            target_classes=[0, 4, 5, 6, 7, 8, 10, 12, 14])
    ),

    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# ---------------------------------------------------------------
# 修复后的 train_pipeline:
# LoadPseudoAnnotations -> ConvertBoxType -> ConvertWeakSupervision
#   -> Resize -> RandomFlip -> PseudoBoxSync -> PackDetInputs
# PseudoBoxSync 在 RandomFlip 之后、PackDetInputs 之前:
#   先缩放 (scale_factor), 再用缩放后的 img_shape 翻转
#   horizontal/vertical: theta -> -theta; diagonal: theta 不变 (模 pi), x/y 都翻
# ---------------------------------------------------------------
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='LoadPseudoAnnotations',
         pkl_path='/mnt/data/liurunxiang/dataset/split_ss_dota/trainval/dota1-rect.pkl'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PseudoBoxSync',
         rel_tol=1e-3),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'pseudo_boxes', 'pseudo_valid')
    )
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(pipeline=train_pipeline))

# optimizer (与基线一致: AdamW 5e-5)
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

# lr schedule (与基线一致: LinearLR 500iter + MultiStepLR [8,11])
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# val: 与基线一致 (在 trainval 上计算 mAP + 新角度指标)
val_evaluator = dict(type='DOTAMetric', metric='mAP')

test_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/mnt/data/liurunxiang/dataset/split_ss_dota/',
        ann_file='trainval/labelTxt/',
        data_prefix=dict(img_path='trainval/images/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))

# 注意: 显式关闭 format_only / merge_patches, 否则 base 的旧 test_evaluator
# (format_only=True, merge_patches=True) 会被 Config 递归合并保留, 评估只出 submission
test_evaluator = dict(type='DOTAMetric', metric='mAP',
                      format_only=False, merge_patches=False)
