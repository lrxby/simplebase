_base_ = [
    '../_base_/datasets/codrone.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# 模型设置 (Rotated FCOS)
model = dict(
    type='mmdet.FCOS',
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
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=512,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=12,  # 【关键】类别数改为 CODrone 的 12 类
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        use_hbbox_loss=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_angle=None,
        loss_centerness=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# 训练数据处理管道 (使用 rbox)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='rbox'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),  # 维持 CODrone 尺寸
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

# 覆盖 train_dataloader 以读取生成的 json 伪标签文件
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='CODRONEDataset',  # 【关键修改】：换回你自己的数据集
        data_root=_base_.data_root,
        # 【注意】确保加上 .bbox.json 后缀，因为 evaluator 会自动附加这个后缀
        ann_file='/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-3/work_dirs/cd/point2rbox_v2_pseudo_labels.bbox.json',  
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(batch_size=4)

val_evaluator = dict(
    type='DOTAMetric',
    metric='mAP',
    iou_thrs=[0.5, 0.75])

# 优化器
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.005))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)