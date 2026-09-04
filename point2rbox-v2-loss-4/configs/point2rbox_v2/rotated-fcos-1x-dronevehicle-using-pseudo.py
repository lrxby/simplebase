_base_ = [
    '../_base_/datasets/dronevehicle.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

# 显式定义 metainfo 确保万无一失
metainfo = dict(classes=('car', 'bus', 'truck', 'van', 'freight_car'))

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
        num_classes=5, 
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

# ---------------------------------------------------------------
# 核心修改：统一训练和验证的 Resize 逻辑
# ---------------------------------------------------------------
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='rbox'), 
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True), # 必须是 FixShape
    dict(type='mmdet.RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True), # 验证集必须也用这个！
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='DroneVehicleDataset',  
        metainfo=metainfo,
        data_root=_base_.data_root,
        ann_file='/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-3/work_dirs/dv-pr2/point2rbox_v2_pseudo_labels.bbox.json', 
        data_prefix=dict(img_path='train/images'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        metainfo=metainfo,
        pipeline=val_pipeline  # 强制覆盖 base 里的 Resize 逻辑
    )
)

test_dataloader = val_dataloader

# 在评估器中显式绑定类别，防止映射到 DOTA 的 15 类上
val_evaluator = dict(
    type='DOTAMetric', 
    metric='mAP', 
    iou_thrs=[0.5], 
    dataset_meta=metainfo
)
test_evaluator = val_evaluator

# 优化器
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.005))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)