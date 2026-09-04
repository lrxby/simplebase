# ---------------------------------------------------------------
# 1. 引入自定义 Pipeline
# ---------------------------------------------------------------
custom_imports = dict(
    imports=['mmrotate.datasets.transforms.loading_pseudo'], 
    allow_failed_imports=False
)

_base_ = [
    '../_base_/datasets/dronevehicle.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

randomness = dict(seed=0, deterministic=False)

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
        num_classes=5,  # DroneVehicle 5类
        in_channels=128,
        feat_channels=128,
        strides=[8],
        
        # ------------------ Loss 开关控制 ------------------
        use_bbox_loss=True,      # 基础回归 GDLoss
        use_size_loss=True,     # SizeLoss
        use_angle_loss=True,    # AngleLoss
        use_voronoi_loss=True,  # VoronoiLoss
        use_ourwater_loss=False,  # [新增] OurWaterLoss (Row 5/6)
        # --------------------------------------------------
        
        # 几何参数
        voronoi_type='gaussian-orientation',
        voronoi_thres=dict(
            default=[0.98, 0.01],
            override=(([0, 3], [0.99, 0.01]), 
                      ([1, 3, 4], [0.95, 0.005]))),
        square_cls=[], 
        
        post_process={},
        
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
            
        loss_voronoi=dict(
            type='VoronoiWatershedLoss', 
            loss_weight=5.0),

        # [新增] 离线伪标签监督
        loss_ourwater=dict(
            type='OurWaterLoss',
            loss_weight=1.0
        ),

        loss_bbox=dict(
            type='GDLoss', 
            loss_type='gwd', 
            loss_weight=5.0),
            
        loss_size=dict(
            type='SizeLoss', 
            loss_weight=1.0, 
            beta=1.0,
            target_classes=None),
            
        loss_angle=dict(
            type='AngleLoss',
            loss_weight=1.0, 
            k_radius=2.0,
            score_alpha=1.0,
            target_classes=None
        )
    ),
    
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# ---------------------------------------------------------------
# 2. 修改 Pipeline (加载 DroneVehicle 离线数据)
# ---------------------------------------------------------------
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    
    # === [关键插入] 加载伪标签 ===
    # 请确认此处路径与生成脚本输出一致
    dict(type='LoadPseudoAnnotations', 
         pkl_path='/mnt/data/xiekaikai/DroneVehicle/train/train.pkl'),
    # ============================

    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='RandomRotate', prob=0.5, angle_range=180),  # Drone通常允许全角度旋转
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    
    # DroneVehicle 通常使用固定尺寸 Resize (800x800)
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
        
    # [关键] 显式指定 meta_keys 以包含 pseudo_boxes
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                   'scale_factor', 'flip', 'flip_direction', 'pseudo_boxes')
    )
]

train_dataloader = dict(batch_size=2,
                        num_workers=2,
                        dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]