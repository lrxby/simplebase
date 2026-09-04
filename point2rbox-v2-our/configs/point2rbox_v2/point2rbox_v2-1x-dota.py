_base_ = [
    '../_base_/datasets/dota.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
angle_version = 'le90'

randomness = dict(seed=0, deterministic=False)

# ====================================================================
# [控制台] 全局数据量控制 (Unified Dataset Size Control)
# ====================================================================
# 修改这里即可统一控制训练、验证、测试集的图片数量
# 整数 (e.g., 100) : 仅读取前 100 张图片 (用于快速调试代码/Loss/显存)
# None             : 读取全部数据 (用于正式训练)
DEBUG_MAX_SAMPLES = 2000 
# ====================================================================


# ====================================================================
# 模型设置 (Model Settings)
# ====================================================================
model = dict(
    type='Point2RBoxV2',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    
    # 主干网络 (ResNet50)
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
    
    # 颈部网络 (FPN)
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    
    # ================================================================
    # 核心检测头 (BBox Head) - 实验核心配置
    # ================================================================
    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        strides=[8], # 特征图步长
        
        # ------------------------------------------------------------
        # [控制台] 核心模式开关 (Experiment Switches)
        # ------------------------------------------------------------
        
        # 1. 分类模式 (Classification Mode)
        # 'cls'   : [Baseline] 仅使用原版点分类 (Point-based)
        # 'roicls': [Ours] 开启级联精修模式 (Conv初筛 -> ROI精修)
        cls_mode='cls', 
        
        # [调试开关] 是否使用 GT 点构造伪造 RoI
        # True: 使用 GT 点构造微小框，用于验证 RoIAlign 代码正确性 (mAP 应约为 0.2)
        # False: 使用网络预测框，用于正常训练 (Target Mode)
        debug_roi_input=False, # [重要] 正常训练时请设为 False
        
        # 2. 几何监督模式 (Geometric Supervision Mode)
        # 'voronoi' : [Baseline] 原版 Voronoi Loss (基于参数分布匹配)
        # 'ourwater': [Ours] 我们的注水池 Loss (基于点云距离匹配)
        geo_mode='voronoi',
        
        # ------------------------------------------------------------
        # [ROI 参数] 仅在 cls_mode='roicls' 时生效
        # ------------------------------------------------------------
        roi_size=(7, 7), # ROI Align 输出的特征图尺寸

        # ------------------------------------------------------------
        # [几何参数] 用于生成伪 Mask (注水盆地)
        # ------------------------------------------------------------
        voronoi_thres=dict(
            default=[0.994, 0.005], # 前景/背景 阈值
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],  # 这些类别强制角度为 0 (如储油罐)
        post_process={11: 1.2}, # 后处理缩放系数
        
        # 编解码器配置
        bbox_coder=dict(type='DistanceAnglePointCoder'),
        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),
        
        # ------------------------------------------------------------
        # [损失函数配置] (Loss Configs)
        # 注意：不要注释掉任何一个，Head 代码会根据 Switch 按需调用
        # ------------------------------------------------------------
        
        # A. 原版点分类 Loss (始终激活，作为 RPN 初筛器)
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        # B. ROI 分类 Loss (当 cls_mode='roicls' 时激活)
        loss_roicls=dict(
            type='mmdet.FocalLoss', # 或 CrossEntropyLoss
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        # C. 原版 Voronoi Loss (当 geo_mode='voronoi' 时激活)
        loss_voronoi=dict(
             type='VoronoiWatershedLoss', 
             loss_weight=5.0,
             down_sample=2,
             topk=0.95,
             alpha=0.1),

        # D. 我们的注水池 Loss (当 geo_mode='ourwater' 时激活)
        loss_ourwater=dict(
            type='OurWaterLoss', 
            loss_weight=1.0,  # [重要] 权重设为 1.0 (因为内部已 *100 归一化)
            down_sample=2,
            voronoi_type='gaussian-orientation',
            default_sigma=4096, # 必须保留，用于 Mask 生成逻辑
            num_gt_samples=256, # [改进] 提高采样点数以稳定梯度
            grid_size=15,       # [改进] 提高预测网格密度
            topk=0.90,          # [改进] 截断比例 (保留 90% 最优点)
            debug=False),       
        
        # E. 辅助 Loss (尺寸和角度约束)
        loss_size=dict(
            type='SizeLoss', 
            loss_weight=5.0, # 适当提高权重以稳定形状
            beta=1.0,
            target_classes=None),
            
        loss_angle=dict(
            type='AngleLoss',
            loss_weight=5.0, # 适当提高权重以稳定方向
            k_radius=2.0,
            score_alpha=1.0,
            target_classes=None
        )
    ),
    
    # 训练和测试配置
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# ====================================================================
# 数据流水线 (Data Pipeline)
# ====================================================================
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # 将强监督框转换为弱监督点 (Weak Supervision)
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

# --------------------------------------------------------------------
# [重要] 统一应用 max_samples 到所有 Dataloader
# 利用 MMEngine 的配置合并机制，这里只会覆盖 max_samples 参数，
# 其他参数（如 batch_size, sampler 等）会保留 base 中的默认值。
# --------------------------------------------------------------------

# 1. 训练集
train_dataloader = dict(
    batch_size=2, 
    dataset=dict(
        pipeline=train_pipeline,
        max_samples=DEBUG_MAX_SAMPLES  # <--- 引用顶部变量
    ))

# 2. 验证集 (Val)
val_dataloader = dict(
    dataset=dict(
        max_samples=DEBUG_MAX_SAMPLES  # <--- 引用顶部变量
    )
)

# 3. 测试集 (Test)
test_dataloader = dict(
    dataset=dict(
        max_samples=DEBUG_MAX_SAMPLES  # <--- 引用顶部变量
    )
)

# ====================================================================
# 优化器与训练策略 (Optimizer & Schedule)
# ====================================================================
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005, # 学习率
        betas=(0.9, 0.999),
        weight_decay=0.05))

# Hooks
custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)