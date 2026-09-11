# audit-fixed-awarmup2.py: AngleLoss warmup 2 epoch（动态决策备用）
# 单变量: 添加 LossWarmupHook(warmup_epochs=2)，其余与 audit-fixed-baseline-s0 一致
_base_ = ['./audit-fixed-baseline-s0.py']
custom_imports = dict(
    imports=[
        'mmrotate.datasets.transforms.loading_pseudo',
        'mmrotate.datasets.transforms.pseudo_box_sync',
        'mmrotate.engine.hooks.loss_warmup_hook'
    ],
    allow_failed_imports=False)
custom_hooks = [
    dict(type='mmdet.SetEpochInfoHook'),
    dict(type='LossWarmupHook', warmup_epochs=2)
]
