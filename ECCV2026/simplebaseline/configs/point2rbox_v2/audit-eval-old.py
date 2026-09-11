# ---------------------------------------------------------------
# audit-eval-old.py
# 旧权重 (rect/2-0.413/epoch_10.pth) 的角度诊断评估配置
# 复用 base 的 val_pipeline (与训练时 val mAP 完全同口径: test_mode=True + GT加载)
# 仅 batch_size 调小 (2) 以控制评估显存; evaluator 用修复后的 DOTAMetric
# 输出 mAP + mAngle_raw + mAngle_longedge + 轴对齐比例 + 长宽比子集
# ---------------------------------------------------------------
_base_ = './audit-fixed-baseline-s0.py'

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/mnt/data/liurunxiang/dataset/split_ss_dota/',
        ann_file='trainval/labelTxt/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        test_mode=True,
        pipeline=_base_.val_pipeline))

test_dataloader = val_dataloader
# 注意: 必须显式关闭 format_only / merge_patches!
# base 的旧 test_evaluator (format_only=True, merge_patches=True) 会经
# mmengine Config 递归合并保留下来, 导致评估只出 submission 不算 mAP
val_evaluator = dict(type='DOTAMetric', metric='mAP',
                     format_only=False, merge_patches=False)
test_evaluator = dict(type='DOTAMetric', metric='mAP',
                      format_only=False, merge_patches=False)
