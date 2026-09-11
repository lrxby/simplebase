# submit-audit-ow3s1.py: 修复后代码上，用 ow3_s1/epoch_12.pth (best 0.4893) 生成 DOTA 官网 Task1 提交包
# 继承修复后训练配置，仅覆盖 test 段为官网提交模式（format_only + merge_patches）
_base_ = ['./audit-fixed-ow3.py']

test_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root='/mnt/data/liurunxiang/dataset/split_ss_dota/',
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=_base_.test_pipeline))

test_evaluator = dict(
    _delete_=True,
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/night/submit_ow3s1_ep12/Task1')
