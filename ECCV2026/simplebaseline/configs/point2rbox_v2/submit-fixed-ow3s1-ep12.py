# submit-fixed-ow3s1-ep12.py: 修复后代码 + ow3_s1/ep12 (best 0.4893) 生成 DOTA 官网 Task1 提交包
# 基于修复后 ow3 config（与训练完全一致），仅覆盖 test 段为官网提交模式
_base_ = ['./audit-fixed-ow3.py']

# 官网提交：test 集无标注，用切好的 test 切片（10833张），merge_patches 自动合并回大图
test_dataloader = dict(
    _delete_=True,
    batch_size=4,
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

# 官网提交模式：只生成 Task1 提交文件（merge_patches 合并回大图），不计算 mAP
test_evaluator = dict(
    _delete_=True,
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/night/submit_ow3s1_ep12/Task1')
