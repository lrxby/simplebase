# DOTA 官网提交配置：用 ourwater3 最优权重生成 Task1 提交包
# 基于 auto-single-ourwater3.py（与训练完全一致），仅覆盖 test 段为官网提交模式
_base_ = ['./auto-single-ourwater3.py']

# 官网提交：test 集无标注，用切好的 test 切片，merge_patches 自动合并回大图
# 只在 test_dataloader 层用 _delete_=True 整体替换（mmengine 深合并会保留 base 的 ann_file）
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

# 官网提交模式：只生成 Task1 提交文件，不计算 mAP
test_evaluator = dict(
    _delete_=True,
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/sb/dt1/submit_ourwater3_s2/Task1')
