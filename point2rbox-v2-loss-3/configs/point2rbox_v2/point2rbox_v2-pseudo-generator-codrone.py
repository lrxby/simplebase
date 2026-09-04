_base_ = 'point2rbox_v2-1x-codrone.py'

# 开启 pseudo_generator 模式，使模型使用真实点生成伪旋转框
model = dict(bbox_head=dict(pseudo_generator=True))

# 定义测试管道，保持尺度为 CODrone 的 (1024, 1024)
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.PackDetInputs')
]

# 将推断的数据加载器替换为 train_dataloader (即 CODrone 的 trainval 集)
test_dataloader = _base_.train_dataloader
test_dataloader['_delete_'] = True
test_dataloader['dataset']['pipeline'] = test_pipeline

# 配置 evaluator，将生成的伪框输出到 CODrone 的 trainval 目录下
test_evaluator = dict(_delete_=True,
                    type='DOTAMetric',
                    metric='mAP',
                    format_only=True,
                    # 生成的 json 将保存在此处，根据你的 data_root 设置
                    outfile_prefix='/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-3/work_dirs/cd/point2rbox_v2_pseudo_labels')