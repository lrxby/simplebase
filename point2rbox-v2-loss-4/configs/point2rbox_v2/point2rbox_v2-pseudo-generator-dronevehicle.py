_base_ = 'point2rbox_v2-1x-dronevehicle.py'

# 开启生成器模式：使用真实点作为中心生成伪框
model = dict(bbox_head=dict(pseudo_generator=True))

# 定义生成流：必须与第一阶段训练逻辑完全一致，且包含完整的 meta_keys
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    # 统一使用 FixShapeResize 确保和训练集尺度逻辑闭环
    dict(type='mmdet.FixShapeResize', width=800, height=800, keep_ratio=True),
    # 必须显式包含 scale_factor，否则伪标签坐标无法映射回原图尺寸
    dict(type='mmdet.PackDetInputs', 
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 劫持 dataloader 为训练集
test_dataloader = _base_.train_dataloader
test_dataloader['_delete_'] = True
test_dataloader['dataset']['pipeline'] = test_pipeline

# 配置导出：生成的 json 包含原图尺度的伪旋转框
test_evaluator = dict(_delete_=True,
                    type='DOTAMetric',
                    metric='mAP',
                    format_only=True,
                    outfile_prefix='/mnt/data/liurunxiang/workplace/point2rbox-v2-loss-3/work_dirs/dv-pr2/point2rbox_v2_pseudo_labels')