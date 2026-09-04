# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from torch import Tensor

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmrotate.registry import MODELS


@MODELS.register_module()
class Point2RBoxV2(SingleStageDetector):
    """
    Point2RBox-v2 (精简单流版本)
    
    该版本主要移除了以下为了"极致性能"而引入的复杂模块，回归到"纯几何约束"的本质：
    1. 移除一致性学习 (Consistency Learning): 不再生成旋转/翻转的双流图像，大幅提速。
    2. 移除 TED (Token-based Edge Detector): 不再使用额外的边缘检测网络，节省显存。
    3. 移除 Copy-Paste 增强: 简化数据流。
    
    它保留了核心机制：通过手动注入 Batch ID 和原始图像，支持 Head 计算 Voronoi 和几何损失。
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # [Deleted] 移除了 self.ted (边缘检测模型) 的初始化
        # [Deleted] 移除了控制一致性损失权重的参数

    def set_epoch(self, epoch):
        """
        同步 Epoch 信息到 bbox_head。
        虽然精简版移除了大部分 Curriculum Learning (课程学习)，
        但保留此接口方便后续扩展或控制某些 Loss 在特定 Epoch 开启。
        """
        self.epoch = epoch
        self.bbox_head.epoch = epoch

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """
        计算损失函数。
        
        模式: 单流极速模式 (Single Stream)
        """
        # 1. 解包 Ground Truths (从 DataSample 中提取 bboxes 和 labels)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)

        # 2. 分配实例 ID (bids) - [关键步骤]
        # Point2RBoxV2Head 的 Voronoi Loss 强依赖 `bids` 来对同一张图内的实例进行分组。
        # bids 格式: [batch_idx, syn_flag, view_idx, instance_idx]
        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances.bboxes)
            # 初始化 bids 张量 (N, 4)
            bids = gt_instances.labels.new_zeros(blen, 4)
            
            # Col 0: Batch Index (这是最重要的，用于区分样本属于哪张图)
            bids[:, 0] = i          
            
            # Col 1: Synthesis Flag (0=真实物体, 1=Copy-Paste物体). 这里全是真实物体，默认为0.
            # bids[:, 1] = 0        
            
            # Col 2: View Index (0=原图, 1=增强图). 这里只有单流，默认为0.
            # bids[:, 2] = 0        
            
            # Col 3: Unique Instance ID (全局唯一ID，防止跨图混淆)
            bids[:, 3] = torch.arange(0, blen, 1) + offset 
            
            # 将生成的 bids 挂载到 gt_instances 对象上，Head 会读取它
            gt_instances.bids = bids
            offset += blen

        # [Deleted] 删除了 rotate_crop / flip 等几何变换逻辑
        # [Deleted] 删除了 Copy-Paste 增强逻辑
        # [Deleted] 删除了 TED 边缘提取逻辑

        # 3. 注入原始图像到 Head - [关键步骤]
        # VoronoiWatershedLoss 需要访问原始像素数据来计算梯度/能量图 (Energy Map)。
        # 标准检测器通常不传 images 给 head，所以这里需要手动赋值。
        self.bbox_head.images = batch_inputs

        # 4. 提取特征 (Backbone + Neck)
        # 仅对原始 Batch 进行一次前向传播，效率最高。
        x = self.extract_feat(batch_inputs)

        # 5. 同步修改后的 GT (主要是 bids) 回 batch_data_samples
        # 因为 Head 的 loss 方法接收的是 batch_data_samples 列表，
        # 我们必须确保第2步生成的 bids 已经被更新进去。
        for data_sample, gt_instances in zip(batch_data_samples, batch_gt_instances):
            data_sample.gt_instances = gt_instances

        # 6. 前向传播计算 Loss
        losses = self.bbox_head.loss(x, batch_data_samples)

        return losses