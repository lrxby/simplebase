# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Dict, List, Optional, Tuple

# import copy
# import torch
# import torch.nn as nn
# from mmdet.models.dense_heads import AnchorFreeHead
# from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
#                                 select_single_mlvl, unpack_gt_instances)
# from mmdet.structures import SampleList
# from mmdet.structures.bbox import cat_boxes
# from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
#                          OptInstanceList, reduce_mean)
# from mmengine import ConfigDict
# from mmengine.structures import InstanceData
# from torch import Tensor

# from mmrotate.registry import MODELS, TASK_UTILS
# from mmrotate.structures import RotatedBoxes
# # [关键引用] 引入我们自己实现的可导 ROI Align 模块
# from mmrotate.models.utils import DifferentiableRoIAlignRotated

# INF = 1e8

# @MODELS.register_module()
# class Point2RBoxV2Head(AnchorFreeHead):
#     """
#     Point2RBoxV2Head (Hybrid Version): 
#     支持通过配置无缝切换 'cls' (Baseline) 和 'roicls' (ROI-Only) 模式。
#     包含完整的 Debug 逻辑和负样本采样。
#     """

#     def __init__(self,
#                  num_classes: int,
#                  in_channels: int,
#                  strides: list = [8],
#                  regress_ranges: list = [(-1, 1e8)],
#                  center_sampling: bool = True,
#                  center_sample_radius: float = 0.75,
#                  angle_version: str = 'le90',
#                  pseudo_generator: bool = False,
#                  voronoi_type: str = 'gaussian-orientation',
#                  voronoi_thres: dict = dict(default=[0.994, 0.005]), 
#                  square_cls: list = [],
#                  post_process: dict = {},
#                  bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
#                  angle_coder: ConfigType = dict(
#                     type='PSCCoder',
#                     angle_version='le90',
#                     dual_freq=False,
#                     num_step=3,
#                     thr_mod=0),
                 
#                  # --- [Switches] ---
#                  cls_mode: str = 'cls',      # 'cls' (默认) 或 'roicls'
#                  geo_mode: str = 'voronoi',
#                  debug_roi_input: bool = False,
                 
#                  # --- [Loss Configs] ---
#                  loss_cls: Optional[ConfigType] = None,
#                  loss_roicls: Optional[ConfigType] = None,
#                  roi_size: tuple = (7, 7),
                 
#                  loss_voronoi: Optional[ConfigType] = None,
#                  loss_ourwater: Optional[ConfigType] = None,
                 
#                  loss_size: ConfigType = dict(
#                      type='SizeLoss', loss_weight=1.0),
#                  loss_angle: ConfigType = dict(
#                      type='AngleLoss', loss_weight=1.0),
                 
#                  norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
#                  init_cfg=dict(
#                      type='Normal',
#                      layer='Conv2d',
#                      std=0.01,
#                      override=[
#                          dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01), 
#                          dict(type='Normal', name='conv_gate', std=0.01, bias_prob=0.01)]),
#                  **kwargs):
        
#         self.angle_coder = TASK_UTILS.build(angle_coder)
        
#         dummy_loss = dict(type='mmdet.L1Loss', loss_weight=0.0)

#         super().__init__(
#             num_classes,
#             in_channels,
#             strides=strides,
#             bbox_coder=bbox_coder,
#             loss_cls=loss_cls if loss_cls is not None else dummy_loss, 
#             loss_bbox=dummy_loss, 
#             norm_cfg=norm_cfg,
#             init_cfg=init_cfg,
#             **kwargs)
            
#         self.regress_ranges = regress_ranges
#         self.center_sampling = center_sampling
#         self.center_sample_radius = center_sample_radius
#         self.angle_version = angle_version
#         self.pseudo_generator = pseudo_generator
#         self.voronoi_type = voronoi_type
#         self.voronoi_thres = voronoi_thres
#         self.square_cls = square_cls
#         self.post_process = post_process
        
#         self.cls_mode = cls_mode
#         self.geo_mode = geo_mode
#         self.debug_roi_input = debug_roi_input
#         self.real_loss_cls_cfg = loss_cls 
        
#         if self.cls_mode == 'roicls':
#             if loss_roicls is None:
#                  raise ValueError("In 'roicls' mode, 'loss_roicls' config must be provided!")
            
#             self.roi_extractor = DifferentiableRoIAlignRotated(
#                 output_size=roi_size,
#                 spatial_scale=1.0 / strides[0], 
#                 sampling_ratio=2 
#             )
#             flat_dim = self.feat_channels * roi_size[0] * roi_size[1]
#             self.roi_cls_layer = nn.Linear(flat_dim, num_classes)
#             self.loss_roicls = MODELS.build(loss_roicls)
        
#         # 2. 构建几何 Loss
#         if self.geo_mode == 'voronoi' and loss_voronoi:
#             self.loss_voronoi = MODELS.build(loss_voronoi)
#         elif self.geo_mode == 'ourwater' and loss_ourwater:
#             self.loss_ourwater = MODELS.build(loss_ourwater)
            
#         self.loss_size = MODELS.build(loss_size)
#         self.loss_angle = MODELS.build(loss_angle)
            
#     def _init_layers(self):
#         super()._init_layers()
#         self.conv_angle = nn.Conv2d(
#             self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
#         self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
#     def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
#         cls_feat = x[0]
#         reg_feat = x[0]

#         for cls_layer in self.cls_convs:
#             cls_feat = cls_layer(cls_feat)
        
#         # 无论什么模式，conv_cls 都要跑一遍（为了结构完整性），但 roicls 模式下不训练它
#         cls_score = self.conv_cls(cls_feat)

#         for reg_layer in self.reg_convs:
#             reg_feat = reg_layer(reg_feat)
#         bbox_pred = self.conv_reg(reg_feat)
#         angle_pred = self.conv_angle(reg_feat)

#         sig_x = bbox_pred[:, 0].exp()
#         sig_y = bbox_pred[:, 1].exp()
#         dx = bbox_pred[:, 2].sigmoid() * 2 - 1 
#         dy = bbox_pred[:, 3].sigmoid() * 2 - 1 
#         bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

#         return (cls_score,), (bbox_pred,), (angle_pred,)
    
#     def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
#         outs = self(x)
#         outputs = unpack_gt_instances(batch_data_samples)
#         (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs
#         loss_inputs = outs + (x, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
#         return self.loss_by_feat(*loss_inputs)

#     def loss_by_feat(
#         self,
#         cls_scores: List[Tensor],
#         bbox_preds: List[Tensor],
#         angle_preds: List[Tensor],
#         rois_features: Tuple[Tensor],
#         batch_gt_instances: InstanceList,
#         batch_img_metas: List[dict],
#         batch_gt_instances_ignore: OptInstanceList = None
#     ) -> Dict[str, Tensor]:
        
#         # --- 1. 准备 Ground Truth ---
#         featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
#         all_level_points = self.prior_generator.grid_priors(
#             featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        
#         labels, bbox_targets, bid_targets = self.get_targets(all_level_points, batch_gt_instances)
#         num_imgs = bbox_preds[0].size(0)
        
#         # 展平所有数据
#         flatten_bbox_preds = torch.cat([bp.permute(0, 2, 3, 1).reshape(-1, 4) for bp in bbox_preds])
#         flatten_angle_preds = torch.cat([ap.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size) for ap in angle_preds])
#         flatten_cls_scores = torch.cat([cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])

#         flatten_labels = torch.cat(labels)
#         flatten_bbox_targets = torch.cat(bbox_targets)
#         flatten_bid_targets = torch.cat(bid_targets)
#         flatten_points = torch.cat([p.repeat(num_imgs, 1) for p in all_level_points])

#         loss_dict = dict()
#         zero_tensor = flatten_bbox_preds.new_tensor(0.0)
        
#         bg_class_ind = self.num_classes
#         pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
#         num_pos = len(pos_inds) # 真实正样本数

#         if self.cls_mode == 'cls':
#             avg_factor = max(reduce_mean(torch.tensor(num_pos, dtype=torch.float, device=bbox_preds[0].device)), 1.0)
#             loss_dict['loss_cls'] = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=avg_factor)
            
#         elif self.cls_mode == 'roicls':
#             # 1. 负样本采样 (Negative Sampling)
#             # 必须采样背景，否则网络会把所有东西都当成前景
#             neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
#             # 比例 1:3
#             num_neg = min(num_pos * 3, len(neg_inds)) 
#             if num_neg > 0:
#                 perm = torch.randperm(len(neg_inds), device=neg_inds.device)[:num_neg]
#                 sampled_neg_inds = neg_inds[perm]
#             else:
#                 # 极端情况：如果没有正样本，随机采一些负样本防止 crash
#                 if len(neg_inds) > 100:
#                     sampled_neg_inds = neg_inds[:100] 
#                 else:
#                     sampled_neg_inds = neg_inds
            
#             # 训练用的总索引
#             train_inds = torch.cat([pos_inds, sampled_neg_inds])
            
#             # 2. 准备训练用的 Box
#             # 正样本：解码预测的 offset 和 angle
#             # 负样本：同样解码，作为抠图位置
#             train_labels = flatten_labels[train_inds]
#             train_bbox_preds = flatten_bbox_preds[train_inds]
#             train_angle_preds = flatten_angle_preds[train_inds]
#             train_points = flatten_points[train_inds]
#             train_bid_targets = flatten_bid_targets[train_inds]
            
#             # 解码角度
#             decoded_angle = self.angle_coder.decode(train_angle_preds, keepdim=True)
#             # 正样本特殊类别处理
#             if len(self.square_cls) > 0:
#                 # 只有正样本需要根据 label 处理，负样本 label 是 bg_class_ind，不会误触
#                 for c in self.square_cls:
#                     decoded_angle[train_labels == c] = 0

#             train_rbox_preds = torch.cat((train_points + train_bbox_preds[:, 2:], 
#                                           train_bbox_preds[:, :2] * 2,
#                                           decoded_angle), -1)
            
#             # 3. 运行 ROI Cls
#             current_feat = rois_features[0] # P2
            
#             # --- [DEBUG START] ---
#             if self.debug_roi_input:
#                 if not hasattr(self, '_debug_logged'):
#                     print(f"\n[DEBUG MODE] Using Fake GT ROIs for Training.")
#                     self._debug_logged = True
                
#                 # 构造 Debug 框：使用 GT 中心点 (train_points)
#                 # 注意：这里我们不仅用 GT 点，还把所有框都变成微小的，包括负样本
#                 # 这样负样本就在其 Anchor 点的位置采一个微小特征
#                 d_cx = train_points[:, 0]
#                 d_cy = train_points[:, 1]
#                 d_w = torch.ones_like(d_cx) * 1.0
#                 d_h = torch.ones_like(d_cy) * 1.0
#                 d_t = torch.zeros_like(d_cx)
                
#                 debug_rboxes = torch.stack([d_cx, d_cy, d_w, d_h, d_t], dim=-1)
                
#                 batch_inds = train_bid_targets[:, 0].float().unsqueeze(1)
#                 rois = torch.cat([batch_inds, debug_rboxes], dim=1)
#             else:
#                 # 正常模式
#                 batch_inds = train_bid_targets[:, 0].float().unsqueeze(1)
#                 rois = torch.cat([batch_inds, train_rbox_preds], dim=1)
#             # --- [DEBUG END] ---
            
#             roi_feats = self.roi_extractor(current_feat, rois)
#             roi_logits = self.roi_cls_layer(roi_feats.view(roi_feats.size(0), -1))
            
#             # 计算 Loss
#             avg_factor = max(num_pos, 1.0)
#             loss_dict['loss_roicls'] = self.loss_roicls(
#                 roi_logits, train_labels, avg_factor=avg_factor)

#         if len(pos_inds) > 0:
#             # 重新提取正样本数据 (从 flatten 数组中)
#             pos_bbox_preds = flatten_bbox_preds[pos_inds]
#             pos_angle_preds = flatten_angle_preds[pos_inds]
#             pos_bbox_targets = flatten_bbox_targets[pos_inds]
#             pos_bid_targets = flatten_bid_targets[pos_inds]
#             pos_points = flatten_points[pos_inds]
#             pos_labels = flatten_labels[pos_inds]
            
#             # 解码用于几何 Loss 的正样本预测框
#             pos_decoded_angle = self.angle_coder.decode(pos_angle_preds, keepdim=True)
#             for c in self.square_cls:
#                 pos_decoded_angle[pos_labels == c] = 0
            
#             pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
#             pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
#                                         pos_bbox_preds[:, :2] * 2,
#                                         pos_decoded_angle), -1)
            
#             # 准备 Score 用于 Loss Size 的加权
#             if self.cls_mode == 'cls':
#                 pos_scores = flatten_cls_scores[pos_inds].sigmoid()
#                 pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]
#             else:                
#                 pos_roi_logits_subset = roi_logits[:num_pos]
#                 pos_scores = pos_roi_logits_subset.sigmoid()
#                 pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]

#             # --- Loss Size & Angle ---
#             loss_dict['loss_size'] = self.loss_size(
#                 pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0], batch_img_metas)
#             loss_dict['loss_angle'] = self.loss_angle(
#                 pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0])
            
#             # --- Geometry Loss (Voronoi / OurWater) ---
#             # 构造 Gaussian
#             cos_r, sin_r = torch.cos(pos_decoded_angle), torch.sin(pos_decoded_angle)
#             R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
#             pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))
            
#             # Grouping
#             bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
#             bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
#             ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
#             ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
#             ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(0, idx, pos_labels, 'amin', include_self=False)
            
#             ins_gaus_preds = pos_gaus_preds.new_zeros(*bid.shape, 4).index_reduce_(0, idx, pos_gaus_preds.view(-1, 4), 'mean', include_self=False).view(-1, 2, 2)
#             ins_rbox_targets = pos_rbox_targets.new_zeros(*bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(0, idx, pos_rbox_targets, 'mean', include_self=False)
#             ori_mu_all = ins_rbox_targets[:, 0:2]

#             # Loss Calculation
#             if self.geo_mode == 'voronoi':
#                 loss_geo = zero_tensor.clone()
#                 for batch_id in range(len(batch_gt_instances)):
#                     group_mask = (ins_batch == batch_id) & (ins_bids != 0)
#                     mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
#                     if len(mu) >= 1:
#                         # ... Thresholds ...
#                         pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
#                         neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
#                         if 'override' in self.voronoi_thres.keys():
#                             for item in self.voronoi_thres['override']:
#                                 for cls in item[0]:
#                                     pos_thres[cls] = item[1][0]
#                                     neg_thres[cls] = item[1][1]
                        
#                         loss_geo += self.loss_voronoi(
#                             (mu, sigma.bmm(sigma)), label, self.images[batch_id],
#                             pos_thres, neg_thres, voronoi=self.voronoi_type)
#                 loss_dict['loss_bbox_vor'] = loss_geo / len(batch_gt_instances)
                
#             elif self.geo_mode == 'ourwater':
#                 loss_geo = zero_tensor.clone()
#                 for batch_id in range(len(batch_gt_instances)):
#                     group_mask = (ins_batch == batch_id) & (ins_bids != 0)
#                     mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
#                     if len(mu) >= 1:
#                         # ... Thresholds ...
#                         pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
#                         neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
#                         if 'override' in self.voronoi_thres.keys():
#                             for item in self.voronoi_thres['override']:
#                                 for cls in item[0]:
#                                     pos_thres[cls] = item[1][0]
#                                     neg_thres[cls] = item[1][1]
                        
#                         loss_geo += self.loss_ourwater(
#                             (mu, sigma.bmm(sigma)), label, self.images[batch_id],
#                             pos_thres, neg_thres, voronoi=self.voronoi_type)
#                 loss_dict['loss_ourwater'] = loss_geo / len(batch_gt_instances)

#         return loss_dict
#     # ====================================================================
#     # 核心推理逻辑: 支持根据 cls_mode 自动切换
#     # ====================================================================
#     def predict(self,
#                 x: Tuple[Tensor],
#                 batch_data_samples: SampleList,
#                 rescale: bool = False) -> InstanceList:
#         """Perform forward propagation of the detection head and predict detection results."""
#         outs = self(x)
        
#         # [Correct] Do NOT unpack here. Pass the original list of DetDataSample.
#         predictions = self.predict_by_feat(
#             *outs, 
#             batch_data_samples=batch_data_samples, 
#             rescale=rescale,
#             x=x) # Pass features for ROI head
            
#         return predictions
    
#     def predict_by_feat(
#             self,
#             cls_scores: List[Tensor],
#             bbox_preds: List[Tensor],
#             angle_preds: List[Tensor],
#             batch_data_samples: Optional[List[dict]] = None,
#             cfg: Optional[ConfigDict] = None,
#             rescale: bool = False,
#             with_nms: bool = True,
#             x: Tuple[Tensor] = None,
#             **kwargs) -> InstanceList:
#         """Transform a batch of output features extracted from the head into bbox results."""
#         assert len(cls_scores) == len(bbox_preds)
#         num_levels = len(cls_scores)

#         featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
#         mlvl_priors = self.prior_generator.grid_priors(
#             featmap_sizes,
#             dtype=cls_scores[0].dtype,
#             device=cls_scores[0].device)

#         result_list = []
        
#         # Iterate over the batch
#         for img_id in range(len(batch_data_samples)):
#             # [Standard MMEngine Way] Access metainfo directly
#             img_meta = batch_data_samples[img_id].metainfo
            
#             # Reconstruct the tuple (gt, ignore, meta) to match your legacy functions' signature
#             # if they expect 'data_sample' to be a tuple.
#             gt_instances = batch_data_samples[img_id].gt_instances
#             gt_ignore = batch_data_samples[img_id].ignored_instances
#             data_sample = (gt_instances, gt_ignore, img_meta)

#             # [Branch A] ROI-Only Inference
#             if self.cls_mode == 'roicls':
#                 # Extract feature for current image: (1, C, H, W)
#                 current_feat = x[0][img_id:img_id+1]
                
#                 bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
#                 angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
                
#                 results = self._predict_single_roi(
#                     bbox_pred_list, 
#                     angle_pred_list, 
#                     current_feat, 
#                     mlvl_priors, 
#                     data_sample, 
#                     cfg, 
#                     rescale, 
#                     with_nms)
            
#             # [Branch B] Conv-Only Inference (Original)
#             else:
#                 cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
#                 bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
#                 angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
                
#                 if self.training or self.pseudo_generator:
#                     predict_func = self._predict_by_feat_single_pseudo
#                 else:
#                     predict_func = self._predict_by_feat_single
                
#                 results = predict_func(
#                     cls_score_list, 
#                     bbox_pred_list, 
#                     angle_pred_list, 
#                     mlvl_priors, 
#                     data_sample, 
#                     cfg, 
#                     rescale, 
#                     with_nms)
            
#             result_list.append(results)
            
#         return result_list

#     # --- 实现细节: 原版推理 ---
#     def _predict_by_feat_original_style(self, cls_scores, bbox_preds, angle_preds, batch_data_samples, cfg, rescale, with_nms):
#         # 保持原有的 predict_by_feat 代码
#         assert len(cls_scores) == len(bbox_preds)
#         num_levels = len(cls_scores)
#         featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
#         mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
#         result_list = []
#         for img_id in range(len(batch_data_samples[2])):
#             data_sample = (batch_data_samples[0][img_id], batch_data_samples[1][img_id], batch_data_samples[2][img_id])
#             cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
#             bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
#             angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
            
#             if self.training or self.pseudo_generator:
#                 predict_func = self._predict_by_feat_single_pseudo
#             else:
#                 predict_func = self._predict_by_feat_single
                
#             results = predict_func(cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, data_sample, cfg, rescale, with_nms)
#             result_list.append(results)
#         return result_list

#     # --- 实现细节: ROI 推理 (新) ---
#     def _predict_by_feat_roi_style(self, bbox_preds, angle_preds, x, cls_scores, batch_data_samples, cfg, rescale, with_nms):
#         num_levels = len(bbox_preds)
#         featmap_sizes = [bbox_preds[i].shape[-2:] for i in range(num_levels)]
#         mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        
#         result_list = []
#         for img_id in range(len(batch_data_samples[2])):
#             data_sample = (batch_data_samples[0][img_id], batch_data_samples[1][img_id], batch_data_samples[2][img_id])
#             bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
#             angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
            
#             # 特征图切片
#             current_feat = x[0][img_id:img_id+1] 
            
#             # 单张图片推理
#             results = self._predict_single_roi(
#                 bbox_pred_list, angle_pred_list, current_feat, mlvl_priors, data_sample, cfg, rescale, with_nms)
#             result_list.append(results)
#         return result_list

#     def _predict_single_roi(self, bbox_pred_list, angle_pred_list, current_feat, mlvl_priors, data_sample, cfg, rescale, with_nms):
#         cfg = self.test_cfg if cfg is None else cfg
#         cfg = copy.deepcopy(cfg)
#         img_meta = data_sample[2]
        
#         # 1. 收集所有预测框
#         all_bbox_preds = []
#         all_angle_preds = []
#         all_priors = []
        
#         for (bbox_pred, angle_pred, priors) in zip(bbox_pred_list, angle_pred_list, mlvl_priors):
#             bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
#             all_bbox_preds.append(bbox_pred)
#             all_angle_preds.append(angle_pred)
#             all_priors.append(priors)
            
#         all_bbox_preds = torch.cat(all_bbox_preds)
#         all_angle_preds = torch.cat(all_angle_preds)
#         all_priors = torch.cat(all_priors)
        
#         # 2. 解码所有框
#         decoded_angle = self.angle_coder.decode(all_angle_preds, keepdim=True)
#         rboxes = torch.cat((all_priors + all_bbox_preds[:, 2:], 
#                             all_bbox_preds[:, :2] * 2, 
#                             decoded_angle), -1)
             
#         # 3. RoIAlign + Linear Classification (全量输入)
#         batch_inds = rboxes.new_zeros(rboxes.size(0), 1)
#         rois = torch.cat([batch_inds, rboxes], dim=1)
        
#         roi_feats = self.roi_extractor(current_feat, rois)
#         roi_logits = self.roi_cls_layer(roi_feats.view(roi_feats.size(0), -1))
#         scores = roi_logits.sigmoid() # (N, num_classes)
        
#         # 4. 后处理 (NMS)
#         score_thr = cfg.get('score_thr', 0.05)
        
#         max_scores, labels = scores.max(dim=1)
#         valid_mask = max_scores > score_thr
        
#         # 将过滤后的结果打包
#         results = InstanceData()
#         results.bboxes = RotatedBoxes(rboxes[valid_mask])
#         results.scores = max_scores[valid_mask] 
#         results.labels = labels[valid_mask]
        
#         # 执行 NMS
#         results = self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)
#         return results

#     def _predict_by_feat_single_pseudo(self, cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, data_sample, cfg, rescale=False, with_nms=True):
#         gt_instances = data_sample[0]
#         img_meta = data_sample[2]
#         scale_factor = [1, 1] if self.training else img_meta['scale_factor']
#         gt_bboxes = gt_instances.bboxes.tensor
#         gt_labels = gt_instances.labels
#         gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).long()
#         cls_score, bbox_pred, angle_pred = cls_score_list[0], bbox_pred_list[0], angle_pred_list[0]
#         H, W = cls_score.shape[1:3]
#         gt_valid_mask = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
#         gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
#         gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)
#         bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx]
#         cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)[gt_idx]
#         angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)[gt_idx]
#         decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
#         bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2, decoded_angle), -1)
#         bboxes[~gt_valid_mask, 2:] = 0
#         bboxes[:, 2:4] = bboxes[:, 2:4] / scale_factor[1]
#         for id in self.post_process.keys():
#             bboxes[gt_labels == id, 2:4] *= self.post_process[id]
#         for id in self.square_cls:
#             bboxes[gt_labels == id, -1] = 0
#         results = InstanceData()
#         results.bboxes = RotatedBoxes(bboxes.detach())
#         results.scores = torch.ones_like(cls_score[:, 0])
#         results.labels = gt_labels
#         return results
    
#     def _predict_by_feat_single(self, cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, data_sample, cfg, rescale=False, with_nms=True):
#         cfg = self.test_cfg if cfg is None else cfg
#         cfg = copy.deepcopy(cfg)
#         img_meta = data_sample[2]
#         nms_pre = cfg.get('nms_pre', -1)
#         mlvl_bbox_preds, mlvl_valid_priors, mlvl_scores, mlvl_labels = [], [], [], []
#         for level_idx, (cls_score, bbox_pred, angle_pred, priors) in enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors)):
#             assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
#             bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
#             cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
#             scores = cls_score.sigmoid() if self.use_sigmoid_cls else cls_score.softmax(-1)[:, :-1]
#             score_thr = cfg.get('score_thr', 0)
#             results = filter_scores_and_topk(scores, score_thr, nms_pre, dict(bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
#             scores, labels, keep_idxs, filtered_results = results
#             bbox_pred = filtered_results['bbox_pred']
#             angle_pred = filtered_results['angle_pred']
#             priors = filtered_results['priors']
#             decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
#             bbox_pred = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)
#             mlvl_bbox_preds.append(bbox_pred)
#             mlvl_valid_priors.append(priors)
#             mlvl_scores.append(scores)
#             mlvl_labels.append(labels)
#         scores = torch.cat(mlvl_scores)
#         labels = torch.cat(mlvl_labels)
#         bboxes = torch.cat(mlvl_bbox_preds)
#         priors = cat_boxes(mlvl_valid_priors)
#         for id in self.post_process.keys():
#             bboxes[labels == id, 2:4] *= self.post_process[id]
#         for id in self.square_cls:
#             bboxes[labels == id, -1] = 0
#         results = InstanceData()
#         results.bboxes = RotatedBoxes(bboxes)
#         results.scores = scores
#         results.labels = labels
#         results = self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)
#         return results
    
#     def get_targets(
#         self, points: List[Tensor], batch_gt_instances: InstanceList
#     ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
#         assert len(points) == len(self.regress_ranges)
#         num_levels = len(points)
#         expanded_regress_ranges = [
#             points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
#                 points[i]) for i in range(num_levels)
#         ]
#         concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
#         concat_points = torch.cat(points, dim=0)
#         num_points = [center.size(0) for center in points]

#         # 使用 multi_apply 并行处理 Batch 中的每张图
#         labels_list, bbox_targets_list, bid_targets_list = multi_apply(
#             self._get_targets_single,
#             batch_gt_instances,
#             points=concat_points,
#             regress_ranges=concat_regress_ranges,
#             num_points_per_lvl=num_points)

#         # 重组回各个 Level
#         labels_list = [labels.split(num_points, 0) for labels in labels_list]
#         bbox_targets_list = [
#             bbox_targets.split(num_points, 0)
#             for bbox_targets in bbox_targets_list
#         ]
#         bid_targets_list = [
#             bid_targets.split(num_points, 0)
#             for bid_targets in bid_targets_list
#         ]

#         concat_lvl_labels = []
#         concat_lvl_bbox_targets = []
#         concat_lvl_bid_targets = []
#         for i in range(num_levels):
#             concat_lvl_labels.append(
#                 torch.cat([labels[i] for labels in labels_list]))
#             bbox_targets = torch.cat(
#                 [bbox_targets[i] for bbox_targets in bbox_targets_list])
#             bid_targets = torch.cat(
#                 [bid_targets[i] for bid_targets in bid_targets_list])
#             concat_lvl_bbox_targets.append(bbox_targets)
#             concat_lvl_bid_targets.append(bid_targets)
            
#         return (concat_lvl_labels, concat_lvl_bbox_targets,
#                 concat_lvl_bid_targets)

#     def _get_targets_single(
#             self, gt_instances: InstanceData, points: Tensor,
#             regress_ranges: Tensor,
#             num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
#         num_points = points.size(0)
#         num_gts = len(gt_instances)
#         gt_bboxes = gt_instances.bboxes
#         gt_labels = gt_instances.labels
        
#         if hasattr(gt_instances, 'bids'):
#             gt_bids = gt_instances.bids
#         else:
#             gt_bids = points.new_zeros((num_gts, 4)) 

#         if num_gts == 0:
#             return gt_labels.new_full((num_points,), self.num_classes), \
#                    gt_bboxes.new_zeros((num_points, 4)), \
#                    gt_bids.new_zeros((num_points, 4))

#         areas = gt_bboxes.areas
#         gt_bboxes = gt_bboxes.tensor

#         areas = areas[None].repeat(num_points, 1)
#         regress_ranges = regress_ranges[:, None, :].expand(
#             num_points, num_gts, 2)
#         points = points[:, None, :].expand(num_points, num_gts, 2)
#         gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
#         gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        
#         offset = points - gt_ctr
#         w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

#         # center_r 计算
#         center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        
#         offset_x, offset_y = offset[..., 0], offset[..., 1]
#         left = w / 2 + offset_x
#         right = w / 2 - offset_x
#         top = h / 2 + offset_y
#         bottom = h / 2 - offset_y
#         bbox_targets = torch.stack((left, top, right, bottom), -1)

#         if self.center_sampling:
#             radius = self.center_sample_radius
#             stride = offset.new_zeros(offset.shape)
#             lvl_begin = 0
#             for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
#                 lvl_end = lvl_begin + num_points_lvl
#                 stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
#                 lvl_begin = lvl_end
                        
#             stride_tensor = torch.cat([points.new_full((n,), self.strides[i]) for i, n in enumerate(num_points_per_lvl)])
#             limit = self.center_sample_radius * stride_tensor[:, None] * center_r[..., 0]         
#             inside_gt_bbox_mask = (abs(offset) < limit.unsqueeze(-1)).all(dim=-1)

#         max_regress_distance = bbox_targets.max(-1)[0]
#         inside_regress_range = (
#             (max_regress_distance >= regress_ranges[..., 0])
#             & (max_regress_distance <= regress_ranges[..., 1]))

#         areas[inside_gt_bbox_mask == 0] = INF
#         areas[inside_regress_range == 0] = INF
#         min_area, min_area_inds = areas.min(dim=1)

#         labels = gt_labels[min_area_inds]
#         labels[min_area == INF] = self.num_classes 
#         bbox_targets = bbox_targets[range(num_points), min_area_inds]
#         angle_targets = gt_angle[range(num_points), min_area_inds]
#         bid_targets = gt_bids[min_area_inds]
#         bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

#         return labels, bbox_targets, bid_targets


# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Dict, List, Optional, Tuple

# import copy
# import torch
# import torch.nn as nn
# from mmdet.models.dense_heads import AnchorFreeHead
# from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
#                                 select_single_mlvl, unpack_gt_instances)
# from mmdet.structures import SampleList
# from mmdet.structures.bbox import cat_boxes
# from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
#                          OptInstanceList, reduce_mean)
# from mmengine import ConfigDict
# from mmengine.structures import InstanceData
# from torch import Tensor

# from mmrotate.registry import MODELS, TASK_UTILS
# from mmrotate.structures import RotatedBoxes
# # 引入可导 ROI Align 模块
# from mmrotate.models.utils import DifferentiableRoIAlignRotated

# INF = 1e8

# @MODELS.register_module()
# class Point2RBoxV2Head(AnchorFreeHead):
#     """
#     Point2RBoxV2Head (Cascade Refinement Version): 
#     1. 训练时：双流监督。ConvHead 负责全图背景抑制，ROIHead 负责正样本特征精修。
#     2. 推理时：级联筛选。ConvHead 初筛 Top-K -> ROIHead 精修 -> 分数融合 -> NMS。
#     """

#     def __init__(self,
#                  num_classes: int,
#                  in_channels: int,
#                  strides: list = [8],
#                  regress_ranges: list = [(-1, 1e8)],
#                  center_sampling: bool = True,
#                  center_sample_radius: float = 0.75,
#                  angle_version: str = 'le90',
#                  pseudo_generator: bool = False,
#                  voronoi_type: str = 'gaussian-orientation',
#                  voronoi_thres: dict = dict(default=[0.994, 0.005]), 
#                  square_cls: list = [],
#                  post_process: dict = {},
#                  bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
#                  angle_coder: ConfigType = dict(
#                     type='PSCCoder',
#                     angle_version='le90',
#                     dual_freq=False,
#                     num_step=3,
#                     thr_mod=0),
                 
#                  # --- [核心开关] ---
#                  cls_mode: str = 'cls',      # 'cls' (Baseline) 或 'roicls' (级联模式)
#                  geo_mode: str = 'voronoi',
                 
#                  # --- [Loss Configs] ---
#                  loss_cls: Optional[ConfigType] = None,
#                  loss_roicls: Optional[ConfigType] = None,
#                  roi_size: tuple = (7, 7),
                 
#                  loss_voronoi: Optional[ConfigType] = None,
#                  loss_ourwater: Optional[ConfigType] = None,
                 
#                  loss_size: ConfigType = dict(
#                      type='SizeLoss', loss_weight=1.0),
#                  loss_angle: ConfigType = dict(
#                      type='AngleLoss', loss_weight=1.0),
                 
#                  norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
#                  init_cfg=dict(
#                      type='Normal',
#                      layer='Conv2d',
#                      std=0.01,
#                      override=[
#                          dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01), 
#                          dict(type='Normal', name='conv_gate', std=0.01, bias_prob=0.01)]),
#                  **kwargs):
        
#         self.angle_coder = TASK_UTILS.build(angle_coder)
        
#         # 为了父类兼容性，构建一个 dummy loss
#         dummy_loss = dict(type='mmdet.L1Loss', loss_weight=0.0)

#         super().__init__(
#             num_classes,
#             in_channels,
#             strides=strides,
#             bbox_coder=bbox_coder,
#             # 这里必须传入真实的 loss_cls，因为它负责全图背景抑制
#             loss_cls=loss_cls if loss_cls is not None else dummy_loss, 
#             loss_bbox=dummy_loss, 
#             norm_cfg=norm_cfg,
#             init_cfg=init_cfg,
#             **kwargs)
            
#         self.regress_ranges = regress_ranges
#         self.center_sampling = center_sampling
#         self.center_sample_radius = center_sample_radius
#         self.angle_version = angle_version
#         self.pseudo_generator = pseudo_generator
#         self.voronoi_type = voronoi_type
#         self.voronoi_thres = voronoi_thres
#         self.square_cls = square_cls
#         self.post_process = post_process
        
#         self.cls_mode = cls_mode
#         self.geo_mode = geo_mode
        
#         # 1. 构建 ROI 组件 (级联精修模式)
#         if self.cls_mode == 'roicls':
#             if loss_roicls is None:
#                  raise ValueError("In 'roicls' mode, 'loss_roicls' config must be provided!")
            
#             self.roi_extractor = DifferentiableRoIAlignRotated(
#                 output_size=roi_size,
#                 spatial_scale=1.0 / strides[0], 
#                 sampling_ratio=2 
#             )
#             flat_dim = self.feat_channels * roi_size[0] * roi_size[1]
#             self.roi_cls_layer = nn.Linear(flat_dim, num_classes)
#             self.loss_roicls = MODELS.build(loss_roicls)
        
#         # 2. 构建几何/辅助 Loss
#         if self.geo_mode == 'voronoi' and loss_voronoi:
#             self.loss_voronoi = MODELS.build(loss_voronoi)
#         elif self.geo_mode == 'ourwater' and loss_ourwater:
#             self.loss_ourwater = MODELS.build(loss_ourwater)
            
#         self.loss_size = MODELS.build(loss_size)
#         self.loss_angle = MODELS.build(loss_angle)
            
#     def _init_layers(self):
#         super()._init_layers()
#         self.conv_angle = nn.Conv2d(
#             self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
#         self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
#     def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
#         cls_feat = x[0]
#         reg_feat = x[0]

#         for cls_layer in self.cls_convs:
#             cls_feat = cls_layer(cls_feat)
        
#         # 无论什么模式，conv_cls 都要跑一遍（为了结构完整性 + 背景抑制）
#         cls_score = self.conv_cls(cls_feat)

#         for reg_layer in self.reg_convs:
#             reg_feat = reg_layer(reg_feat)
#         bbox_pred = self.conv_reg(reg_feat)
#         angle_pred = self.conv_angle(reg_feat)

#         sig_x = bbox_pred[:, 0].exp()
#         sig_y = bbox_pred[:, 1].exp()
#         dx = bbox_pred[:, 2].sigmoid() * 2 - 1 
#         dy = bbox_pred[:, 3].sigmoid() * 2 - 1 
#         bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

#         return (cls_score,), (bbox_pred,), (angle_pred,)
    
#     def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
#         outs = self(x)
#         outputs = unpack_gt_instances(batch_data_samples)
#         (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs
#         loss_inputs = outs + (x, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
#         return self.loss_by_feat(*loss_inputs)

#     def loss_by_feat(
#         self,
#         cls_scores: List[Tensor],
#         bbox_preds: List[Tensor],
#         angle_preds: List[Tensor],
#         rois_features: Tuple[Tensor],
#         batch_gt_instances: InstanceList,
#         batch_img_metas: List[dict],
#         batch_gt_instances_ignore: OptInstanceList = None
#     ) -> Dict[str, Tensor]:
        
#         # --- 1. 准备 Ground Truth ---
#         featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
#         all_level_points = self.prior_generator.grid_priors(
#             featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        
#         labels, bbox_targets, bid_targets = self.get_targets(all_level_points, batch_gt_instances)
#         num_imgs = bbox_preds[0].size(0)
        
#         # 展平所有数据
#         flatten_bbox_preds = torch.cat([bp.permute(0, 2, 3, 1).reshape(-1, 4) for bp in bbox_preds])
#         flatten_angle_preds = torch.cat([ap.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size) for ap in angle_preds])
#         flatten_cls_scores = torch.cat([cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])

#         flatten_labels = torch.cat(labels)
#         flatten_bbox_targets = torch.cat(bbox_targets)
#         flatten_bid_targets = torch.cat(bid_targets)
#         flatten_points = torch.cat([p.repeat(num_imgs, 1) for p in all_level_points])

#         loss_dict = dict()
#         zero_tensor = flatten_bbox_preds.new_tensor(0.0)
        
#         bg_class_ind = self.num_classes
#         pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
#         num_pos = len(pos_inds) 

#         # ====================================================================
#         # 第一部分：全图 Lcls Loss (负责广度/背景抑制)
#         # ====================================================================
#         # 无论何种模式，都计算这个 Loss，保证背景被压制
#         avg_factor = max(reduce_mean(torch.tensor(num_pos, dtype=torch.float, device=bbox_preds[0].device)), 1.0)
#         loss_dict['loss_cls'] = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=avg_factor)

#         # ====================================================================
#         # 第二部分：正样本 ROI Loss (负责精度/特征对齐精修)
#         # ====================================================================
#         if self.cls_mode == 'roicls' and len(pos_inds) > 0:
#             # 1. 准备训练用的 Box (仅正样本)
#             pos_bbox_preds_train = flatten_bbox_preds[pos_inds]
#             pos_angle_preds_train = flatten_angle_preds[pos_inds]
#             pos_points_train = flatten_points[pos_inds]
#             pos_bid_targets_train = flatten_bid_targets[pos_inds]
#             pos_labels_train = flatten_labels[pos_inds]
            
#             # 解码角度
#             decoded_angle = self.angle_coder.decode(pos_angle_preds_train, keepdim=True)
#             for c in self.square_cls:
#                 decoded_angle[pos_labels_train == c] = 0
            
#             # 组合成绝对坐标框
#             pos_rbox_preds = torch.cat((pos_points_train + pos_bbox_preds_train[:, 2:], 
#                                         pos_bbox_preds_train[:, :2] * 2,
#                                         decoded_angle), -1)
            
#             # 2. 构造 ROIs
#             batch_inds = pos_bid_targets_train[:, 0].float().unsqueeze(1)
#             rois = torch.cat([batch_inds, pos_rbox_preds], dim=1)
            
#             # 3. 运行 ROI Cls
#             current_feat = rois_features[0] 
#             roi_feats = self.roi_extractor(current_feat, rois)
#             roi_logits = self.roi_cls_layer(roi_feats.view(roi_feats.size(0), -1))
            
#             # 4. 计算 Loss (CrossEntropy)
#             # 这里的 avg_factor 就是 num_pos，因为只算了正样本
#             loss_dict['loss_roicls'] = self.loss_roicls(
#                 roi_logits, pos_labels_train, avg_factor=num_pos)

#         # ====================================================================
#         # 第三部分：几何 Loss 和 辅助 Loss (只针对正样本)
#         # ====================================================================
#         if len(pos_inds) > 0:
#             # 重新提取正样本数据
#             pos_bbox_preds = flatten_bbox_preds[pos_inds]
#             pos_angle_preds = flatten_angle_preds[pos_inds]
#             pos_bbox_targets = flatten_bbox_targets[pos_inds]
#             pos_bid_targets = flatten_bid_targets[pos_inds]
#             pos_points = flatten_points[pos_inds]
#             pos_labels = flatten_labels[pos_inds]
            
#             pos_decoded_angle = self.angle_coder.decode(pos_angle_preds, keepdim=True)
#             for c in self.square_cls:
#                 pos_decoded_angle[pos_labels == c] = 0
            
#             pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
#             pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
#                                         pos_bbox_preds[:, :2] * 2,
#                                         pos_decoded_angle), -1)
            
#             # 准备 Score 用于 Loss Size 的加权
#             if self.cls_mode == 'roicls':
#                 # 如果开启了 ROI，优先使用 ROI 的分数（因为它更准）
#                 # 注意：上面计算 loss_roicls 时生成了 roi_logits
#                 # 我们这里可以直接复用，或者简单起见用 sigmoid
#                 pos_scores = roi_logits.sigmoid()
#                 pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]
#             else:
#                 pos_scores = flatten_cls_scores[pos_inds].sigmoid()
#                 pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]

#             loss_dict['loss_size'] = self.loss_size(
#                 pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0], batch_img_metas)
#             loss_dict['loss_angle'] = self.loss_angle(
#                 pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0])
            
#             # --- Geometry Loss (Voronoi / OurWater) ---
#             cos_r, sin_r = torch.cos(pos_decoded_angle), torch.sin(pos_decoded_angle)
#             R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
#             pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))
            
#             bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
#             bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
#             ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
#             ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
#             ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(0, idx, pos_labels, 'amin', include_self=False)
            
#             ins_gaus_preds = pos_gaus_preds.new_zeros(*bid.shape, 4).index_reduce_(0, idx, pos_gaus_preds.view(-1, 4), 'mean', include_self=False).view(-1, 2, 2)
#             ins_rbox_targets = pos_rbox_targets.new_zeros(*bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(0, idx, pos_rbox_targets, 'mean', include_self=False)
#             ori_mu_all = ins_rbox_targets[:, 0:2]

#             if self.geo_mode == 'voronoi':
#                 loss_geo = zero_tensor.clone()
#                 for batch_id in range(len(batch_gt_instances)):
#                     group_mask = (ins_batch == batch_id) & (ins_bids != 0)
#                     mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
#                     if len(mu) >= 1:
#                         pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
#                         neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
#                         if 'override' in self.voronoi_thres.keys():
#                             for item in self.voronoi_thres['override']:
#                                 for cls in item[0]:
#                                     pos_thres[cls] = item[1][0]
#                                     neg_thres[cls] = item[1][1]
#                         loss_geo += self.loss_voronoi(
#                             (mu, sigma.bmm(sigma)), label, self.images[batch_id],
#                             pos_thres, neg_thres, voronoi=self.voronoi_type)
#                 loss_dict['loss_bbox_vor'] = loss_geo / len(batch_gt_instances)
                
#             elif self.geo_mode == 'ourwater':
#                 loss_geo = zero_tensor.clone()
#                 for batch_id in range(len(batch_gt_instances)):
#                     group_mask = (ins_batch == batch_id) & (ins_bids != 0)
#                     mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
#                     if len(mu) >= 1:
#                         pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
#                         neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
#                         if 'override' in self.voronoi_thres.keys():
#                             for item in self.voronoi_thres['override']:
#                                 for cls in item[0]:
#                                     pos_thres[cls] = item[1][0]
#                                     neg_thres[cls] = item[1][1]
#                         loss_geo += self.loss_ourwater(
#                             (mu, sigma.bmm(sigma)), label, self.images[batch_id],
#                             pos_thres, neg_thres, voronoi=self.voronoi_type)
#                 loss_dict['loss_ourwater'] = loss_geo / len(batch_gt_instances)

#         return loss_dict

#     # ====================================================================
#     # 核心推理逻辑: 级联筛选 (Cascade)
#     # ====================================================================
#     def predict(self,
#                 x: Tuple[Tensor],
#                 batch_data_samples: SampleList,
#                 rescale: bool = False) -> InstanceList:
#         outs = self(x)
#         # 直接传递原始 batch_data_samples，修复 TypeError
#         predictions = self.predict_by_feat(
#             *outs, 
#             batch_data_samples=batch_data_samples, 
#             rescale=rescale,
#             x=x)
#         return predictions
    
#     def predict_by_feat(self,
#                         cls_scores: List[Tensor],
#                         bbox_preds: List[Tensor],
#                         angle_preds: List[Tensor],
#                         batch_data_samples: Optional[List[dict]] = None,
#                         cfg: Optional[ConfigDict] = None,
#                         rescale: bool = False,
#                         with_nms: bool = True,
#                         x: Optional[Tuple[Tensor]] = None,
#                         **kwargs) -> InstanceList:
        
#         assert len(cls_scores) == len(bbox_preds)
#         num_levels = len(cls_scores)
#         featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
#         mlvl_priors = self.prior_generator.grid_priors(
#             featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)

#         result_list = []
#         for img_id in range(len(batch_data_samples)):
#             img_meta = batch_data_samples[img_id].metainfo
            
#             # [核心分支] 级联推理模式
#             if self.cls_mode == 'roicls':
#                 current_feat = x[0][img_id:img_id+1]
#                 bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
#                 angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
#                 cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
                
#                 results = self._predict_single_cascade(
#                     cls_score_list,
#                     bbox_pred_list, 
#                     angle_pred_list, 
#                     current_feat, 
#                     mlvl_priors, 
#                     img_meta, 
#                     cfg, 
#                     rescale, 
#                     with_nms)
            
#             # [兼容分支] 原始推理模式
#             else:
#                 cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
#                 bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
#                 angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
                
#                 if self.training or self.pseudo_generator:
#                     # 构造伪标签用的数据
#                     gt_instances = batch_data_samples[img_id].gt_instances
#                     data_sample = (gt_instances, None, img_meta)
#                     predict_func = self._predict_by_feat_single_pseudo
#                     results = predict_func(
#                         cls_score_list, bbox_pred_list, angle_pred_list, 
#                         mlvl_priors, data_sample, cfg, rescale, with_nms)
#                 else:
#                     predict_func = self._predict_by_feat_single
#                     results = predict_func(
#                         cls_score_list, bbox_pred_list, angle_pred_list, 
#                         mlvl_priors, img_meta, cfg, rescale, with_nms)
            
#             result_list.append(results)
#         return result_list

#     # --- 新增: 级联推理函数 ---
#     def _predict_single_cascade(self, cls_score_list, bbox_pred_list, angle_pred_list, current_feat, mlvl_priors, img_meta, cfg, rescale, with_nms):
#         cfg = self.test_cfg if cfg is None else cfg
#         nms_pre = cfg.get('nms_pre', 2000) # 初筛数量
        
#         # 1. 收集所有层级的候选框
#         mlvl_bboxes = []
#         mlvl_scores = []
#         mlvl_priors_list = []
        
#         for (cls_score, bbox_pred, angle_pred, priors) in zip(cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors):
#             bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
#             cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
#             scores = cls_score.sigmoid()
            
#             # 第一步过滤: 去掉极低分背景
#             valid_mask = scores.max(dim=1)[0] > 0.05
            
#             if valid_mask.sum() == 0:
#                 continue
                
#             scores = scores[valid_mask]
#             bbox_pred = bbox_pred[valid_mask]
#             angle_pred = angle_pred[valid_mask]
#             priors = priors[valid_mask]
            
#             decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
#             bboxes = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)
            
#             mlvl_bboxes.append(bboxes)
#             mlvl_scores.append(scores)
        
#         if len(mlvl_bboxes) == 0:
#             return InstanceData()
            
#         proposals = torch.cat(mlvl_bboxes)
#         conv_scores = torch.cat(mlvl_scores)
        
#         # 2. Top-K 筛选 (扔掉 90% 的垃圾)
#         if proposals.shape[0] > nms_pre:
#             # 根据 Conv 分数排序
#             max_scores, _ = conv_scores.max(dim=1)
#             _, topk_inds = max_scores.topk(nms_pre)
#             proposals = proposals[topk_inds]
#             conv_scores = conv_scores[topk_inds]
            
#         # 3. ROI 精修
#         batch_inds = proposals.new_zeros(proposals.size(0), 1)
#         rois = torch.cat([batch_inds, proposals], dim=1)
        
#         roi_feats = self.roi_extractor(current_feat, rois)
#         roi_logits = self.roi_cls_layer(roi_feats.view(roi_feats.size(0), -1))
#         roi_scores = roi_logits.softmax(dim=1) # 或者是 sigmoid，取决于训练时的 Loss
        
#         # 4. 分数融合 (Fusion)
#         # 公式: final = sqrt(conv * roi)
#         final_scores = torch.sqrt(conv_scores * roi_scores)
        
#         # 5. 准备 NMS
#         # 此时还没有 label，我们需要 argmax 拿到 label
#         max_scores, labels = final_scores.max(dim=1)
        
#         # 过滤低分
#         valid_mask = max_scores > cfg.get('score_thr', 0.05)
        
#         results = InstanceData()
#         results.bboxes = RotatedBoxes(proposals[valid_mask])
#         results.scores = max_scores[valid_mask]
#         results.labels = labels[valid_mask]
        
#         # 6. 后处理 (NMS)
#         return self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)

#     # --- 保持原有辅助函数 ---
#     def _predict_by_feat_single(self, cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, img_meta, cfg, rescale=False, with_nms=True):
#         # [修改] 参数列表去掉了 data_sample，直接接收 img_meta
#         # 这是 Baseline 模式的推理逻辑
#         nms_pre = cfg.get('nms_pre', -1)
#         mlvl_bbox_preds, mlvl_valid_priors, mlvl_scores, mlvl_labels = [], [], [], []
#         for (cls_score, bbox_pred, angle_pred, priors) in zip(cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors):
#             bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
#             angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
#             cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
#             scores = cls_score.sigmoid()
#             results = filter_scores_and_topk(scores, cfg.get('score_thr', 0), nms_pre, dict(bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
#             scores, labels, _, filtered_results = results
#             bbox_pred = filtered_results['bbox_pred']
#             angle_pred = filtered_results['angle_pred']
#             priors = filtered_results['priors']
#             decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
#             bbox_pred = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)
#             mlvl_bbox_preds.append(bbox_pred)
#             mlvl_scores.append(scores)
#             mlvl_labels.append(labels)
        
#         if len(mlvl_bbox_preds) == 0:
#              return InstanceData()
             
#         results = InstanceData()
#         results.bboxes = RotatedBoxes(torch.cat(mlvl_bbox_preds))
#         results.scores = torch.cat(mlvl_scores)
#         results.labels = torch.cat(mlvl_labels)
#         return self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)

#     def _predict_by_feat_single_pseudo(self, cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, data_sample, cfg, rescale=False, with_nms=True):
#         gt_instances, _, img_meta = data_sample
#         scale = [1, 1] if self.training else img_meta['scale_factor']
#         gt_bboxes, gt_labels = gt_instances.bboxes.tensor, gt_instances.labels
#         gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale[1]).long()
#         cls_s, bbox_p, angle_p = cls_score_list[0], bbox_pred_list[0], angle_pred_list[0]
#         H, W = cls_s.shape[1:3]
#         valid = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
#         idx = (gt_pos[:, 1] * W + gt_pos[:, 0]).clamp(0, cls_s[0].numel() - 1)
#         bbox_p = bbox_p.permute(1, 2, 0).reshape(-1, 4)[idx]
#         angle_p = angle_p.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)[idx]
#         decoded_a = self.angle_coder.decode(angle_p, keepdim=True)
#         bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_p[:, :2] * 2, decoded_a), -1)
#         bboxes[~valid, 2:] = 0
#         bboxes[:, 2:4] /= scale[1]
#         for i, s in self.post_process.items(): bboxes[gt_labels == i, 2:4] *= s
#         for i in self.square_cls: bboxes[gt_labels == i, -1] = 0
#         return InstanceData(bboxes=RotatedBoxes(bboxes.detach()), scores=torch.ones_like(idx, dtype=torch.float), labels=gt_labels)

#     def get_targets(self, points, batch_gt_instances):
#         num_levels = len(points)
#         expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
#         concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
#         concat_points = torch.cat(points, dim=0)
#         num_points_per_lvl = [p.size(0) for p in points]
#         res = multi_apply(self._get_targets_single, batch_gt_instances, points=concat_points, regress_ranges=concat_regress_ranges, num_points_per_lvl=num_points_per_lvl)
#         return ([labels.split(num_points_per_lvl, 0) for labels in res[0]], [bbox.split(num_points_per_lvl, 0) for bbox in res[1]], [bid.split(num_points_per_lvl, 0) for bid in res[2]])

#     def _get_targets_single(self, gt_instances, points, regress_ranges, num_points_per_lvl):
#         num_p, num_gts = points.size(0), len(gt_instances)
#         if num_gts == 0: return gt_instances.labels.new_full((num_p,), self.num_classes), gt_instances.bboxes.tensor.new_zeros((num_p, 5)), points.new_zeros((num_p, 4))
#         gt_bboxes, gt_labels = gt_instances.bboxes.tensor, gt_instances.labels
#         gt_bids = gt_instances.bids if hasattr(gt_instances, 'bids') else points.new_zeros((num_gts, 4))
#         areas = gt_instances.bboxes.areas[None].repeat(num_p, 1)
#         regress_ranges = regress_ranges[:, None, :].expand(num_p, num_gts, 2)
#         points = points[:, None, :].expand(num_p, num_gts, 2)
#         gt_bboxes = gt_bboxes[None].expand(num_p, num_gts, 5)
#         offset = points - gt_bboxes[..., :2]
#         w, h = gt_bboxes[..., 2].clone(), gt_bboxes[..., 3].clone()
#         bbox_targets = torch.stack((w/2 + offset[...,0], h/2 + offset[...,1], w/2 - offset[...,0], h/2 - offset[...,1]), -1)
        
#         # 广播修正: (N, M, 1)
#         stride_tensor = torch.cat([points.new_full((n,), self.strides[i]) for i, n in enumerate(num_points_per_lvl)])
#         center_sampling_radius = self.center_sample_radius * stride_tensor[:, None] * torch.clamp((w*h).sqrt()/64, 1, 5)[..., None]
#         inside_gt = (abs(offset) < center_sampling_radius.unsqueeze(-1)).all(dim=-1)
        
#         max_reg = bbox_targets.max(-1)[0]
#         inside_reg = (max_reg >= regress_ranges[..., 0]) & (max_reg <= regress_ranges[..., 1])
#         areas[inside_gt == 0] = INF; areas[inside_reg == 0] = INF
#         min_area, min_idx = areas.min(dim=1)
#         labels = gt_labels[min_idx]; labels[min_area == INF] = self.num_classes
#         return labels, torch.cat((bbox_targets[range(num_p), min_idx], gt_bboxes[range(num_p), min_idx, 4:5]), -1), gt_bids[min_idx]

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, unpack_gt_instances)
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes
from mmrotate.models.utils import DifferentiableRoIAlignRotated

INF = 1e8

@MODELS.register_module()
class Point2RBoxV2Head(AnchorFreeHead):
    """
    Point2RBoxV2Head (Clean Config Version): 
    移除了 voronoi_type 的冗余配置，由 Loss 模块自行管理。
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 strides: list = [8],
                 regress_ranges: list = [(-1, 1e8)],
                 center_sampling: bool = True,
                 center_sample_radius: float = 0.75,
                 angle_version: str = 'le90',
                 pseudo_generator: bool = False,
                 # [修改] 移除了 voronoi_type，因为它应该在 loss config 里配置
                 voronoi_thres: dict = dict(default=[0.994, 0.005]), 
                 square_cls: list = [],
                 post_process: dict = {},
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 angle_coder: ConfigType = dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
                 
                 # --- Switches ---
                 cls_mode: str = 'cls',
                 geo_mode: str = 'voronoi',
                 debug_roi_input: bool = False,
                 
                 # --- Loss Configs ---
                 loss_cls: Optional[ConfigType] = None,
                 loss_roicls: Optional[ConfigType] = None,
                 roi_size: tuple = (7, 7),
                 
                 loss_voronoi: Optional[ConfigType] = None,
                 loss_ourwater: Optional[ConfigType] = None,
                 
                 loss_size: ConfigType = dict(
                     type='SizeLoss', loss_weight=1.0),
                 loss_angle: ConfigType = dict(
                     type='AngleLoss', loss_weight=1.0),
                 
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(type='Normal', name='conv_cls', std=0.01, bias_prob=0.01), 
                         dict(type='Normal', name='conv_gate', std=0.01, bias_prob=0.01)]),
                 **kwargs):
        
        self.angle_coder = TASK_UTILS.build(angle_coder)
        dummy_loss = dict(type='mmdet.L1Loss', loss_weight=0.0)

        super().__init__(
            num_classes,
            in_channels,
            strides=strides,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls if loss_cls is not None else dummy_loss, 
            loss_bbox=dummy_loss, 
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
            
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.angle_version = angle_version
        self.pseudo_generator = pseudo_generator
        # self.voronoi_type = voronoi_type  <-- [修改] 删除此行
        self.voronoi_thres = voronoi_thres
        self.square_cls = square_cls
        self.post_process = post_process
        
        self.cls_mode = cls_mode
        self.geo_mode = geo_mode
        self.debug_roi_input = debug_roi_input
        self.real_loss_cls_cfg = loss_cls
        
        if self.cls_mode == 'roicls':
            if loss_roicls is None:
                 raise ValueError("In 'roicls' mode, 'loss_roicls' config must be provided!")
            
            self.roi_extractor = DifferentiableRoIAlignRotated(
                output_size=roi_size,
                spatial_scale=1.0 / strides[0], 
                sampling_ratio=2 
            )
            # AvgPool 模式
            flat_dim = self.feat_channels 
            self.roi_cls_layer = nn.Linear(flat_dim, num_classes)
            self.loss_roicls = MODELS.build(loss_roicls)
        
        if self.geo_mode == 'voronoi' and loss_voronoi:
            self.loss_voronoi = MODELS.build(loss_voronoi)
        elif self.geo_mode == 'ourwater' and loss_ourwater:
            self.loss_ourwater = MODELS.build(loss_ourwater)
            
        self.loss_size = MODELS.build(loss_size)
        self.loss_angle = MODELS.build(loss_angle)
            
    def _init_layers(self):
        super()._init_layers()
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        cls_feat = x[0]
        reg_feat = x[0]

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        sig_x = bbox_pred[:, 0].exp()
        sig_y = bbox_pred[:, 1].exp()
        dx = bbox_pred[:, 2].sigmoid() * 2 - 1 
        dy = bbox_pred[:, 3].sigmoid() * 2 - 1 
        bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

        return (cls_score,), (bbox_pred,), (angle_pred,)
    
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        outs = self(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs
        loss_inputs = outs + (x, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        return self.loss_by_feat(*loss_inputs)

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        rois_features: Tuple[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        
        # --- 1. 准备 Ground Truth ---
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        
        labels, bbox_targets, bid_targets = self.get_targets(all_level_points, batch_gt_instances)
        num_imgs = bbox_preds[0].size(0)
        
        flatten_bbox_preds = torch.cat([bp.permute(0, 2, 3, 1).reshape(-1, 4) for bp in bbox_preds])
        flatten_angle_preds = torch.cat([ap.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size) for ap in angle_preds])
        flatten_cls_scores = torch.cat([cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bid_targets = torch.cat(bid_targets)
        flatten_points = torch.cat([p.repeat(num_imgs, 1) for p in all_level_points])

        loss_dict = dict()
        zero_tensor = flatten_bbox_preds.new_tensor(0.0)
        
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds) 

        # 1. Conv Head Loss (Baseline)
        if self.cls_mode == 'cls':
            avg_factor = max(reduce_mean(torch.tensor(num_pos, dtype=torch.float, device=bbox_preds[0].device)), 1.0)
            loss_dict['loss_cls'] = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=avg_factor)
            
        # 2. ROI Head Loss (AvgPool Version)
        elif self.cls_mode == 'roicls':
            # 1. 负样本采样 (实验：全量负样本)
            neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
            sampled_neg_inds = neg_inds 
            
            train_inds = torch.cat([pos_inds, sampled_neg_inds])
            
            # 2. 准备训练用的 Box
            train_labels = flatten_labels[train_inds]
            train_bbox_preds = flatten_bbox_preds[train_inds]
            train_angle_preds = flatten_angle_preds[train_inds]
            train_points = flatten_points[train_inds]
            train_bid_targets = flatten_bid_targets[train_inds]
            
            decoded_angle = self.angle_coder.decode(train_angle_preds, keepdim=True)
            if len(self.square_cls) > 0:
                for c in self.square_cls:
                    decoded_angle[train_labels == c] = 0
            
            train_rbox_preds = torch.cat((train_points + train_bbox_preds[:, 2:], 
                                          train_bbox_preds[:, :2] * 2,
                                          decoded_angle), -1)
            
            # --- [DEBUG START] ---
            if self.debug_roi_input:
                if not hasattr(self, '_debug_logged'):
                    print(f"\n[DEBUG MODE] Using Fake GT ROIs (1x1) for ALL samples.")
                    self._debug_logged = True
                
                d_cx = train_points[:, 0]
                d_cy = train_points[:, 1]
                # 使用 stride 大小
                d_w = torch.ones_like(d_cx) * self.strides[0]
                d_h = torch.ones_like(d_cy) * self.strides[0]
                d_t = torch.zeros_like(d_cx)
                
                debug_rboxes = torch.stack([d_cx, d_cy, d_w, d_h, d_t], dim=-1)
                batch_inds = train_bid_targets[:, 0].float().unsqueeze(1)
                rois = torch.cat([batch_inds, debug_rboxes], dim=1)
            else:
                batch_inds = train_bid_targets[:, 0].float().unsqueeze(1)
                rois = torch.cat([batch_inds, train_rbox_preds], dim=1)
            # --- [DEBUG END] ---
            
            # 3. 运行 ROI Cls (AvgPool)
            current_feat = rois_features[0] 
            roi_feats = self.roi_extractor(current_feat, rois)
            roi_feats_pooled = F.adaptive_avg_pool2d(roi_feats, (1, 1))
            roi_logits = self.roi_cls_layer(roi_feats_pooled.view(roi_feats_pooled.size(0), -1))
            
            avg_factor = max(num_pos, 1.0)
            loss_dict['loss_roicls'] = self.loss_roicls(
                roi_logits, train_labels, avg_factor=avg_factor)

        # 3. 几何/辅助 Loss (仅正样本)
        if len(pos_inds) > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_angle_preds = flatten_angle_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_bid_targets = flatten_bid_targets[pos_inds]
            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]
            
            pos_decoded_angle = self.angle_coder.decode(pos_angle_preds, keepdim=True)
            for c in self.square_cls:
                pos_decoded_angle[pos_labels == c] = 0
            
            pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle), -1)
            
            if self.cls_mode == 'cls':
                pos_scores = flatten_cls_scores[pos_inds].sigmoid()
                pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]
            else:
                pos_roi_logits_subset = roi_logits[:num_pos]
                pos_scores = pos_roi_logits_subset.sigmoid()
                pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]

            loss_dict['loss_size'] = self.loss_size(
                pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0], batch_img_metas)
            loss_dict['loss_angle'] = self.loss_angle(
                pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0])
            
            cos_r, sin_r = torch.cos(pos_decoded_angle), torch.sin(pos_decoded_angle)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))
            
            bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
            bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
            ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
            ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
            ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(0, idx, pos_labels, 'amin', include_self=False)
            
            ins_gaus_preds = pos_gaus_preds.new_zeros(*bid.shape, 4).index_reduce_(0, idx, pos_gaus_preds.view(-1, 4), 'mean', include_self=False).view(-1, 2, 2)
            ins_rbox_targets = pos_rbox_targets.new_zeros(*bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(0, idx, pos_rbox_targets, 'mean', include_self=False)
            ori_mu_all = ins_rbox_targets[:, 0:2]

            if self.geo_mode == 'voronoi':
                loss_geo = zero_tensor.clone()
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
                    if len(mu) >= 1:
                        pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                        neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                        if 'override' in self.voronoi_thres.keys():
                            for item in self.voronoi_thres['override']:
                                for cls in item[0]:
                                    pos_thres[cls] = item[1][0]
                                    neg_thres[cls] = item[1][1]
                        # [修改] 移除了 voronoi=self.voronoi_type 传参
                        loss_geo += self.loss_voronoi(
                            (mu, sigma.bmm(sigma)), label, self.images[batch_id],
                            pos_thres, neg_thres)
                loss_dict['loss_bbox_vor'] = loss_geo / len(batch_gt_instances)
                
            elif self.geo_mode == 'ourwater':
                loss_geo = zero_tensor.clone()
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
                    if len(mu) >= 1:
                        pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                        neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                        if 'override' in self.voronoi_thres.keys():
                            for item in self.voronoi_thres['override']:
                                for cls in item[0]:
                                    pos_thres[cls] = item[1][0]
                                    neg_thres[cls] = item[1][1]
                        # [修改] 移除了 voronoi=self.voronoi_type 传参
                        loss_geo += self.loss_ourwater(
                            (mu, sigma.bmm(sigma)), label, self.images[batch_id],
                            pos_thres, neg_thres)
                loss_dict['loss_ourwater'] = loss_geo / len(batch_gt_instances)

        return loss_dict
    # def loss_by_feat(
    #     self,
    #     cls_scores: List[Tensor],
    #     bbox_preds: List[Tensor],
    #     angle_preds: List[Tensor],
    #     rois_features: Tuple[Tensor],
    #     batch_gt_instances: InstanceList,
    #     batch_img_metas: List[dict],
    #     batch_gt_instances_ignore: OptInstanceList = None
    # ) -> Dict[str, Tensor]:
        
    #     # [新增] 引入时间库和定义计时函数
    #     import time
    #     timers = {}
    #     def tick(tag):
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         timers[tag] = time.time()

    #     tick('start') # --- [T0] 开始 ---

    #     # --- 1. 准备 Ground Truth (两者共用) ---
    #     featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
    #     all_level_points = self.prior_generator.grid_priors(
    #         featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        
    #     labels, bbox_targets, bid_targets = self.get_targets(all_level_points, batch_gt_instances)
    #     num_imgs = bbox_preds[0].size(0)
        
    #     # 展平所有数据
    #     flatten_bbox_preds = torch.cat([bp.permute(0, 2, 3, 1).reshape(-1, 4) for bp in bbox_preds])
    #     flatten_angle_preds = torch.cat([ap.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size) for ap in angle_preds])
    #     flatten_cls_scores = torch.cat([cs.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cs in cls_scores])

    #     flatten_labels = torch.cat(labels)
    #     flatten_bbox_targets = torch.cat(bbox_targets)
    #     flatten_bid_targets = torch.cat(bid_targets)
    #     flatten_points = torch.cat([p.repeat(num_imgs, 1) for p in all_level_points])

    #     loss_dict = dict()
    #     zero_tensor = flatten_bbox_preds.new_tensor(0.0)
        
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
    #     num_pos = len(pos_inds) 

    #     tick('prep_done') # --- [T1] 数据准备结束 ---

    #     # ====================================================================
    #     # 1. Conv Head Loss (Baseline 模式)
    #     # ====================================================================
    #     if self.cls_mode == 'cls':
    #         # 计时：Lcls 计算开始
    #         tick('lcls_start')
            
    #         avg_factor = max(reduce_mean(torch.tensor(num_pos, dtype=torch.float, device=bbox_preds[0].device)), 1.0)
    #         loss_dict['loss_cls'] = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=avg_factor)
            
    #         # 计时：Lcls 计算结束
    #         tick('lcls_end')
            
    #         # [打印 Baseline 耗时]
    #         # 只有10%的概率打印，防止刷屏
    #         if torch.rand(1).item() < 0.1:
    #             t_prep = (timers['prep_done'] - timers['start']) * 1000
    #             t_loss = (timers['lcls_end'] - timers['lcls_start']) * 1000
    #             t_total = (timers['lcls_end'] - timers['start']) * 1000
    #             print(f"\n[Baseline Time] Prep: {t_prep:.1f}ms | ConvLoss: {t_loss:.1f}ms | Total: {t_total:.1f}ms")

    #     # ====================================================================
    #     # 2. ROI Head Loss (ROI 模式)
    #     # ====================================================================
    #     elif self.cls_mode == 'roicls':
    #         # 1. 全量负样本
    #         neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
    #         sampled_neg_inds = neg_inds 
            
    #         train_inds = torch.cat([pos_inds, sampled_neg_inds])
            
    #         # 2. 准备训练用的 Box
    #         train_labels = flatten_labels[train_inds]
    #         train_bbox_preds = flatten_bbox_preds[train_inds]
    #         train_angle_preds = flatten_angle_preds[train_inds]
    #         train_points = flatten_points[train_inds]
    #         train_bid_targets = flatten_bid_targets[train_inds]
            
    #         decoded_angle = self.angle_coder.decode(train_angle_preds, keepdim=True)
    #         if len(self.square_cls) > 0:
    #             for c in self.square_cls:
    #                 decoded_angle[train_labels == c] = 0
            
    #         train_rbox_preds = torch.cat((train_points + train_bbox_preds[:, 2:], 
    #                                       train_bbox_preds[:, :2] * 2,
    #                                       decoded_angle), -1)
            
    #         # --- [DEBUG START] ---
    #         if self.debug_roi_input:
    #             if not hasattr(self, '_debug_logged'):
    #                 print(f"\n[DEBUG MODE] Using Fake GT ROIs (stride size) for ALL samples.")
    #                 self._debug_logged = True
                
    #             d_cx = train_points[:, 0]
    #             d_cy = train_points[:, 1]
    #             d_w = torch.ones_like(d_cx) * self.strides[0]
    #             d_h = torch.ones_like(d_cy) * self.strides[0]
    #             d_t = torch.zeros_like(d_cx)
                
    #             debug_rboxes = torch.stack([d_cx, d_cy, d_w, d_h, d_t], dim=-1)
    #             batch_inds = train_bid_targets[:, 0].float().unsqueeze(1)
    #             rois = torch.cat([batch_inds, debug_rboxes], dim=1)
    #         else:
    #             batch_inds = train_bid_targets[:, 0].float().unsqueeze(1)
    #             rois = torch.cat([batch_inds, train_rbox_preds], dim=1)
    #         # --- [DEBUG END] ---
            
    #         tick('roi_construct_done') # --- ROI 构造耗时 ---

    #         # 3. 运行 ROI Cls
    #         current_feat = rois_features[0] 
            
    #         # [关键计时] RoIAlign
    #         roi_feats = self.roi_extractor(current_feat, rois)
    #         tick('roi_align_done') 
            
    #         # [关键计时] AvgPool + Linear
    #         roi_feats_pooled = F.adaptive_avg_pool2d(roi_feats, (1, 1))
    #         roi_logits = self.roi_cls_layer(roi_feats_pooled.view(roi_feats_pooled.size(0), -1))
    #         tick('linear_done') 
            
    #         # 计算 Loss
    #         total_samples = train_labels.size(0)
    #         avg_factor = max(total_samples, 1.0)
    #         loss_dict['loss_roicls'] = self.loss_roicls(
    #             roi_logits, train_labels, avg_factor=avg_factor)
    #         tick('all_done')

    #         # [打印 ROI 耗时]
    #         if torch.rand(1).item() < 0.1:
    #             t_prep = (timers['prep_done'] - timers['start']) * 1000
    #             t_construct = (timers['roi_construct_done'] - timers['prep_done']) * 1000
    #             t_align = (timers['roi_align_done'] - timers['roi_construct_done']) * 1000
    #             t_linear = (timers['linear_done'] - timers['roi_align_done']) * 1000
    #             t_total = (timers['all_done'] - timers['start']) * 1000
    #             print(f"\n[ROI Time] Prep:{t_prep:.1f}ms | Construct:{t_construct:.1f}ms | Align:{t_align:.1f}ms | Linear:{t_linear:.1f}ms | Total:{t_total:.1f}ms | Count:{rois.shape[0]}")

    #     # ====================================================================
    #     # 3. 几何/辅助 Loss (仅正样本)
    #     # ====================================================================
    #     if len(pos_inds) > 0:
    #         pos_bbox_preds = flatten_bbox_preds[pos_inds]
    #         pos_angle_preds = flatten_angle_preds[pos_inds]
    #         pos_bbox_targets = flatten_bbox_targets[pos_inds]
    #         pos_bid_targets = flatten_bid_targets[pos_inds]
    #         pos_points = flatten_points[pos_inds]
    #         pos_labels = flatten_labels[pos_inds]
            
    #         pos_decoded_angle = self.angle_coder.decode(pos_angle_preds, keepdim=True)
    #         for c in self.square_cls:
    #             pos_decoded_angle[pos_labels == c] = 0
            
    #         pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
    #         pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
    #                                     pos_bbox_preds[:, :2] * 2,
    #                                     pos_decoded_angle), -1)
            
    #         if self.cls_mode == 'cls':
    #             pos_scores = flatten_cls_scores[pos_inds].sigmoid()
    #             pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]
    #         else:
    #             pos_roi_logits_subset = roi_logits[:num_pos]
    #             pos_scores = pos_roi_logits_subset.sigmoid()
    #             pos_scores = torch.gather(pos_scores, 1, pos_labels[:, None])[:, 0]

    #         loss_dict['loss_size'] = self.loss_size(
    #             pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0], batch_img_metas)
    #         loss_dict['loss_angle'] = self.loss_angle(
    #             pos_rbox_preds, pos_scores, pos_labels, pos_bid_targets[:, 0])
            
    #         cos_r, sin_r = torch.cos(pos_decoded_angle), torch.sin(pos_decoded_angle)
    #         R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    #         pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))
            
    #         bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
    #         bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
    #         ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
    #         ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
    #         ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(0, idx, pos_labels, 'amin', include_self=False)
            
    #         ins_gaus_preds = pos_gaus_preds.new_zeros(*bid.shape, 4).index_reduce_(0, idx, pos_gaus_preds.view(-1, 4), 'mean', include_self=False).view(-1, 2, 2)
    #         ins_rbox_targets = pos_rbox_targets.new_zeros(*bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(0, idx, pos_rbox_targets, 'mean', include_self=False)
    #         ori_mu_all = ins_rbox_targets[:, 0:2]

    #         if self.geo_mode == 'voronoi':
    #             loss_geo = zero_tensor.clone()
    #             for batch_id in range(len(batch_gt_instances)):
    #                 group_mask = (ins_batch == batch_id) & (ins_bids != 0)
    #                 mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
    #                 if len(mu) >= 1:
    #                     pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
    #                     neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
    #                     if 'override' in self.voronoi_thres.keys():
    #                         for item in self.voronoi_thres['override']:
    #                             for cls in item[0]:
    #                                 pos_thres[cls] = item[1][0]
    #                                 neg_thres[cls] = item[1][1]
    #                     loss_geo += self.loss_voronoi(
    #                         (mu, sigma.bmm(sigma)), label, self.images[batch_id],
    #                         pos_thres, neg_thres)
    #             loss_dict['loss_bbox_vor'] = loss_geo / len(batch_gt_instances)
                
    #         elif self.geo_mode == 'ourwater':
    #             loss_geo = zero_tensor.clone()
    #             for batch_id in range(len(batch_gt_instances)):
    #                 group_mask = (ins_batch == batch_id) & (ins_bids != 0)
    #                 mu, sigma, label = ori_mu_all[group_mask], ins_gaus_preds[group_mask], ins_labels[group_mask]
    #                 if len(mu) >= 1:
    #                     pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
    #                     neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
    #                     if 'override' in self.voronoi_thres.keys():
    #                         for item in self.voronoi_thres['override']:
    #                             for cls in item[0]:
    #                                 pos_thres[cls] = item[1][0]
    #                                 neg_thres[cls] = item[1][1]
    #                     loss_geo += self.loss_ourwater(
    #                         (mu, sigma.bmm(sigma)), label, self.images[batch_id],
    #                         pos_thres, neg_thres)
    #             loss_dict['loss_ourwater'] = loss_geo / len(batch_gt_instances)

    #     return loss_dict
    
    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = False) -> InstanceList:
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, 
            batch_data_samples=batch_data_samples, 
            rescale=rescale,
            x=x)
        return predictions
    
    def predict_by_feat(self, cls_scores, bbox_preds, angle_preds, batch_data_samples=None, cfg=None, rescale=False, with_nms=True, x=None, **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_data_samples)):
            img_meta = batch_data_samples[img_id].metainfo
            gt_instances = batch_data_samples[img_id].gt_instances
            gt_ignore = batch_data_samples[img_id].ignored_instances
            data_sample = (gt_instances, gt_ignore, img_meta)

            if self.cls_mode == 'roicls':
                current_feat = x[0][img_id:img_id+1]
                bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
                angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
                
                results = self._predict_single_roi(
                    bbox_pred_list, angle_pred_list, current_feat, 
                    mlvl_priors, data_sample, cfg, rescale, with_nms)
            else:
                cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
                bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
                angle_pred_list = select_single_mlvl(angle_preds, img_id, detach=True)
                
                if self.training or self.pseudo_generator:
                    predict_func = self._predict_by_feat_single_pseudo
                else:
                    predict_func = self._predict_by_feat_single
                
                results = predict_func(
                    cls_score_list, bbox_pred_list, angle_pred_list, 
                    mlvl_priors, data_sample, cfg, rescale, with_nms)
            
            result_list.append(results)
        return result_list

    def _predict_single_roi(self, bbox_pred_list, angle_pred_list, current_feat, mlvl_priors, data_sample, cfg, rescale, with_nms):
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_meta = data_sample[2]
        
        all_bbox_preds = []
        all_angle_preds = []
        all_priors = []
        
        for (bbox_pred, angle_pred, priors) in zip(bbox_pred_list, angle_pred_list, mlvl_priors):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            all_bbox_preds.append(bbox_pred)
            all_angle_preds.append(angle_pred)
            all_priors.append(priors)
            
        all_bbox_preds = torch.cat(all_bbox_preds)
        all_angle_preds = torch.cat(all_angle_preds)
        all_priors = torch.cat(all_priors)
        
        decoded_angle = self.angle_coder.decode(all_angle_preds, keepdim=True)
        rboxes = torch.cat((all_priors + all_bbox_preds[:, 2:], 
                            all_bbox_preds[:, :2] * 2, 
                            decoded_angle), -1)
        
        batch_inds = rboxes.new_zeros(rboxes.size(0), 1)
        rois = torch.cat([batch_inds, rboxes], dim=1)
        
        roi_feats = self.roi_extractor(current_feat, rois)
        # [修改点 3] 推理时也要 AvgPool
        roi_feats_pooled = F.adaptive_avg_pool2d(roi_feats, (1, 1))
        roi_logits = self.roi_cls_layer(roi_feats_pooled.view(roi_feats_pooled.size(0), -1))
        scores = roi_logits.sigmoid()
        
        score_thr = cfg.get('score_thr', 0.05)
        max_scores, labels = scores.max(dim=1)
        valid_mask = max_scores > score_thr
        
        results = InstanceData()
        results.bboxes = RotatedBoxes(rboxes[valid_mask])
        results.scores = max_scores[valid_mask] 
        results.labels = labels[valid_mask]
        
        results = self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)
        return results

    def _predict_by_feat_single(self, cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, data_sample, cfg, rescale=False, with_nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_meta = data_sample[2]
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bbox_preds, mlvl_valid_priors, mlvl_scores, mlvl_labels = [], [], [], []
        for level_idx, (cls_score, bbox_pred, angle_pred, priors) in enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else cls_score.softmax(-1)[:, :-1]
            score_thr = cfg.get('score_thr', 0)
            results = filter_scores_and_topk(scores, score_thr, nms_pre, dict(bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']
            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)
            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
        scores = torch.cat(mlvl_scores)
        labels = torch.cat(mlvl_labels)
        bboxes = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        for id in self.post_process.keys():
            bboxes[labels == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[labels == id, -1] = 0
        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = scores
        results.labels = labels
        results = self._bbox_post_process(results=results, cfg=cfg, rescale=rescale, with_nms=with_nms, img_meta=img_meta)
        return results

    def _predict_by_feat_single_pseudo(self, cls_score_list, bbox_pred_list, angle_pred_list, mlvl_priors, data_sample, cfg, rescale=False, with_nms=True):
        gt_instances = data_sample[0]
        img_meta = data_sample[2]
        scale_factor = [1, 1] if self.training else img_meta['scale_factor']
        gt_bboxes = gt_instances.bboxes.tensor
        gt_labels = gt_instances.labels
        gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).long()
        cls_score, bbox_pred, angle_pred = cls_score_list[0], bbox_pred_list[0], angle_pred_list[0]
        H, W = cls_score.shape[1:3]
        gt_valid_mask = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
        gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
        gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)[gt_idx]
        angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)[gt_idx]
        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2, decoded_angle), -1)
        bboxes[~gt_valid_mask, 2:] = 0
        bboxes[:, 2:4] = bboxes[:, 2:4] / scale_factor[1]
        for id in self.post_process.keys():
            bboxes[gt_labels == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[gt_labels == id, -1] = 0
        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes.detach())
        results.scores = torch.ones_like(cls_score[:, 0])
        results.labels = gt_labels
        return results

    def get_targets(
        self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        labels_list, bbox_targets_list, bid_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0)
            for bid_targets in bid_targets_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bid_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            bid_targets = torch.cat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bid_targets.append(bid_targets)
            
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_bid_targets)

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        
        if hasattr(gt_instances, 'bids'):
            gt_bids = gt_instances.bids
        else:
            gt_bids = points.new_zeros((num_gts, 4)) 

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bids.new_zeros((num_points, 4))

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.tensor

        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        
        offset = points - gt_ctr
        w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

        center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            
            stride_tensor = torch.cat([points.new_full((n,), self.strides[i]) for i, n in enumerate(num_points_per_lvl)])
            limit = self.center_sample_radius * stride_tensor[:, None] * center_r[..., 0]
            inside_gt_bbox_mask = (abs(offset) < limit.unsqueeze(-1)).all(dim=-1)

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes 
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        bbox_targets = torch.cat((bbox_targets, angle_targets), -1)

        return labels, bbox_targets, bid_targets