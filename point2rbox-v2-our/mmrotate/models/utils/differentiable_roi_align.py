# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DifferentiableRoIAlignRotated(nn.Module):
#     """
#     纯 PyTorch 实现的可导旋转 RoI Align (Differentiable for ROIs)。
#     支持梯度回传至 ROI 坐标 (x, y, w, h, theta)。
#     完全对齐 mmcv.ops.RoIAlignRotated (aligned=True) 的逻辑。
#     """
#     def __init__(self, output_size, spatial_scale, sampling_ratio=2, clockwise=True):
#         """
#         Args:
#             output_size (tuple): (h, w)
#             spatial_scale (float): 特征图缩放比例 (如 1/8)
#             sampling_ratio (int): 每个 Bin 采样的点数 (0 表示自动，通常设为 2)。
#             clockwise (bool): 是否顺时针旋转，MMRotate 默认为 True。
#         """
#         super(DifferentiableRoIAlignRotated, self).__init__()
#         # 兼容处理 output_size 为 int 的情况
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             self.output_size = output_size
            
#         self.spatial_scale = spatial_scale
#         self.sampling_ratio = sampling_ratio if sampling_ratio > 0 else 1
#         self.clockwise = clockwise

#     def forward(self, features, rois):
#         """
#         Args:
#             features: (N, C, H, W)
#             rois: (K, 6) [batch_ind, x, y, w, h, theta]
#         """
#         num_rois = rois.shape[0]
#         batch_size, channels, height, width = features.shape
#         out_h, out_w = self.output_size
        
#         # 过采样网格
#         grid_h = out_h * self.sampling_ratio
#         grid_w = out_w * self.sampling_ratio

#         # 1. 解析 ROIs
#         batch_inds = rois[:, 0].long()
        
#         # ================= [核心修复点 START] =================
#         # 错误写法: rois_feat = rois[:, 1:] * self.spatial_scale (导致 theta 被缩放)
#         # 正确写法: 分别处理坐标/尺寸和角度
        
#         # 坐标和尺寸 (x, y, w, h) 需要映射到特征图尺度 -> 乘以 spatial_scale
#         cx = rois[:, 1] * self.spatial_scale
#         cy = rois[:, 2] * self.spatial_scale
#         w  = rois[:, 3] * self.spatial_scale
#         h  = rois[:, 4] * self.spatial_scale
        
#         # 角度 (theta) 是几何属性，不受尺度缩放影响 -> 保持原值！
#         theta = rois[:, 5] 
#         # ================= [核心修复点 END] ===================

#         # 2. 构建采样网格 (Affine Transformation)
#         # 生成 Bin Center，范围 [-0.5, 0.5] * w/h
#         # 这里的逻辑是为了模拟 "在 box 内部均匀撒点"
#         _y = (torch.arange(grid_h, dtype=rois.dtype, device=rois.device) + 0.5) / grid_h - 0.5
#         _x = (torch.arange(grid_w, dtype=rois.dtype, device=rois.device) + 0.5) / grid_w - 0.5
        
#         _y, _x = torch.meshgrid(_y, _x, indexing='ij')
#         # (K, grid_h * grid_w)
#         grid_x = _x.reshape(-1).unsqueeze(0).expand(num_rois, -1)
#         grid_y = _y.reshape(-1).unsqueeze(0).expand(num_rois, -1)

#         # 旋转矩阵
#         if self.clockwise:
#             cos_t = torch.cos(theta)
#             sin_t = torch.sin(theta)
#         else:
#             cos_t = torch.cos(-theta)
#             sin_t = torch.sin(-theta)

#         # 仿射变换
#         # 缩放: grid (-0.5~0.5) * w
#         gx = grid_x * w.unsqueeze(1) 
#         gy = grid_y * h.unsqueeze(1)

#         # 旋转 + 平移 (得到特征图上的绝对坐标 real_x, real_y)
#         # x' = x*cos - y*sin + cx
#         x_sample = gx * cos_t.unsqueeze(1) - gy * sin_t.unsqueeze(1) + cx.unsqueeze(1)
#         y_sample = gx * sin_t.unsqueeze(1) + gy * cos_t.unsqueeze(1) + cy.unsqueeze(1)

#         # 3. 归一化到 [-1, 1] 供 grid_sample 使用
#         # 对应 mmcv aligned=True (align_corners=False)
#         # 公式: coordinate / width * 2 - 1
#         x_grid = (x_sample / width) * 2.0 - 1.0
#         y_grid = (y_sample / height) * 2.0 - 1.0
        
#         # (K, grid_h, grid_w, 2)
#         grid = torch.stack([x_grid, y_grid], dim=2).view(num_rois, grid_h, grid_w, 2)

#         # 4. 执行采样 (Batch Loop)
#         # 初始化输出: (K, C, grid_h, grid_w)
#         output_raw = torch.zeros(num_rois, channels, grid_h, grid_w, 
#                                  dtype=features.dtype, device=features.device)
        
#         # 针对每个 batch index 进行循环采样
#         # 因为 grid_sample 需要 Input 和 Grid 的 batch 维度一致
#         for b_idx in range(batch_size):
#             mask = (batch_inds == b_idx)
#             if not mask.any(): continue
            
#             batch_grid = grid[mask] 
#             # 取出当前图片特征 (1, C, H, W)
#             batch_feat = features[b_idx:b_idx+1]
            
#             # 扩展 input 以匹配当前 batch 中 RoI 的数量
#             batch_feat_expanded = batch_feat.expand(mask.sum(), -1, -1, -1)
            
#             # 采样 (F.grid_sample 支持对 grid 求导)
#             batch_out = F.grid_sample(batch_feat_expanded, batch_grid, 
#                                       mode='bilinear', padding_mode='zeros', align_corners=False)
            
#             output_raw[mask] = batch_out

#         # 5. AvgPool 下采样 (将过采样的特征图池化回目标尺寸)
#         if self.sampling_ratio > 1:
#             output = F.avg_pool2d(output_raw, kernel_size=self.sampling_ratio, stride=self.sampling_ratio)
#         else:
#             output = output_raw

#         return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableRoIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio=2, clockwise=True, chunk_size=1000):
        super(DifferentiableRoIAlignRotated, self).__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio if sampling_ratio > 0 else 1
        self.clockwise = clockwise
        self.chunk_size = chunk_size # 内部自动分批大小

    def forward(self, features, rois):
        # 如果 RoI 数量太少，直接走原逻辑
        if rois.shape[0] <= self.chunk_size:
            return self._do_roi_align(features, rois)
        
        # 如果 RoI 数量太多，循环分批处理
        results = []
        for i in range(0, rois.shape[0], self.chunk_size):
            chunk_rois = rois[i : i + self.chunk_size]
            # 每一批单独计算，节省中间变量显存
            chunk_out = self._do_roi_align(features, chunk_rois)
            results.append(chunk_out)
        
        return torch.cat(results, dim=0)

    def _do_roi_align(self, features, rois):
        """原有的核心计算逻辑"""
        num_rois = rois.shape[0]
        batch_size, channels, height, width = features.shape
        out_h, out_w = self.output_size
        grid_h, grid_w = out_h * self.sampling_ratio, out_w * self.sampling_ratio

        # 1. 解析 ROIs (角度修正版)
        batch_inds = rois[:, 0].long()
        cx = rois[:, 1] * self.spatial_scale
        cy = rois[:, 2] * self.spatial_scale
        w  = rois[:, 3] * self.spatial_scale
        h  = rois[:, 4] * self.spatial_scale
        theta = rois[:, 5]

        # 2. 构建采样网格
        _y = (torch.arange(grid_h, dtype=rois.dtype, device=rois.device) + 0.5) / grid_h - 0.5
        _x = (torch.arange(grid_w, dtype=rois.dtype, device=rois.device) + 0.5) / grid_w - 0.5
        _y, _x = torch.meshgrid(_y, _x, indexing='ij')
        grid_x = _x.reshape(-1).unsqueeze(0).expand(num_rois, -1)
        grid_y = _y.reshape(-1).unsqueeze(0).expand(num_rois, -1)

        cos_t = torch.cos(theta) if self.clockwise else torch.cos(-theta)
        sin_t = torch.sin(theta) if self.clockwise else torch.sin(-theta)

        gx, gy = grid_x * w.unsqueeze(1), grid_y * h.unsqueeze(1)
        x_sample = gx * cos_t.unsqueeze(1) - gy * sin_t.unsqueeze(1) + cx.unsqueeze(1)
        y_sample = gx * sin_t.unsqueeze(1) + gy * cos_t.unsqueeze(1) + cy.unsqueeze(1)

        # 3. 归一化并构造 Grid
        x_grid, y_grid = (x_sample / width) * 2.0 - 1.0, (y_sample / height) * 2.0 - 1.0
        grid = torch.stack([x_grid, y_grid], dim=2).view(num_rois, grid_h, grid_w, 2)

        # 4. 采样
        output_raw = torch.zeros(num_rois, channels, grid_h, grid_w, dtype=features.dtype, device=features.device)
        unique_batches = batch_inds.unique()
        for b_idx in unique_batches:
            mask = (batch_inds == b_idx)
            batch_feat = features[b_idx : b_idx + 1].expand(mask.sum(), -1, -1, -1)
            output_raw[mask] = F.grid_sample(batch_feat, grid[mask], mode='bilinear', padding_mode='zeros', align_corners=False)

        # 5. 池化
        return F.avg_pool2d(output_raw, self.sampling_ratio) if self.sampling_ratio > 1 else output_raw