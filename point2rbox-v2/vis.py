# # import torch
# # import numpy as np
# # import cv2
# # import os
# # import random
# # import matplotlib.pyplot as plt
# # import tqdm
# # from shapely.geometry import Polygon
# # import torch.nn.functional as F
# # from mmdet.apis import init_detector, inference_detector
# # import sys

# # # ==============================================================================
# # # 1. 全局配置
# # # ==============================================================================
# # CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-dota.py'
# # CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2/work_dirs/1/std/e2e/epoch_12.pth'
# # DATA_ROOT = '/mnt/data/xiekaikai/split_ss_dota/trainval'
# # SAVE_DIR = 'vis/dota'
# # DEVICE = 'cuda:1'
# # NUM_VIS = 100

# # os.makedirs(SAVE_DIR, exist_ok=True)

# # CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
# #            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
# #            'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
# #            'harbor', 'swimming-pool', 'helicopter')

# # BEST_CFG = dict(default_sigma=4096, down_sample=2, pos_thres=0.994, neg_thres=0.005)

# # # ==============================================================================
# # # 2. 核心算法逻辑
# # # ==============================================================================

# # def find_max_iou_box_vectorized(mask_pts, center_pt, device='cuda'):
# #     if len(mask_pts) < 3: return None
# #     pts = torch.tensor(mask_pts, dtype=torch.float32, device=device)
# #     center = torch.tensor(center_pt, dtype=torch.float32, device=device)
# #     pts_centered = pts - center
# #     N = pts.shape[0]
# #     angles = torch.linspace(0, torch.pi, 90, device=device)
# #     cos_a = torch.cos(angles); sin_a = torch.sin(angles)
# #     R = torch.stack([torch.stack([cos_a, -sin_a], dim=-1), torch.stack([sin_a, cos_a], dim=-1)], -2)
# #     pts_rot = torch.matmul(R, pts_centered.T).transpose(1, 2).abs()
# #     quantiles = torch.linspace(0.5, 1.0, 31, device=device)
# #     sorted_x, _ = torch.sort(pts_rot[..., 0], dim=1)
# #     sorted_y, _ = torch.sort(pts_rot[..., 1], dim=1)
# #     indices = ((quantiles * (N - 1)).long()).clamp(0, N-1)
# #     W_h = sorted_x[:, indices]; H_h = sorted_y[:, indices]
# #     Box_A = (W_h * 2) * (H_h * 2)
# #     in_x = pts_rot[..., 0].unsqueeze(-1) <= W_h.unsqueeze(1)
# #     in_y = pts_rot[..., 1].unsqueeze(-1) <= H_h.unsqueeze(1)
# #     inter = (in_x & in_y).sum(dim=1).float()
# #     iou_mat = inter / (N + Box_A - inter + 1e-6)
# #     val, idx = torch.max(iou_mat.view(-1), 0)
# #     b_a_i, b_q_i = idx // len(quantiles), idx % len(quantiles)
# #     return ((center_pt[0], center_pt[1]), (W_h[b_a_i, b_q_i].item()*2, H_h[b_a_i, b_q_i].item()*2), np.degrees(angles[b_a_i].item()))

# # def get_watershed_wh_row3(mask_pts, center_pt, angle_rad):
# #     if len(mask_pts) == 0: return 0.0, 0.0
# #     R_inv = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)], [np.sin(-angle_rad), np.cos(-angle_rad)]])
# #     pts_rot = np.dot(mask_pts - center_pt, R_inv.T)
# #     return max(2 * np.max(np.abs(pts_rot[:, 0])), 1.0), max(2 * np.max(np.abs(pts_rot[:, 1])), 1.0)

# # def compute_iou_shapely(poly1, poly2):
# #     try:
# #         p1, p2 = Polygon(poly1), Polygon(poly2)
# #         if not p1.is_valid: p1 = p1.convex_hull
# #         if not p2.is_valid: p2 = p2.convex_hull
# #         inter_area = p1.intersection(p2).area
# #         return inter_area / (p1.area + p2.area - inter_area + 1e-6)
# #     except: return 0.0

# # def draw_rbox(ax, poly, color, linestyle='-'):
# #     """支持输入 4角点 或 ((cx,cy),(w,h),a) 格式"""
# #     if isinstance(poly, (tuple, list)) and len(poly) == 3:
# #         pts = cv2.boxPoints(poly)
# #     else:
# #         pts = poly
# #     pts = np.vstack([pts, pts[0]])
# #     ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, linestyle=linestyle)

# # # ==============================================================================
# # # 3. 主可视化程序
# # # ==============================================================================

# # def run_visualization():
# #     print(f"Initializing Model...")
# #     model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    
# #     img_dir = os.path.join(DATA_ROOT, 'images')
# #     lbl_dir = os.path.join(DATA_ROOT, 'labelTxt')
# #     all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
# #     selected_imgs = random.sample(all_imgs, min(NUM_VIS, len(all_imgs)))

# #     for img_name in tqdm.tqdm(selected_imgs):
# #         file_id = os.path.splitext(img_name)[0]
# #         img_path = os.path.join(img_dir, img_name)
# #         lbl_path = os.path.join(lbl_dir, file_id + '.txt')
# #         if not os.path.exists(lbl_path): continue

# #         img = cv2.imread(img_path)
# #         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         H, W = img.shape[:2]
        
# #         gt_points, gt_polys = [], []
# #         with open(lbl_path, 'r') as f:
# #             for line in f.readlines():
# #                 parts = line.strip().split()
# #                 if len(parts) < 9: continue
# #                 poly = np.array(list(map(float, parts[:8]))).reshape(4, 2)
# #                 gt_polys.append(poly)
# #                 gt_points.append(np.mean(poly, axis=0))
        
# #         if not gt_points: continue

# #         # 生成 Standard Mask
# #         D = BEST_CFG['down_sample']
# #         # [优化] 修复 UserWarning: 列表先转 numpy
# #         mu = torch.from_numpy(np.array(gt_points)).to(DEVICE).float()
# #         mm = (mu / D).round()
# #         h_m, w_m = H // D, W // D
# #         ax_x = torch.linspace(0, w_m-1, w_m, device=DEVICE)
# #         ax_y = torch.linspace(0, h_m-1, h_m, device=DEVICE)
# #         gx, gy = torch.meshgrid(ax_x, ax_y, indexing='xy')
# #         xy_grid = torch.stack([gx, gy], -1)
        
# #         vor = torch.zeros((len(gt_points), h_m, w_m), device=DEVICE)
# #         for j in range(len(gt_points)):
# #             d = (xy_grid - mm[j]).pow(2).sum(-1)
# #             vor[j] = torch.exp(-0.5 * d / (BEST_CFG['default_sigma']/(D**2)))
# #         val, vor_idx = torch.max(vor, 0)
        
# #         if D > 1:
# #             vor_idx = F.interpolate(vor_idx[None,None].float(), (H,W), mode='nearest')[0,0].long()
# #             val = F.interpolate(val[None,None], (H,W), mode='bilinear', align_corners=True)[0,0]
        
# #         markers = (vor_idx + 1).cpu().numpy().astype(np.int32)
# #         markers[val.cpu().numpy() < BEST_CFG['pos_thres']] = 0
# #         markers[val.cpu().numpy() < BEST_CFG['neg_thres']] = len(gt_points) + 1
# #         cv2.watershed(cv2.medianBlur(img, 3), markers)
        
# #         # [修正错误点] 模型预测获取角度
# #         pred_res = inference_detector(model, img)
# #         bboxes_obj = pred_res.pred_instances.bboxes
# #         # 兼容性处理：检查 bboxes_obj 是否有 .tensor 属性
# #         if hasattr(bboxes_obj, 'tensor'):
# #             pred_bboxes = bboxes_obj.tensor
# #         else:
# #             pred_bboxes = bboxes_obj # 它已经是一个 Tensor
            
# #         dists = torch.cdist(mu, pred_bboxes[:, :2])
# #         match_idx = dists.min(dim=1)[1]
# #         pred_angles = pred_bboxes[match_idx, 4].cpu().numpy()

# #         row_results = {1: [], 2: [], 3: []}
# #         for idx in range(len(gt_points)):
# #             pts_y, pts_x = np.where(markers == idx + 1)
# #             if len(pts_x) < 3: continue
# #             mask_pts = np.stack([pts_x, pts_y], axis=1)
# #             gt_c = gt_points[idx]
            
# #             r1_box = find_max_iou_box_vectorized(mask_pts, gt_c, DEVICE)
# #             r2_box = cv2.minAreaRect(mask_pts.astype(np.float32))
# #             w3, h3 = get_watershed_wh_row3(mask_pts, gt_c, pred_angles[idx])
# #             r3_box = ((gt_c[0], gt_c[1]), (w3, h3), np.degrees(pred_angles[idx]))
            
# #             row_results[1].append((r1_box, compute_iou_shapely(cv2.boxPoints(r1_box), gt_polys[idx])))
# #             row_results[2].append((r2_box, compute_iou_shapely(cv2.boxPoints(r2_box), gt_polys[idx])))
# #             row_results[3].append((r3_box, compute_iou_shapely(cv2.boxPoints(r3_box), gt_polys[idx])))

# #         # 开始绘图
# #         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# #         plt.subplots_adjust(top=0.9)
        
# #         ax_a = axes[0, 0]
# #         ax_a.imshow(img_rgb)
# #         vis_markers = markers.copy().astype(float)
# #         vis_markers[(vis_markers == 0) | (vis_markers > len(gt_points))] = np.nan
# #         ax_a.imshow(vis_markers, cmap='jet', alpha=0.4)
# #         ax_a.set_title("Plot A: Watershed Basins")

# #         titles = {1: "B: Row 1 (Max IoU Search)", 2: "C: Row 2 (MinAreaRect Outer)", 3: "D: Row 3 (Pred Angle Guided)"}
# #         for row_id, ax in zip([1, 2, 3], [axes[0, 1], axes[1, 0], axes[1, 1]]):
# #             ax.imshow(img_rgb)
# #             ious = []
# #             for j in range(len(row_results[row_id])):
# #                 p_box, iou = row_results[row_id][j]
# #                 draw_rbox(ax, gt_polys[j], 'green')
# #                 draw_rbox(ax, p_box, 'red')
# #                 ax.text(gt_points[j][0], gt_points[j][1], f"{iou:.2f}", color='yellow', fontsize=7, fontweight='bold')
# #                 ious.append(iou)
# #             ax.set_title(f"{titles[row_id]}\nMean IoU: {np.mean(ious):.4f}" if ious else titles[row_id])

# #         plt.suptitle(f"Image ID: {file_id}\nGreen: GT, Red: Estimated Box", fontsize=16)
# #         plt.savefig(os.path.join(SAVE_DIR, f"{file_id}_comparison.png"), bbox_inches='tight', dpi=150)
# #         plt.close()

# # if __name__ == '__main__':
# #     run_visualization()

# import torch
# import numpy as np
# import cv2
# import os
# import random
# import matplotlib.pyplot as plt
# import tqdm
# from shapely.geometry import Polygon
# import torch.nn.functional as F
# from mmdet.apis import init_detector, inference_detector
# import sys

# # ==============================================================================
# # 1. 全局配置
# # ==============================================================================
# CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-dota.py'
# CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2/work_dirs/1/std/e2e/epoch_12.pth'
# DATA_ROOT = '/mnt/data/xiekaikai/split_ss_dota/trainval'
# SAVE_DIR = 'vis/dota1'
# DEVICE = 'cuda:1'
# NUM_VIS = 20 # 随机抽取20张

# os.makedirs(SAVE_DIR, exist_ok=True)

# CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#            'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
#            'harbor', 'swimming-pool', 'helicopter')

# # 这里的参数应该与之前测试 Row 1/2/3 时保持一致，即 Standard Mode 的参数
# BEST_CFG = dict(default_sigma=4096, down_sample=2, pos_thres=0.994, neg_thres=0.005)

# # ==============================================================================
# # 2. 核心算法逻辑
# # ==============================================================================

# def find_max_iou_box_vectorized(mask_pts, center_pt, device='cuda'):
#     """Row 1: 暴力搜索与 Mask IoU 最大的框 (修正版)"""
#     if len(mask_pts) < 3: return None
    
#     # 转换为 Tensor
#     pts = torch.from_numpy(mask_pts).float().to(device)
#     center = torch.tensor(center_pt, dtype=torch.float32, device=device)
#     pts_centered = pts - center
#     N = pts.shape[0]

#     # 1. 角度搜索：0 ~ 180 度 (pi)
#     num_angles = 180 
#     angles = torch.linspace(0, torch.pi, num_angles, device=device)
    
#     cos_a = torch.cos(angles); sin_a = torch.sin(angles)
#     # 构建逆时针旋转矩阵
#     R = torch.stack([
#         torch.stack([cos_a, -sin_a], dim=-1),
#         torch.stack([sin_a, cos_a], dim=-1)
#     ], dim=-2)

#     # 旋转点集: (A, N, 2)
#     pts_rot = torch.matmul(R, pts_centered.T).transpose(1, 2).abs()

#     # 2. 尺度搜索
#     quantiles = torch.linspace(0.5, 1.0, 51, device=device)
#     sorted_x, _ = torch.sort(pts_rot[..., 0], dim=1)
#     sorted_y, _ = torch.sort(pts_rot[..., 1], dim=1)
    
#     indices = ((quantiles * (N - 1)).long()).clamp(0, N-1)
#     W_half = sorted_x[:, indices] # (A, Q)
#     H_half = sorted_y[:, indices] # (A, Q)
#     Box_Area = (W_half * 2) * (H_half * 2)

#     # 3. 计算 IoU
#     in_x = pts_rot[..., 0].unsqueeze(-1) <= W_half.unsqueeze(1)
#     in_y = pts_rot[..., 1].unsqueeze(-1) <= H_half.unsqueeze(1)
#     inter = (in_x & in_y).sum(dim=1).float()
    
#     iou_mat = inter / (N + Box_Area - inter + 1e-6)
    
#     # 4. 取最大值
#     val, idx = torch.max(iou_mat.view(-1), 0)
#     b_a_i, b_q_i = idx // len(quantiles), idx % len(quantiles)
    
#     # 5. 提取参数
#     angle_rad = angles[b_a_i].item()
#     w = W_half[b_a_i, b_q_i].item() * 2
#     h = H_half[b_a_i, b_q_i].item() * 2
    
#     # [关键修正] 返回 -degrees(angle)，符合 OpenCV 定义
#     return ((center_pt[0], center_pt[1]), (w, h), -np.degrees(angle_rad))

# def get_watershed_wh_row3(mask_pts, center_pt, angle_rad):
#     """Row 3: 按预测角度投影算宽高"""
#     if len(mask_pts) == 0: return 0.0, 0.0
#     # 投影到预测轴上
#     R_inv = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)], [np.sin(-angle_rad), np.cos(-angle_rad)]])
#     pts_rot = np.dot(mask_pts - center_pt, R_inv.T)
#     # 取最大跨度
#     return max(2 * np.max(np.abs(pts_rot[:, 0])), 1.0), max(2 * np.max(np.abs(pts_rot[:, 1])), 1.0)

# def compute_iou_shapely(poly1, poly2):
#     try:
#         p1, p2 = Polygon(poly1), Polygon(poly2)
#         if not p1.is_valid: p1 = p1.convex_hull
#         if not p2.is_valid: p2 = p2.convex_hull
#         inter_area = p1.intersection(p2).area
#         return inter_area / (p1.area + p2.area - inter_area + 1e-6)
#     except: return 0.0

# def draw_rbox(ax, poly, color, linestyle='-'):
#     """统一绘图接口"""
#     if isinstance(poly, (tuple, list)) and len(poly) == 3:
#         # ((cx,cy), (w,h), a)
#         pts = cv2.boxPoints(poly)
#     else:
#         # np.array [[x,y], ...]
#         pts = poly
#     pts = np.vstack([pts, pts[0]]) # 闭合
#     ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, linestyle=linestyle)

# # ==============================================================================
# # 3. 主可视化程序
# # ==============================================================================

# def run_visualization():
#     print(f"Initializing Model...")
#     # [关键修正] 避免 TED 权重加载错误，如果 strict=True 报错，可手动过滤
#     # 这里我们直接用 init_detector 加载，它通常会自动处理不匹配的键
#     try:
#         model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
#     except Exception as e:
#         print(f"Model init warning: {e}. Trying to load with strict=False equivalent...")
#         # 如果直接崩了，可以在这里手动加载 state_dict，但通常 init_detector 只是打印警告
#         pass
    
#     img_dir = os.path.join(DATA_ROOT, 'images')
#     lbl_dir = os.path.join(DATA_ROOT, 'labelTxt')
#     all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    
#     if len(all_imgs) == 0:
#         print("No images found!")
#         return
        
#     selected_imgs = random.sample(all_imgs, min(NUM_VIS, len(all_imgs)))

#     print("Start visualizing...")
#     for img_name in tqdm.tqdm(selected_imgs):
#         file_id = os.path.splitext(img_name)[0]
#         img_path = os.path.join(img_dir, img_name)
#         lbl_path = os.path.join(lbl_dir, file_id + '.txt')
#         if not os.path.exists(lbl_path): continue

#         # 1. 读取数据
#         img = cv2.imread(img_path)
#         if img is None: continue
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         H, W = img.shape[:2]
        
#         gt_points, gt_polys = [], []
#         try:
#             with open(lbl_path, 'r') as f:
#                 for line in f.readlines():
#                     parts = line.strip().split()
#                     if len(parts) < 9: continue
#                     poly = np.array(list(map(float, parts[:8]))).reshape(4, 2)
#                     gt_polys.append(poly)
#                     gt_points.append(np.mean(poly, axis=0))
#         except: continue
        
#         if not gt_points: continue

#         # 2. 生成 Standard Mask (各向同性种子)
#         D = BEST_CFG['down_sample']
        
#         # [优化] 使用 numpy array 转换，避免 UserWarning
#         gt_points_np = np.array(gt_points, dtype=np.float32)
#         mu = torch.from_numpy(gt_points_np).to(DEVICE)
        
#         mm = (mu / D).round()
#         h_m, w_m = H // D, W // D
        
#         ax_x = torch.linspace(0, w_m-1, w_m, device=DEVICE)
#         ax_y = torch.linspace(0, h_m-1, h_m, device=DEVICE)
#         gx, gy = torch.meshgrid(ax_x, ax_y, indexing='xy')
#         xy_grid = torch.stack([gx, gy], -1)
        
#         vor = torch.zeros((len(gt_points), h_m, w_m), device=DEVICE)
        
#         # 简单的 Standard Gaussian 生成
#         sigma_val = BEST_CFG['default_sigma'] / (D**2)
#         for j in range(len(gt_points)):
#             d = (xy_grid - mm[j]).pow(2).sum(-1)
#             vor[j] = torch.exp(-0.5 * d / sigma_val)
            
#         val, vor_idx = torch.max(vor, 0)
        
#         if D > 1:
#             vor_idx = F.interpolate(vor_idx[None,None].float(), (H,W), mode='nearest')[0,0].long()
#             val = F.interpolate(val[None,None], (H,W), mode='bilinear', align_corners=True)[0,0]
        
#         markers = (vor_idx + 1).cpu().numpy().astype(np.int32)
        
#         # 应用阈值
#         pos_mask = val.cpu().numpy() < BEST_CFG['pos_thres']
#         neg_mask = val.cpu().numpy() < BEST_CFG['neg_thres']
#         markers[pos_mask] = 0
#         markers[neg_mask] = len(gt_points) + 1
        
#         # 分水岭
#         cv2.watershed(cv2.medianBlur(img, 3), markers)
        
#         # 3. 模型预测获取角度 (Row 3 关键)
#         pred_res = inference_detector(model, img)
        
#         # [关键修正] 兼容不同版本的 mmrotate 输出结构
#         if hasattr(pred_res, 'pred_instances'):
#             bboxes_obj = pred_res.pred_instances.bboxes
#         else:
#             bboxes_obj = pred_res.bboxes # 旧版本可能是 list 或 tensor
            
#         if hasattr(bboxes_obj, 'tensor'):
#             pred_bboxes = bboxes_obj.tensor
#         elif isinstance(bboxes_obj, torch.Tensor):
#             pred_bboxes = bboxes_obj
#         else:
#             # 如果是其他格式，尝试转换
#             pred_bboxes = torch.tensor(bboxes_obj, device=DEVICE)
            
#         if len(pred_bboxes) == 0:
#             pred_angles = np.zeros(len(gt_points))
#         else:
#             dists = torch.cdist(mu, pred_bboxes[:, :2])
#             match_idx = dists.min(dim=1)[1]
#             pred_angles = pred_bboxes[match_idx, 4].cpu().numpy()

#         # 4. 计算三行数据
#         row_results = {1: [], 2: [], 3: []}
        
#         for idx in range(len(gt_points)):
#             # 提取 Mask 像素点
#             pts_y, pts_x = np.where(markers == idx + 1)
#             if len(pts_x) < 3: 
#                 # Mask 生成失败，用 0 填充
#                 row_results[1].append((None, 0.0))
#                 row_results[2].append((None, 0.0))
#                 row_results[3].append((None, 0.0))
#                 continue
                
#             mask_pts = np.stack([pts_x, pts_y], axis=1)
#             gt_c = gt_points[idx]
            
#             # Row 1: Max IoU 搜索
#             r1_box = find_max_iou_box_vectorized(mask_pts, gt_c, DEVICE)
#             iou1 = compute_iou_shapely(cv2.boxPoints(r1_box), gt_polys[idx]) if r1_box else 0.0
            
#             # Row 2: 最小外接矩形
#             r2_box = cv2.minAreaRect(mask_pts.astype(np.float32))
#             iou2 = compute_iou_shapely(cv2.boxPoints(r2_box), gt_polys[idx])
            
#             # Row 3: 预测角度投影
#             w3, h3 = get_watershed_wh_row3(mask_pts, gt_c, pred_angles[idx])
#             r3_box = ((gt_c[0], gt_c[1]), (w3, h3), np.degrees(pred_angles[idx]))
#             iou3 = compute_iou_shapely(cv2.boxPoints(r3_box), gt_polys[idx])
            
#             row_results[1].append((r1_box, iou1))
#             row_results[2].append((r2_box, iou2))
#             row_results[3].append((r3_box, iou3))

#         # 5. 绘图 (4 子图)
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         plt.subplots_adjust(top=0.9, hspace=0.25)
        
#         # 图 A: Watershed Basins
#         ax_a = axes[0, 0]
#         ax_a.imshow(img_rgb)
#         vis_markers = markers.copy().astype(float)
#         vis_markers[(vis_markers == 0) | (vis_markers > len(gt_points))] = np.nan
#         ax_a.imshow(vis_markers, cmap='jet', alpha=0.5, interpolation='nearest')
#         ax_a.set_title("A: Standard Watershed Basins (Isotropic Seed)", fontsize=12)
#         ax_a.axis('off')

#         # 图 B, C, D
#         titles = {
#             1: "B: Row 1 (Max IoU Search w/ Quantiles)", 
#             2: "C: Row 2 (MinAreaRect / Outer Box)", 
#             3: "D: Row 3 (Pred Angle Guided)"
#         }
        
#         for row_id, ax in zip([1, 2, 3], [axes[0, 1], axes[1, 0], axes[1, 1]]):
#             ax.imshow(img_rgb)
#             ious = []
#             for j in range(len(row_results[row_id])):
#                 p_box, iou = row_results[row_id][j]
                
#                 # 画 GT
#                 draw_rbox(ax, gt_polys[j], 'lime', linestyle='--') # GT 用虚线绿色
                
#                 # 画 Pred
#                 if p_box is not None:
#                     draw_rbox(ax, p_box, 'red')
#                     # 标注 IoU
#                     ax.text(gt_points[j][0], gt_points[j][1], f"{iou:.2f}", color='yellow', fontsize=8, fontweight='bold', 
#                             bbox=dict(facecolor='black', alpha=0.5, pad=0))
#                     ious.append(iou)
            
#             mean_iou = np.mean(ious) if ious else 0
#             ax.set_title(f"{titles[row_id]}\nMean IoU: {mean_iou:.4f}", fontsize=12, color='blue')
#             ax.axis('off')

#         plt.suptitle(f"Image ID: {file_id}\nGreen Dashed: GT | Red Solid: Estimated Box", fontsize=16)
        
#         out_path = os.path.join(SAVE_DIR, f"{file_id}_comparison.png")
#         plt.savefig(out_path, bbox_inches='tight', dpi=100)
#         plt.close()

# if __name__ == '__main__':
#     run_visualization()

import torch
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tqdm
from shapely.geometry import Polygon
import torch.nn.functional as F
from mmdet.apis import init_detector, inference_detector
import sys

# ==============================================================================
# 1. 全局配置
# ==============================================================================
CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-dota.py'
CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2/work_dirs/1/std/e2e/epoch_12.pth'
DATA_ROOT = '/mnt/data/xiekaikai/split_ss_dota/trainval'
SAVE_DIR = 'vis/dota2' # 修改保存路径以区分
DEVICE = 'cuda:1'
NUM_VIS = 20 # 目标可视化数量（必须成功画出的数量）
MAX_INSTANCES_PER_IMG = 4 # [关键] 仅可视化少于等于4个目标的图片

os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
           'harbor', 'swimming-pool', 'helicopter')

BEST_CFG = dict(default_sigma=4096, down_sample=2, pos_thres=0.994, neg_thres=0.005)

# ==============================================================================
# 2. 核心算法逻辑
# ==============================================================================

def find_max_iou_box_vectorized(mask_pts, center_pt, device='cuda'):
    """Row 1: 暴力搜索与 Mask IoU 最大的框 (修正版)"""
    if len(mask_pts) < 3: return None
    pts = torch.from_numpy(mask_pts).float().to(device)
    center = torch.tensor(center_pt, dtype=torch.float32, device=device)
    pts_centered = pts - center
    N = pts.shape[0]

    # 1. 角度搜索：0 ~ 180 度 (pi)
    num_angles = 180 
    angles = torch.linspace(0, torch.pi, num_angles, device=device)
    
    cos_a = torch.cos(angles); sin_a = torch.sin(angles)
    # 构建逆时针旋转矩阵
    R = torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a, cos_a], dim=-1)
    ], dim=-2)

    # 旋转点集: (A, N, 2)
    pts_rot = torch.matmul(R, pts_centered.T).transpose(1, 2).abs()

    # 2. 尺度搜索
    quantiles = torch.linspace(0.5, 1.0, 51, device=device)
    sorted_x, _ = torch.sort(pts_rot[..., 0], dim=1)
    sorted_y, _ = torch.sort(pts_rot[..., 1], dim=1)
    
    indices = ((quantiles * (N - 1)).long()).clamp(0, N-1)
    W_half = sorted_x[:, indices] # (A, Q)
    H_half = sorted_y[:, indices] # (A, Q)
    Box_Area = (W_half * 2) * (H_half * 2)

    # 3. 计算 IoU
    in_x = pts_rot[..., 0].unsqueeze(-1) <= W_half.unsqueeze(1)
    in_y = pts_rot[..., 1].unsqueeze(-1) <= H_half.unsqueeze(1)
    inter = (in_x & in_y).sum(dim=1).float()
    
    iou_mat = inter / (N + Box_Area - inter + 1e-6)
    
    # 4. 取最大值
    val, idx = torch.max(iou_mat.view(-1), 0)
    b_a_i, b_q_i = idx // len(quantiles), idx % len(quantiles)
    
    # 5. 提取参数
    angle_rad = angles[b_a_i].item()
    w = W_half[b_a_i, b_q_i].item() * 2
    h = H_half[b_a_i, b_q_i].item() * 2
    
    # [关键修正] 返回 -degrees(angle)，符合 OpenCV 定义
    return ((center_pt[0], center_pt[1]), (w, h), -np.degrees(angle_rad))

def get_watershed_wh_row3(mask_pts, center_pt, angle_rad):
    """Row 3: 按预测角度投影算宽高"""
    if len(mask_pts) == 0: return 0.0, 0.0
    R_inv = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)], [np.sin(-angle_rad), np.cos(-angle_rad)]])
    pts_rot = np.dot(mask_pts - center_pt, R_inv.T)
    return max(2 * np.max(np.abs(pts_rot[:, 0])), 1.0), max(2 * np.max(np.abs(pts_rot[:, 1])), 1.0)

def compute_iou_shapely(poly1, poly2):
    try:
        p1, p2 = Polygon(poly1), Polygon(poly2)
        if not p1.is_valid: p1 = p1.convex_hull
        if not p2.is_valid: p2 = p2.convex_hull
        inter_area = p1.intersection(p2).area
        return inter_area / (p1.area + p2.area - inter_area + 1e-6)
    except: return 0.0

def draw_rbox(ax, poly, color, linestyle='-'):
    """统一绘图接口"""
    if isinstance(poly, (tuple, list)) and len(poly) == 3:
        pts = cv2.boxPoints(poly)
    else:
        pts = poly
    pts = np.vstack([pts, pts[0]]) 
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, linestyle=linestyle)

# ==============================================================================
# 3. 主可视化程序
# ==============================================================================

def run_visualization():
    print(f"Initializing Model...")
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    except Exception as e:
        print(f"Model init warning: {e}. Trying to load with strict=False equivalent...")
        pass
    
    img_dir = os.path.join(DATA_ROOT, 'images')
    lbl_dir = os.path.join(DATA_ROOT, 'labelTxt')
    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    
    # 随机打乱以增加多样性
    random.shuffle(all_imgs)
    
    success_count = 0
    pbar = tqdm.tqdm(total=NUM_VIS)
    
    for img_name in all_imgs:
        if success_count >= NUM_VIS: break
        
        file_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, file_id + '.txt')
        if not os.path.exists(lbl_path): continue

        # 1. 预读取 GT，筛选稀疏图片
        gt_points, gt_polys = [], []
        try:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                # [关键] 过滤掉目标太多的图片
                if len(lines) > MAX_INSTANCES_PER_IMG or len(lines) == 0: 
                    continue
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 9: continue
                    poly = np.array(list(map(float, parts[:8]))).reshape(4, 2)
                    gt_polys.append(poly)
                    gt_points.append(np.mean(poly, axis=0))
        except: continue
        
        if not gt_points: continue

        # 2. 读取图片
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # 3. 生成 Standard Mask (各向同性种子)
        D = BEST_CFG['down_sample']
        gt_points_np = np.array(gt_points, dtype=np.float32)
        mu = torch.from_numpy(gt_points_np).to(DEVICE)
        
        mm = (mu / D).round()
        h_m, w_m = H // D, W // D
        
        ax_x = torch.linspace(0, w_m-1, w_m, device=DEVICE)
        ax_y = torch.linspace(0, h_m-1, h_m, device=DEVICE)
        gx, gy = torch.meshgrid(ax_x, ax_y, indexing='xy')
        xy_grid = torch.stack([gx, gy], -1)
        
        vor = torch.zeros((len(gt_points), h_m, w_m), device=DEVICE)
        sigma_val = BEST_CFG['default_sigma'] / (D**2)
        for j in range(len(gt_points)):
            d = (xy_grid - mm[j]).pow(2).sum(-1)
            vor[j] = torch.exp(-0.5 * d / sigma_val)
            
        val, vor_idx = torch.max(vor, 0)
        
        if D > 1:
            vor_idx = F.interpolate(vor_idx[None,None].float(), (H,W), mode='nearest')[0,0].long()
            val = F.interpolate(val[None,None], (H,W), mode='bilinear', align_corners=True)[0,0]
        
        markers = (vor_idx + 1).cpu().numpy().astype(np.int32)
        pos_mask = val.cpu().numpy() < BEST_CFG['pos_thres']
        neg_mask = val.cpu().numpy() < BEST_CFG['neg_thres']
        markers[pos_mask] = 0
        markers[neg_mask] = len(gt_points) + 1
        cv2.watershed(cv2.medianBlur(img, 3), markers)
        
        # 4. 模型预测
        pred_res = inference_detector(model, img)
        if hasattr(pred_res, 'pred_instances'):
            bboxes_obj = pred_res.pred_instances.bboxes
        else:
            bboxes_obj = pred_res.bboxes 
            
        if hasattr(bboxes_obj, 'tensor'):
            pred_bboxes = bboxes_obj.tensor
        elif isinstance(bboxes_obj, torch.Tensor):
            pred_bboxes = bboxes_obj
        else:
            pred_bboxes = torch.tensor(bboxes_obj, device=DEVICE)
            
        if len(pred_bboxes) == 0:
            pred_angles = np.zeros(len(gt_points))
        else:
            dists = torch.cdist(mu, pred_bboxes[:, :2])
            match_idx = dists.min(dim=1)[1]
            pred_angles = pred_bboxes[match_idx, 4].cpu().numpy()

        # 5. 计算数据
        row_results = {1: [], 2: [], 3: []}
        for idx in range(len(gt_points)):
            pts_y, pts_x = np.where(markers == idx + 1)
            if len(pts_x) < 3: 
                row_results[1].append((None, 0.0))
                row_results[2].append((None, 0.0))
                row_results[3].append((None, 0.0))
                continue
                
            mask_pts = np.stack([pts_x, pts_y], axis=1)
            gt_c = gt_points[idx]
            
            # Row 1
            r1_box = find_max_iou_box_vectorized(mask_pts, gt_c, DEVICE)
            iou1 = compute_iou_shapely(cv2.boxPoints(r1_box), gt_polys[idx]) if r1_box else 0.0
            
            # Row 2
            r2_box = cv2.minAreaRect(mask_pts.astype(np.float32))
            iou2 = compute_iou_shapely(cv2.boxPoints(r2_box), gt_polys[idx])
            
            # Row 3
            w3, h3 = get_watershed_wh_row3(mask_pts, gt_c, pred_angles[idx])
            r3_box = ((gt_c[0], gt_c[1]), (w3, h3), np.degrees(pred_angles[idx]))
            iou3 = compute_iou_shapely(cv2.boxPoints(r3_box), gt_polys[idx])
            
            row_results[1].append((r1_box, iou1))
            row_results[2].append((r2_box, iou2))
            row_results[3].append((r3_box, iou3))

        # 6. 绘图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(top=0.9, hspace=0.25)
        
        ax_a = axes[0, 0]
        ax_a.imshow(img_rgb)
        vis_markers = markers.copy().astype(float)
        vis_markers[(vis_markers == 0) | (vis_markers > len(gt_points))] = np.nan
        ax_a.imshow(vis_markers, cmap='jet', alpha=0.5, interpolation='nearest')
        ax_a.set_title("A: Standard Watershed Basins (Isotropic Seed)", fontsize=12)
        ax_a.axis('off')

        titles = {1: "B: Row 1 (Max IoU Search)", 2: "C: Row 2 (MinAreaRect)", 3: "D: Row 3 (Pred Angle Guided)"}
        for row_id, ax in zip([1, 2, 3], [axes[0, 1], axes[1, 0], axes[1, 1]]):
            ax.imshow(img_rgb)
            ious = []
            for j in range(len(row_results[row_id])):
                p_box, iou = row_results[row_id][j]
                draw_rbox(ax, gt_polys[j], 'lime', linestyle='--')
                if p_box is not None:
                    draw_rbox(ax, p_box, 'red')
                    ax.text(gt_points[j][0], gt_points[j][1], f"{iou:.2f}", color='yellow', fontsize=8, fontweight='bold', 
                            bbox=dict(facecolor='black', alpha=0.5, pad=0))
                    ious.append(iou)
            mean_iou = np.mean(ious) if ious else 0
            ax.set_title(f"{titles[row_id]}\nMean IoU: {mean_iou:.4f}", fontsize=12, color='blue')
            ax.axis('off')

        plt.suptitle(f"Image ID: {file_id}\nGreen Dashed: GT | Red Solid: Estimated Box", fontsize=16)
        plt.savefig(os.path.join(SAVE_DIR, f"{file_id}_comparison.png"), bbox_inches='tight', dpi=100)
        plt.close()
        
        success_count += 1
        pbar.update(1)

    print(f"Visualization complete. Saved {success_count} images to {SAVE_DIR}")

if __name__ == '__main__':
    run_visualization()