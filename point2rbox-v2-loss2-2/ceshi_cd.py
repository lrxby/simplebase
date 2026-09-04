# import torch
# import numpy as np
# import cv2
# import os
# import tqdm
# import random
# import math
# import sys
# import datetime
# from shapely.geometry import Polygon
# import torch.nn.functional as F

# from mmdet.apis import init_detector, inference_detector
# from mmrotate.structures import RotatedBoxes

# # ==============================================================================
# # 1. 全局配置 (CoDrone Row 3)
# # ==============================================================================
# NUM_TEST_IMAGES = -1  
# # 请修改为您的 CoDrone 配置文件路径
# CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-codrone.py'
# # 请修改为您的 CoDrone 权重路径
# CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2-loss2-2/work_dirs/cd/std0.1/e2e/epoch_4.pth'
# DATA_ROOT = '/mnt/data/xiekaikai/split_ss_codrone'
# DEVICE = 'cuda:7'

# LOG_DIR = 'ceshi/codrone'
# os.makedirs(LOG_DIR, exist_ok=True)
# TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# LOG_FILE = os.path.join(LOG_DIR, f'result_row3_hybrid_{TIMESTAMP}.txt')

# CLASSES = ('car', 'truck', 'bus', 'traffic-light', 'traffic-sign', 'bridge', 
#            'people', 'bicycle', 'motor', 'tricycle', 'boat', 'ship')

# # ==============================================================================
# # 2. 严格复刻参数
# # ==============================================================================
# BASELINE_CFG = dict(
#     default_sigma=2048, 
#     down_sample=2,
#     thres_default=[0.97, 0.05], # CoDrone 建议阈值
#     thres_override=[],
#     square_cls=[1, 9, 5], # truck, tricycle, bridge (参考之前 Row 1/2)
#     post_process={11: 1.3, 6: 1.2} # ship, people
# )

# # ... [中间的日志类和工具函数与上面完全一致，已省略以节省篇幅] ...
# # 请直接复制上面的 Logger, compute_size_iou, get_thres_for_label, 
# # rbox2poly_custom, compute_iou_shapely, calc_angle_diff, 
# # get_watershed_wh_original_loss, gaussian_2d 
# # 以及核心 generate_mask_for_hybrid 函数到这里。
# # 它们是通用的。

# # (为方便您运行，这里重复核心部分，确保代码完整性)
# # ==============================================================================
# # 3. 日志工具 & 核心函数
# # ==============================================================================
# class Logger(object):
#     def __init__(self, filename):
#         self.terminal = sys.stdout; self.log = open(filename, "a", encoding='utf-8')
#     def write(self, message): self.terminal.write(message); self.log.write(message); self.log.flush()
#     def flush(self): self.terminal.flush(); self.log.flush()

# def compute_size_iou(w1, h1, w2, h2):
#     def _iou(wa, ha, wb, hb):
#         i = min(wa, wb) * min(ha, hb); u = (wa * ha) + (wb * hb) - i
#         return i / (u + 1e-6)
#     return max(_iou(w1, h1, w2, h2), _iou(w1, h1, h2, w2))

# def get_thres_for_label(label_id):
#     for cls_list, thres in BASELINE_CFG['thres_override']:
#         if label_id in cls_list: return thres[0], thres[1]
#     return BASELINE_CFG['thres_default'][0], BASELINE_CFG['thres_default'][1]

# def rbox2poly_custom(cx, cy, w, h, a_rad):
#     theta = a_rad; cos_a = math.cos(theta); sin_a = math.sin(theta)
#     R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
#     corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
#     return np.dot(corners, R.T) + np.array([cx, cy])

# def compute_iou_shapely(box1, box2):
#     try:
#         p1 = Polygon(box1); p2 = Polygon(box2)
#         if not p1.is_valid: p1 = p1.convex_hull
#         if not p2.is_valid: p2 = p2.convex_hull
#         inter = p1.intersection(p2).area
#         return inter / (p1.area + p2.area - inter + 1e-6)
#     except: return 0.0

# def calc_angle_diff(angle1_rad, angle2_rad):
#     deg1 = np.degrees(angle1_rad); deg2 = np.degrees(angle2_rad)
#     diff = abs(deg1 - deg2) % 180
#     return min(diff, 180 - diff)

# def get_watershed_wh_original_loss(mask_pts, center_pt, angle_rad):
#     if len(mask_pts) == 0: return 0.0, 0.0
#     pts_rot = np.dot(mask_pts - center_pt, np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)], [np.sin(-angle_rad), np.cos(-angle_rad)]]).T)
#     return max(2 * np.max(np.abs(pts_rot[:, 0])), 1.0), max(2 * np.max(np.abs(pts_rot[:, 1])), 1.0)

# def gaussian_2d(xy, mu, sigma):
#     dxy = (xy - mu).unsqueeze(-1)
#     inv_sigma = torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)
#     return torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(inv_sigma))

# def generate_mask_for_hybrid(img_rgb, gt_points, gt_labels, pred_results, device='cuda'):
#     H, W = img_rgb.shape[:2]; D = BASELINE_CFG['down_sample']; DEFAULT_SIGMA = BASELINE_CFG['default_sigma']
#     if len(gt_points) == 0: return np.zeros((H, W), dtype=np.int32), np.array([])
#     if hasattr(pred_results, 'pred_instances'): pred_bboxes = pred_results.pred_instances.bboxes
#     else: pred_bboxes = pred_results.bboxes
#     if hasattr(pred_bboxes, 'tensor'): pred_bboxes = pred_bboxes.tensor
#     if len(pred_bboxes) == 0: return np.zeros((H, W), dtype=np.int32), np.array([])

#     gt_tensor = torch.tensor(gt_points, device=device, dtype=torch.float32)
#     pred_centers = pred_bboxes[:, :2]
#     match_idxs = torch.cdist(gt_tensor, pred_centers).min(dim=1)[1]
#     matched_bboxes = pred_bboxes[match_idxs]
    
#     w, h, a = matched_bboxes[:, 2], matched_bboxes[:, 3], matched_bboxes[:, 4]
#     cos_a, sin_a = torch.cos(a), torch.sin(a)
#     R = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1).reshape(-1, 2, 2)
#     Lambda = torch.zeros_like(R); Lambda[:, 0, 0] = (w/2)**2; Lambda[:, 1, 1] = (h/2)**2
#     sigma_phys = torch.bmm(R, torch.bmm(Lambda, R.transpose(1, 2)))
    
#     L, V = torch.linalg.eigh(sigma_phys); L = L.clamp(min=1e-7)
#     scale_factor = DEFAULT_SIGMA / (L[:, 0:1] * L[:, 1:2]).sqrt()
#     sigma_norm = V.matmul(torch.diag_embed(L * scale_factor)).matmul(V.permute(0, 2, 1))
#     sg = sigma_norm / (D ** 2)
    
#     h_map, w_map = H // D, W // D
#     xy = torch.stack(torch.meshgrid(torch.linspace(0, h_map-1, h_map, device=device), torch.linspace(0, w_map-1, w_map, device=device), indexing='xy'), -1)
    
#     J = len(gt_points); vor = torch.zeros((J, h_map, w_map), device=device)
#     mm = (gt_tensor / D).round()
#     for j in range(J): vor[j] = gaussian_2d(xy.view(-1, 2), mm[j][None], sg[j][None]).view(h_map, w_map)
#     val, vor_idx = torch.max(vor, 0)
    
#     if D > 1:
#         vor_idx = F.interpolate(vor_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
#         val = F.interpolate(val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
#     kernel = torch.ones((1, 1, 3, 3), device=device); kernel[0, 0, 1, 1] = -8
#     ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
#     vor_idx += 1 
    
#     pos_map = torch.full_like(val, BASELINE_CFG['thres_default'][0])
#     neg_map = torch.full_like(val, BASELINE_CFG['thres_default'][1])
#     inst_pos = []; inst_neg = []
#     for lbl in gt_labels: p, n = get_thres_for_label(lbl); inst_pos.append(p); inst_neg.append(n)
    
#     mask_inst = vor_idx > 0
#     if mask_inst.any():
#         idx_temp = vor_idx[mask_inst] - 1
#         pos_map[mask_inst] = torch.tensor(inst_pos, device=device)[idx_temp]
#         neg_map[mask_inst] = torch.tensor(inst_neg, device=device)[idx_temp]
    
#     vor_idx[val < pos_map] = 0; vor_idx[val < neg_map] = J + 1; vor_idx[ridges] = J + 1
    
#     markers_np = vor_idx.cpu().numpy().astype(np.int32)
#     img_blur = cv2.medianBlur(img_rgb, 3)
#     cv2.watershed(img_blur, markers_np)
#     return markers_np, matched_bboxes[:, 4].cpu().numpy()

# # ==============================================================================
# # 7. Loader (CoDrone)
# # ==============================================================================
# class CoDroneLoader:
#     def __init__(self, root_path, sample_num=-1):
#         self.img_dir = os.path.join(root_path, 'trainval', 'images')
#         self.label_dir = os.path.join(root_path, 'trainval', 'annfiles')
#         self.all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
#         if sample_num > 0 and sample_num < len(self.all_files):
#             print(f"Randomly sampling {sample_num} images...")
#             self.files = random.sample(self.all_files, sample_num)
#         else:
#             print(f"Loading ALL {len(self.all_files)} images...")
#             self.files = self.all_files

#     def load_item(self, idx):
#         file_id = self.files[idx]
#         img_path = os.path.join(self.img_dir, f"{file_id}.png")
#         if not os.path.exists(img_path): img_path = os.path.join(self.img_dir, f"{file_id}.jpg")
#         if not os.path.exists(img_path): return None
        
#         label_path = os.path.join(self.label_dir, f"{file_id}.txt")
#         points, gt_boxes, gt_labels = [], [], []
#         try:
#             with open(label_path, 'r') as f:
#                 for line in f.readlines():
#                     parts = line.strip().split()
#                     if len(parts) < 9: continue
#                     poly = np.array(list(map(float, parts[:8]))).reshape(4, 2).astype(np.float32)
#                     cls_name = parts[8]
#                     if cls_name not in CLASSES: continue
#                     points.append(np.mean(poly, axis=0))
#                     gt_boxes.append(poly) 
#                     gt_labels.append(CLASSES.index(cls_name))
#         except: return None
#         return {'img_path': img_path, 'points': np.array(points), 'gt_boxes': gt_boxes, 'gt_labels': gt_labels}

# # ==============================================================================
# # 8. 主程序
# # ==============================================================================
# def run_hybrid_analysis():
#     sys.stdout = Logger(LOG_FILE)
#     print(f"Row 3 Analysis (Hybrid) - CoDrone")
#     print(f"Config: {BASELINE_CFG}")
    
#     print(f"Loading model...")
#     model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
#     loader = CoDroneLoader(DATA_ROOT, sample_num=NUM_TEST_IMAGES)
    
#     cls_stats = {k: {'iou': 0.0, 'size_iou': 0.0, 'angle_diff': 0.0, 'count': 0} for k in CLASSES}
    
#     print(f"Starting Analysis...")
#     for i in tqdm.tqdm(range(len(loader.files))):
#         data = loader.load_item(i)
#         if data is None: continue
        
#         img = cv2.imread(data['img_path'])
#         if img is None: continue
        
#         result = inference_detector(model, img)
#         mask, pred_angles_rad = generate_mask_for_hybrid(
#             img, data['points'], data['gt_labels'], result, device=DEVICE)
        
#         min_len = min(len(data['gt_boxes']), len(pred_angles_rad))
        
#         for idx in range(min_len):
#             gt_poly = data['gt_boxes'][idx]
#             label_id = data['gt_labels'][idx]
#             pred_a_rad = pred_angles_rad[idx]
#             label_name = CLASSES[label_id]
            
#             pts_y, pts_x = np.where(mask == idx + 1)
#             if len(pts_x) < 3: 
#                 cls_stats[label_name]['count'] += 1
#                 continue
            
#             mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
            
#             if label_id in BASELINE_CFG['square_cls']: final_angle = 0.0
#             else: final_angle = pred_a_rad
            
#             gt_cx, gt_cy = data['points'][idx]
#             w_pred, h_pred = get_watershed_wh_original_loss(mask_pts, np.array([gt_cx, gt_cy]), final_angle)
            
#             if label_id in BASELINE_CFG['post_process']:
#                 scale = BASELINE_CFG['post_process'][label_id]
#                 w_pred *= scale; h_pred *= scale
                
#             pred_box = rbox2poly_custom(gt_cx, gt_cy, w_pred, h_pred, final_angle)
            
#             iou = compute_iou_shapely(pred_box, gt_poly)
#             gt_rect = cv2.minAreaRect(gt_poly.astype(np.float32))
#             size_iou = compute_size_iou(w_pred, h_pred, gt_rect[1][0], gt_rect[1][1])
            
#             if label_id in BASELINE_CFG['square_cls']: ang_diff = 0.0
#             else: ang_diff = calc_angle_diff(pred_a_rad, np.radians(gt_rect[2]))
            
#             cls_stats[label_name]['iou'] += iou
#             cls_stats[label_name]['size_iou'] += size_iou
#             cls_stats[label_name]['angle_diff'] += ang_diff
#             cls_stats[label_name]['count'] += 1
            
#     print("\n" + "="*85)
#     print(f"{'Class':<18} | {'mIoU':<8} | {'mSize':<8} | {'mAngle(°)':<10} | {'Count':<6}")
#     print("-" * 85)
#     total_iou, total_size, total_ang, total_n = 0, 0, 0, 0
#     for k in CLASSES:
#         s = cls_stats[k]; n = s['count']
#         if n > 0:
#             avg_iou = s['iou'] / n; avg_size = s['size_iou'] / n; avg_ang = s['angle_diff'] / n
#             total_iou += s['iou']; total_size += s['size_iou']; total_ang += s['angle_diff']; total_n += n
#         else: avg_iou = 0.0; avg_size = 0.0; avg_ang = 0.0
#         print(f"{k:<18} | {avg_iou:.4f}   | {avg_size:.4f}   | {avg_ang:.2f}       | {n:<6}")
#     print("-" * 85)
#     denom = max(1, total_n)
#     print(f"OVERALL | mIoU: {total_iou/denom:.4f} | mSize: {total_size/denom:.4f} | mAngle: {total_ang/denom:.2f}°")
#     print("="*85)
#     print(f"Results saved to {LOG_FILE}")

# if __name__ == '__main__':
#     run_hybrid_analysis()

# import torch
# import numpy as np
# import cv2
# import os
# import tqdm
# import random
# import math
# import sys
# import datetime
# from shapely.geometry import Polygon
# import torch.nn.functional as F

# from mmdet.apis import init_detector, inference_detector
# from mmrotate.structures import RotatedBoxes

# # ==============================================================================
# # 1. 全局配置 (CoDrone Row 3 - Standard Mode)
# # ==============================================================================
# NUM_TEST_IMAGES = -1 
# CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-codrone.py'
# CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2-loss2-2/work_dirs/cd/std0.1/e2e/epoch_4.pth'
# DATA_ROOT = '/mnt/data/xiekaikai/split_ss_codrone'
# DEVICE = 'cuda:5'

# LOG_DIR = 'ceshi/codrone'
# os.makedirs(LOG_DIR, exist_ok=True)
# TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# LOG_FILE = os.path.join(LOG_DIR, f'result_row3_standard_mode_{TIMESTAMP}.txt')

# CLASSES = ('car', 'truck', 'bus', 'traffic-light', 'traffic-sign', 'bridge', 
#            'people', 'bicycle', 'motor', 'tricycle', 'boat', 'ship')

# # ==============================================================================
# # 2. 严格复刻参数
# # ==============================================================================
# BASELINE_CFG = dict(
#     default_sigma=2048, 
#     down_sample=2,
#     thres_default=[0.97, 0.05], 
#     thres_override=[],
#     square_cls=[1, 9, 5], 
#     post_process={11: 1.3, 6: 1.2} 
# )

# # ==============================================================================
# # 3. 工具函数
# # ==============================================================================
# class Logger(object):
#     def __init__(self, filename):
#         self.terminal = sys.stdout; self.log = open(filename, "a", encoding='utf-8')
#     def write(self, message): self.terminal.write(message); self.log.write(message); self.log.flush()
#     def flush(self): self.terminal.flush(); self.log.flush()

# def compute_size_iou(w1, h1, w2, h2):
#     def _iou(wa, ha, wb, hb):
#         i = min(wa, wb) * min(ha, hb); u = (wa * ha) + (wb * hb) - i
#         return i / (u + 1e-6)
#     return max(_iou(w1, h1, w2, h2), _iou(w1, h1, h2, w2))

# def get_thres_for_label(label_id):
#     for cls_list, thres in BASELINE_CFG['thres_override']:
#         if label_id in cls_list: return thres[0], thres[1]
#     return BASELINE_CFG['thres_default'][0], BASELINE_CFG['thres_default'][1]

# def rbox2poly_custom(cx, cy, w, h, a_rad):
#     theta = a_rad; cos_a = math.cos(theta); sin_a = math.sin(theta)
#     R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
#     corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
#     return np.dot(corners, R.T) + np.array([cx, cy])

# def compute_iou_shapely(box1, box2):
#     try:
#         p1 = Polygon(box1); p2 = Polygon(box2)
#         if not p1.is_valid: p1 = p1.convex_hull
#         if not p2.is_valid: p2 = p2.convex_hull
#         inter = p1.intersection(p2).area
#         return inter / (p1.area + p2.area - inter + 1e-6)
#     except: return 0.0

# def calc_angle_diff(angle1_rad, angle2_rad):
#     deg1 = np.degrees(angle1_rad); deg2 = np.degrees(angle2_rad)
#     diff = abs(deg1 - deg2) % 180
#     return min(diff, 180 - diff)

# def get_watershed_wh_original_loss(mask_pts, center_pt, angle_rad):
#     if len(mask_pts) == 0: return 0.0, 0.0
#     cos_inv = np.cos(-angle_rad); sin_inv = np.sin(-angle_rad)
#     R_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])
#     pts_rot = np.dot(mask_pts - center_pt, R_inv.T)
#     return max(2 * np.max(np.abs(pts_rot[:, 0])), 1.0), max(2 * np.max(np.abs(pts_rot[:, 1])), 1.0)

# def gaussian_2d(xy, mu, sigma):
#     dxy = (xy - mu).unsqueeze(-1)
#     inv_sigma = torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)
#     return torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(inv_sigma))

# def generate_mask_standard_with_pred(img_rgb, gt_points, gt_labels, pred_results, device='cuda'):
#     H, W = img_rgb.shape[:2]; D = BASELINE_CFG['down_sample']; DEFAULT_SIGMA = BASELINE_CFG['default_sigma']
#     if len(gt_points) == 0: return np.zeros((H, W), dtype=np.int32), np.array([])
    
#     if hasattr(pred_results, 'pred_instances'): pred_bboxes = pred_results.pred_instances.bboxes
#     else: pred_bboxes = pred_results.bboxes
#     if hasattr(pred_bboxes, 'tensor'): pred_bboxes = pred_bboxes.tensor
#     if len(pred_bboxes) == 0: return np.zeros((H, W), dtype=np.int32), np.array([])

#     gt_tensor = torch.tensor(gt_points, device=device, dtype=torch.float32)
#     pred_centers = pred_bboxes[:, :2]
#     match_idxs = torch.cdist(gt_tensor, pred_centers).min(dim=1)[1]
#     matched_bboxes = pred_bboxes[match_idxs]
#     pred_angles = matched_bboxes[:, 4].cpu().numpy()

#     J = len(gt_points)
#     sigma_standard = torch.eye(2, device=device).unsqueeze(0).repeat(J, 1, 1) * DEFAULT_SIGMA
#     sg = sigma_standard / (D ** 2)
    
#     h_map, w_map = H // D, W // D
#     xy = torch.stack(torch.meshgrid(torch.linspace(0, h_map-1, h_map, device=device), torch.linspace(0, w_map-1, w_map, device=device), indexing='xy'), -1)
#     vor = torch.zeros((J, h_map, w_map), device=device)
#     mm = (gt_tensor / D).round()
    
#     for j in range(J): 
#         vor[j] = gaussian_2d(xy.view(-1, 2), mm[j][None], sg[j][None]).view(h_map, w_map)
#     val, vor_idx = torch.max(vor, 0)
    
#     if D > 1:
#         vor_idx = F.interpolate(vor_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
#         val = F.interpolate(val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
#     kernel = torch.ones((1, 1, 3, 3), device=device); kernel[0, 0, 1, 1] = -8
#     ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
#     vor_idx += 1 
    
#     pos_map = torch.full_like(val, BASELINE_CFG['thres_default'][0])
#     neg_map = torch.full_like(val, BASELINE_CFG['thres_default'][1])
#     inst_pos = []; inst_neg = []
#     for lbl in gt_labels:
#         p, n = get_thres_for_label(lbl)
#         inst_pos.append(p); inst_neg.append(n)
    
#     mask_inst = vor_idx > 0
#     if mask_inst.any():
#         idx_temp = vor_idx[mask_inst] - 1
#         pos_map[mask_inst] = torch.tensor(inst_pos, device=device)[idx_temp]
#         neg_map[mask_inst] = torch.tensor(inst_neg, device=device)[idx_temp]
    
#     vor_idx[val < pos_map] = 0; vor_idx[val < neg_map] = J + 1; vor_idx[ridges] = J + 1
    
#     markers_np = vor_idx.cpu().numpy().astype(np.int32)
#     img_blur = cv2.medianBlur(img_rgb, 3)
#     cv2.watershed(img_blur, markers_np)
    
#     return markers_np, pred_angles

# # ==============================================================================
# # 4. Loader & 主程序
# # ==============================================================================
# class CoDroneLoader:
#     def __init__(self, root_path, sample_num=-1):
#         self.img_dir = os.path.join(root_path, 'trainval', 'images')
#         self.label_dir = os.path.join(root_path, 'trainval', 'annfiles')
#         self.all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
#         if sample_num > 0 and sample_num < len(self.all_files):
#             self.files = random.sample(self.all_files, sample_num)
#         else:
#             self.files = self.all_files

#     def load_item(self, idx):
#         file_id = self.files[idx]
#         img_path = os.path.join(self.img_dir, f"{file_id}.png")
#         if not os.path.exists(img_path): img_path = os.path.join(self.img_dir, f"{file_id}.jpg")
#         if not os.path.exists(img_path): return None
#         label_path = os.path.join(self.label_dir, f"{file_id}.txt")
#         points, gt_boxes, gt_labels = [], [], []
#         try:
#             with open(label_path, 'r') as f:
#                 for line in f.readlines():
#                     parts = line.strip().split()
#                     if len(parts) < 9: 
#                         continue
#                     poly = np.array(list(map(float, parts[:8]))).reshape(4, 2).astype(np.float32)
#                     cls_name = parts[8]
#                     if cls_name not in CLASSES: 
#                         continue
#                     points.append(np.mean(poly, axis=0))
#                     gt_boxes.append(poly)
#                     gt_labels.append(CLASSES.index(cls_name))
#         except: return None
#         return {'img_path': img_path, 'points': np.array(points), 'gt_boxes': gt_boxes, 'gt_labels': gt_labels}

# def print_current_stats(cls_stats, current_idx, total_len):
#     print(f"\n[Progress] {current_idx}/{total_len} ({(current_idx/total_len)*100:.1f}%)")
#     print("="*85); print(f"{'Class':<18} | {'mIoU':<8} | {'mSize':<8} | {'mAngle(°)':<10} | {'Count':<6}"); print("-" * 85)
#     total_iou = total_size = total_ang = total_n = 0
#     for k in CLASSES:
#         s = cls_stats[k]; n = s['count']
#         if n > 0:
#             avg_iou = s['iou']/n
#             avg_size = s['size_iou']/n
#             avg_ang = s['angle_diff']/n
#             total_iou += s['iou']; total_size += s['size_iou']; total_ang += s['angle_diff']; total_n += n
#         else: avg_iou = avg_size = avg_ang = 0.0
#         print(f"{k:<18} | {avg_iou:.4f}   | {avg_size:.4f}   | {avg_ang:.2f}       | {n:<6}")
#     print("-" * 85); denom = max(1, total_n)
#     print(f"OVERALL | mIoU: {total_iou/denom:.4f} | mSize: {total_size/denom:.4f} | mAngle: {total_ang/denom:.2f}°")
#     print("="*85 + "\n")

# def run_row3_standard_mode():
#     sys.stdout = Logger(LOG_FILE)
#     print(f"Row 3 Analysis (Standard Mode) - CoDrone")
#     model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
#     loader = CoDroneLoader(DATA_ROOT, sample_num=NUM_TEST_IMAGES)
#     cls_stats = {k: {'iou': 0.0, 'size_iou': 0.0, 'angle_diff': 0.0, 'count': 0} for k in CLASSES}
    
#     total_files = len(loader.files); log_interval = max(1, total_files // 10)
#     for i in tqdm.tqdm(range(total_files)):
#         data = loader.load_item(i)
#         if data is None: continue
#         img = cv2.imread(data['img_path'])
#         if img is None: continue
        
#         result = inference_detector(model, img)
#         mask, pred_angles_rad = generate_mask_standard_with_pred(img, data['points'], data['gt_labels'], result, device=DEVICE)
        
#         min_len = min(len(data['gt_boxes']), len(pred_angles_rad))
#         for idx in range(min_len):
#             gt_poly = data['gt_boxes'][idx]; label_id = data['gt_labels'][idx]; pred_a_rad = pred_angles_rad[idx]; label_name = CLASSES[label_id]
#             pts_y, pts_x = np.where(mask == idx + 1)
#             if len(pts_x) < 3: cls_stats[label_name]['count'] += 1; continue
            
#             mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
#             final_angle = 0.0 if label_id in BASELINE_CFG['square_cls'] else pred_a_rad
#             gt_cx, gt_cy = data['points'][idx]
#             w_pred, h_pred = get_watershed_wh_original_loss(mask_pts, np.array([gt_cx, gt_cy]), final_angle)
            
#             if label_id in BASELINE_CFG['post_process']:
#                 scale = BASELINE_CFG['post_process'][label_id]; w_pred *= scale; h_pred *= scale
            
#             pred_box = rbox2poly_custom(gt_cx, gt_cy, w_pred, h_pred, final_angle)
#             iou = compute_iou_shapely(pred_box, gt_poly)
#             gt_rect = cv2.minAreaRect(gt_poly.astype(np.float32))
#             size_iou = compute_size_iou(w_pred, h_pred, gt_rect[1][0], gt_rect[1][1])
#             ang_diff = 0.0 if label_id in BASELINE_CFG['square_cls'] else calc_angle_diff(pred_a_rad, np.radians(gt_rect[2]))
            
#             cls_stats[label_name]['iou'] += iou; cls_stats[label_name]['size_iou'] += size_iou; cls_stats[label_name]['angle_diff'] += ang_diff; cls_stats[label_name]['count'] += 1
            
#         if (i + 1) % log_interval == 0: print_current_stats(cls_stats, i + 1, total_files)
#     print_current_stats(cls_stats, total_files, total_files); print(f"Results saved to {LOG_FILE}")

# if __name__ == '__main__':
#     run_row3_standard_mode()

import torch
import numpy as np
import cv2
import os
import tqdm
import random
import math
import sys
import datetime
from shapely.geometry import Polygon
import torch.nn.functional as F

from mmdet.apis import init_detector, inference_detector

# ==============================================================================
# 1. 全局配置 (CoDrone Row 3 - Standard Mode + LED)
# ==============================================================================
NUM_TEST_IMAGES = -1  # -1 代表跑全部，也支持输入小数如 0.05
CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-codrone.py'
CHECKPOINT_FILE = '/mnt/data/liurunxiang/workplace/point2rbox-v2-loss2-2/work_dirs/cd/std0.1/e2e/epoch_4.pth'
DATA_ROOT = '/mnt/data/xiekaikai/split_ss_codrone'
DEVICE = 'cuda:1'

LOG_DIR = 'ceshi/codrone'
os.makedirs(LOG_DIR, exist_ok=True)
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f'result_row3_standard_mode_LED_{TIMESTAMP}.txt')

CLASSES = ('car', 'truck', 'bus', 'traffic-light', 'traffic-sign', 'bridge', 
           'people', 'bicycle', 'motor', 'tricycle', 'boat', 'ship')

# ==============================================================================
# 2. 严格复刻参数
# ==============================================================================
BASELINE_CFG = dict(
    default_sigma=2048, 
    down_sample=2,
    thres_default=[0.97, 0.05], 
    thres_override=[],
    square_cls=[], 
    post_process={} 
)

# ==============================================================================
# 3. 工具函数与 LED 统一评测体系
# ==============================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout; self.log = open(filename, "a", encoding='utf-8')
    def write(self, message): self.terminal.write(message); self.log.write(message); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()

def get_thres_for_label(label_id):
    for cls_list, thres in BASELINE_CFG['thres_override']:
        if label_id in cls_list: return thres[0], thres[1]
    return BASELINE_CFG['thres_default'][0], BASELINE_CFG['thres_default'][1]

def rbox2poly_custom(cx, cy, w, h, a_rad):
    theta = a_rad; cos_a = math.cos(theta); sin_a = math.sin(theta)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    return np.dot(corners, R.T) + np.array([cx, cy])

def compute_iou_shapely(box1, box2):
    try:
        p1 = Polygon(box1); p2 = Polygon(box2)
        if not p1.is_valid: p1 = p1.convex_hull
        if not p2.is_valid: p2 = p2.convex_hull
        inter = p1.intersection(p2).area
        return inter / (p1.area + p2.area - inter + 1e-6)
    except: return 0.0

def calc_angle_diff(angle1_deg, angle2_deg):
    deg1 = angle1_deg % 180; deg2 = angle2_deg % 180
    diff = abs(deg1 - deg2)
    return min(diff, 180 - diff)

def compute_metrics(box_pred, box_gt, w_p, h_p, w_g, h_g, ang_p_rad, ang_g_deg, label_id):
    """LED (长边对齐) 统一评价体系"""
    # 1. 多边形 IoU
    iou = compute_iou_shapely(box_pred, box_gt)
    
    # 2. 预测框 LED 对齐
    if h_p > w_p:
        w_p, h_p = h_p, w_p
        ang_p_rad += np.pi / 2
        
    # 3. GT框 LED 对齐
    if h_g > w_g:
        w_g, h_g = h_g, w_g
        ang_g_deg += 90.0

    # 4. Size IoU
    def _siou(wa, ha, wb, hb):
        i = min(wa, wb) * min(ha, hb)
        u = wa*ha + wb*hb - i
        return i / (u + 1e-6)
    size_iou = _siou(w_p, h_p, w_g, h_g) 
    
    # 5. 角度误差
    if label_id in BASELINE_CFG['square_cls']: 
        ang_diff = 0.0
    else: 
        ang_diff = calc_angle_diff(np.degrees(ang_p_rad), ang_g_deg)
        
    return iou, size_iou, ang_diff

def get_watershed_wh_original_loss(mask_pts, center_pt, angle_rad):
    if len(mask_pts) == 0: return 0.0, 0.0
    cos_inv = np.cos(-angle_rad); sin_inv = np.sin(-angle_rad)
    R_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])
    pts_rot = np.dot(mask_pts - center_pt, R_inv.T)
    return max(2 * np.max(np.abs(pts_rot[:, 0])), 1.0), max(2 * np.max(np.abs(pts_rot[:, 1])), 1.0)

def gaussian_2d(xy, mu, sigma):
    dxy = (xy - mu).unsqueeze(-1)
    inv_sigma = torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)
    return torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(inv_sigma))

# ==============================================================================
# 4. Mask 生成核心
# ==============================================================================
def generate_mask_standard_with_pred(img_rgb, gt_points, gt_labels, pred_results, device='cuda'):
    H, W = img_rgb.shape[:2]; D = BASELINE_CFG['down_sample']; DEFAULT_SIGMA = BASELINE_CFG['default_sigma']
    if len(gt_points) == 0: return np.zeros((H, W), dtype=np.int32), np.array([])
    
    if hasattr(pred_results, 'pred_instances'): pred_bboxes = pred_results.pred_instances.bboxes
    else: pred_bboxes = pred_results.bboxes
    if hasattr(pred_bboxes, 'tensor'): pred_bboxes = pred_bboxes.tensor
    if len(pred_bboxes) == 0: return np.zeros((H, W), dtype=np.int32), np.array([])

    gt_tensor = torch.tensor(gt_points, device=device, dtype=torch.float32)
    pred_centers = pred_bboxes[:, :2]
    match_idxs = torch.cdist(gt_tensor, pred_centers).min(dim=1)[1]
    matched_bboxes = pred_bboxes[match_idxs]
    pred_angles = matched_bboxes[:, 4].cpu().numpy()

    J = len(gt_points)
    sigma_standard = torch.eye(2, device=device).unsqueeze(0).repeat(J, 1, 1) * DEFAULT_SIGMA
    sg = sigma_standard / (D ** 2)
    
    h_map, w_map = H // D, W // D
    ax_x = torch.linspace(0, w_map-1, w_map, device=device)
    ax_y = torch.linspace(0, h_map-1, h_map, device=device)
    grid_x, grid_y = torch.meshgrid(ax_x, ax_y, indexing='xy')
    xy = torch.stack([grid_x, grid_y], -1) 
    
    vor = torch.zeros((J, h_map, w_map), device=device)
    mm = (gt_tensor / D).round()
    
    for j in range(J): 
        vor[j] = gaussian_2d(xy.view(-1, 2), mm[j][None], sg[j][None]).view(h_map, w_map)
    val, vor_idx = torch.max(vor, 0)
    
    if D > 1:
        vor_idx = F.interpolate(vor_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
        val = F.interpolate(val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
    kernel = torch.ones((1, 1, 3, 3), device=device); kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
    vor_idx += 1 
    
    pos_map = torch.full_like(val, BASELINE_CFG['thres_default'][0])
    neg_map = torch.full_like(val, BASELINE_CFG['thres_default'][1])
    inst_pos = []; inst_neg = []
    for lbl in gt_labels:
        p, n = get_thres_for_label(lbl)
        inst_pos.append(p); inst_neg.append(n)
    
    mask_inst = vor_idx > 0
    if mask_inst.any():
        idx_temp = vor_idx[mask_inst] - 1
        pos_map[mask_inst] = torch.tensor(inst_pos, device=device)[idx_temp]
        neg_map[mask_inst] = torch.tensor(inst_neg, device=device)[idx_temp]
    
    vor_idx[val < pos_map] = 0; vor_idx[val < neg_map] = J + 1; vor_idx[ridges] = J + 1
    
    markers_np = vor_idx.cpu().numpy().astype(np.int32)
    img_blur = cv2.medianBlur(img_rgb, 3)
    cv2.watershed(img_blur, markers_np)
    
    return markers_np, pred_angles

# ==============================================================================
# 5. Loader & 主程序
# ==============================================================================
class CoDroneLoader:
    def __init__(self, root_path, sample_num=-1):
        self.img_dir = os.path.join(root_path, 'trainval', 'images')
        self.label_dir = os.path.join(root_path, 'trainval', 'annfiles')
        self.all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
        
        if sample_num > 0:
            if isinstance(sample_num, float) and sample_num <= 1.0:
                k = int(len(self.all_files) * sample_num)
            else:
                k = int(sample_num)
            if k < len(self.all_files):
                self.files = random.sample(self.all_files, k)
            else:
                self.files = self.all_files
        else:
            self.files = self.all_files

    def load_item(self, idx):
        file_id = self.files[idx]
        img_path = os.path.join(self.img_dir, f"{file_id}.png")
        if not os.path.exists(img_path): img_path = os.path.join(self.img_dir, f"{file_id}.jpg")
        if not os.path.exists(img_path): return None
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        points, gt_boxes, gt_labels = [], [], []
        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 9: 
                        continue
                    poly = np.array(list(map(float, parts[:8]))).reshape(4, 2).astype(np.float32)
                    cls_name = parts[8]
                    if cls_name not in CLASSES: 
                        continue
                    points.append(np.mean(poly, axis=0))
                    gt_boxes.append(poly)
                    gt_labels.append(CLASSES.index(cls_name))
        except: return None
        return {'img_path': img_path, 'points': np.array(points), 'gt_boxes': gt_boxes, 'gt_labels': gt_labels}

def print_current_stats(cls_stats, current_idx, total_len):
    print(f"\n[Progress] {current_idx}/{total_len} ({(current_idx/total_len)*100:.1f}%)")
    print("="*85); print(f"{'Class':<18} | {'mIoU':<8} | {'mSize':<8} | {'mAngle(°)':<10} | {'Count':<6}"); print("-" * 85)
    total_iou = total_size = total_ang = total_n = 0
    for k in CLASSES:
        s = cls_stats[k]; n = s['count']
        if n > 0:
            avg_iou = s['iou']/n
            avg_size = s['size_iou']/n
            avg_ang = s['angle_diff']/n
            total_iou += s['iou']; total_size += s['size_iou']; total_ang += s['angle_diff']; total_n += n
        else: avg_iou = avg_size = avg_ang = 0.0
        print(f"{k:<18} | {avg_iou:.4f}   | {avg_size:.4f}   | {avg_ang:.2f}       | {n:<6}")
    print("-" * 85); denom = max(1, total_n)
    print(f"OVERALL | mIoU: {total_iou/denom:.4f} | mSize: {total_size/denom:.4f} | mAngle: {total_ang/denom:.2f}°")
    print("="*85 + "\n")

def run_row3_standard_mode():
    sys.stdout = Logger(LOG_FILE)
    print(f"Row 3 Analysis (Standard Mode + Pred Angle + LED Fixed) - CoDrone")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    loader = CoDroneLoader(DATA_ROOT, sample_num=NUM_TEST_IMAGES)
    cls_stats = {k: {'iou': 0.0, 'size_iou': 0.0, 'angle_diff': 0.0, 'count': 0} for k in CLASSES}
    
    total_files = len(loader.files); log_interval = max(1, total_files // 10)
    for i in tqdm.tqdm(range(total_files)):
        data = loader.load_item(i)
        if data is None: continue
        img = cv2.imread(data['img_path'])
        if img is None: continue
        
        result = inference_detector(model, img)
        mask, pred_angles_rad = generate_mask_standard_with_pred(img, data['points'], data['gt_labels'], result, device=DEVICE)
        
        min_len = min(len(data['gt_boxes']), len(pred_angles_rad))
        for idx in range(min_len):
            gt_poly = data['gt_boxes'][idx]; label_id = data['gt_labels'][idx]; pred_a_rad = pred_angles_rad[idx]; label_name = CLASSES[label_id]
            pts_y, pts_x = np.where(mask == idx + 1)
            if len(pts_x) < 3: cls_stats[label_name]['count'] += 1; continue
            
            mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
            final_angle = 0.0 if label_id in BASELINE_CFG['square_cls'] else pred_a_rad
            gt_cx, gt_cy = data['points'][idx]
            w_pred, h_pred = get_watershed_wh_original_loss(mask_pts, np.array([gt_cx, gt_cy]), final_angle)
            
            if label_id in BASELINE_CFG['post_process']:
                scale = BASELINE_CFG['post_process'][label_id]
                w_pred *= scale
                h_pred *= scale
            
            pred_box = rbox2poly_custom(gt_cx, gt_cy, w_pred, h_pred, final_angle)
            
            # --- 【关键修复：调用统一的 LED 评测体系】 ---
            gt_rect = cv2.minAreaRect(gt_poly.astype(np.float32))
            w_g, h_g = gt_rect[1]
            ang_g_deg = gt_rect[2]
            
            iou, size_iou, ang_diff = compute_metrics(
                pred_box, gt_poly,
                w_pred, h_pred, w_g, h_g, final_angle, ang_g_deg, label_id
            )
            
            cls_stats[label_name]['iou'] += iou
            cls_stats[label_name]['size_iou'] += size_iou
            cls_stats[label_name]['angle_diff'] += ang_diff
            cls_stats[label_name]['count'] += 1
            
        if (i + 1) % log_interval == 0: print_current_stats(cls_stats, i + 1, total_files)
    print_current_stats(cls_stats, total_files, total_files); print(f"Results saved to {LOG_FILE}")

if __name__ == '__main__':
    run_row3_standard_mode()