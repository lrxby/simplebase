# import os
# import cv2
# import torch
# import numpy as np
# import itertools
# import random
# from tqdm import tqdm
# from shapely.geometry import Polygon
# import torch.nn.functional as F
# from concurrent.futures import ProcessPoolExecutor

# # ==============================================================================
# # 1. 核心算法逻辑
# # ==============================================================================

# def get_gaussian_base(points, image_shape, default_sigma, down_sample, device='cuda'):
#     """
#     预计算高斯热图的基础分布（val 和 vor_idx）。
#     这一步最耗时，每个 (sigma, DS) 组合只需跑一次。
#     """
#     H, W = image_shape
#     D = down_sample
#     h, w = H // D, W // D
#     J = len(points)
    
#     if J == 0: return None

#     mu = torch.tensor(points, device=device, dtype=torch.float32)
#     sigma_val = default_sigma
#     sigma = torch.eye(2, device=device).unsqueeze(0).repeat(J, 1, 1) * sigma_val
    
#     x = torch.linspace(0, h-1, h, device=device)
#     y = torch.linspace(0, w-1, w, device=device)
#     xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    
#     vor_maps = torch.zeros((J, h, w), device=device)
#     mm = (mu / D).round()
#     sg = sigma / (D ** 2)
    
#     # 高斯核计算
#     for j in range(J):
#         dxy = (xy.view(-1, 2) - mm[j][None]).unsqueeze(-1)
#         # 简化计算：因为是等方差对角阵，可加速
#         vor_maps[j] = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sg[j][None], dxy))).view(h, w)
        
#     val, vor_idx = torch.max(vor_maps, 0)
    
#     if D > 1:
#         vor_idx = F.interpolate(vor_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
#         val = F.interpolate(val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
#     return val, vor_idx

# def fast_watershed_fit(val, vor_idx, image, params, num_instances):
#     """
#     仅进行阈值分割、分水岭和框拟合。此部分不涉及指数运算，速度极快。
#     """
#     device = val.device
#     J = num_instances
#     H, W = val.shape
    
#     # 边缘检测
#     kernel = torch.ones((1, 1, 3, 3), device=device)
#     kernel[0, 0, 1, 1] = -8
#     ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
    
#     markers = vor_idx.clone() + 1
#     markers[val < params['pos_thres']] = 0 
#     markers[val < params['neg_thres']] = J + 1 
#     markers[ridges] = J + 1
    
#     # Watershed (CPU)
#     markers_np = markers.cpu().numpy().astype(np.int32)
#     cv2.watershed(image, markers_np)
    
#     return markers_np

# # ==============================================================================
# # 2. 几何工具
# # ==============================================================================

# def mask2rbox(mask_pts):
#     if len(mask_pts) < 3: return None
#     rect = cv2.minAreaRect(mask_pts.astype(np.float32))
#     return cv2.boxPoints(rect) 

# def poly2rbox(poly):
#     poly = np.array(poly).reshape(-1, 2).astype(np.float32)
#     return cv2.boxPoints(cv2.minAreaRect(poly))

# def compute_iou(box1, box2):
#     try:
#         p1, p2 = Polygon(box1), Polygon(box2)
#         if not p1.is_valid or not p2.is_valid: return 0.0
#         inter = p1.intersection(p2).area
#         return inter / (p1.area + p2.area - inter + 1e-6)
#     except: return 0.0

# # ==============================================================================
# # 3. 数据加载 (多进程)
# # ==============================================================================

# def load_single_item(args):
#     img_dir, label_dir, file_id = args
#     img_path = os.path.join(img_dir, f"{file_id}.png")
#     if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
#     img = cv2.imread(img_path)
#     if img is None: return None
    
#     label_path = os.path.join(label_dir, f"{file_id}.txt")
#     points, gt_boxes = [], []
#     try:
#         with open(label_path, 'r') as f:
#             for line in f.readlines():
#                 parts = line.strip().split()
#                 if len(parts) < 9: continue 
#                 poly = list(map(float, parts[:8]))
#                 poly_np = np.array(poly).reshape(4, 2)
#                 points.append(np.mean(poly_np, axis=0))
#                 gt_boxes.append(poly2rbox(poly))
#         return {'image': img, 'points': np.array(points), 'gt_boxes': gt_boxes}
#     except: return None

# class DOTALoader:
#     def __init__(self, root_path, sample_ratio=0.1):
#         self.img_dir = os.path.join(root_path, 'images')
#         self.label_dir = os.path.join(root_path, 'labelTxt')
#         all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
#         sample_num = max(1, int(len(all_files) * sample_ratio))
#         self.files = random.sample(all_files, sample_num)
#         print(f"[Init] Sampling {len(self.files)} images (Ratio: {sample_ratio})")

#     def load_all(self):
#         tasks = [(self.img_dir, self.label_dir, f_id) for f_id in self.files]
#         with ProcessPoolExecutor() as executor:
#             results = list(tqdm(executor.map(load_single_item, tasks, chunksize=10), total=len(tasks), desc="Loading Images"))
#         return [r for r in results if r is not None]

# # ==============================================================================
# # 4. 主搜索逻辑 (重构循环结构)
# # ==============================================================================

# def run_grid_search():
#     data_root = '/mnt/data/xiekaikai/split_ss_dota/trainval' 
#     output_file = 'tuning_results.txt'
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
#     # 建议 sample_ratio 设为 0.01~0.05 进行快速调参，0.1 依然较慢
#     sample_ratio = 1 
    
#     search_sigma = [2048,4096,512,1024]
#     search_downsample = [2]
#     search_pos = [round(x, 2) for x in np.arange(0.95, 1.00, 0.005)] # 粗搜建议步长 0.05
#     search_neg = [round(x, 2) for x in np.arange(0.005, 0.05, 0.005)] # 粗搜建议步长 0.05
    
#     dataset = DOTALoader(data_root, sample_ratio=sample_ratio).load_all()
    
#     best_iou, best_params = 0, None
    
#     with open(output_file, 'w') as f_out:
#         f_out.write(f"sigma,DS,pos,neg,Avg_IoU\n")
        
#         # [核心优化：改变循环嵌套]
#         for sig, ds in itertools.product(search_sigma, search_downsample):
#             print(f"\n>>> Pre-computing Gaussian Bases for sigma={sig}, DS={ds}...")
#             bases = []
#             for data in dataset:
#                 bases.append(get_gaussian_base(data['points'], data['image'].shape[:2], sig, ds, device))
            
#             # 遍历阈值组合 (现在这部分极快)
#             threshold_combos = list(itertools.product(search_pos, search_neg))
#             for pos, neg in tqdm(threshold_combos, desc=f"Testing Thres (sig={sig}, ds={ds})", leave=False):
#                 if pos <= neg: continue
                
#                 total_iou, total_count = 0, 0
#                 params = {'pos_thres': pos, 'neg_thres': neg}
                
#                 for i, data in enumerate(dataset):
#                     if bases[i] is None: continue
#                     val, vor_idx = bases[i]
#                     markers = fast_watershed_fit(val, vor_idx, data['image'], params, len(data['points']))
                    
#                     for idx, gt_box in enumerate(data['gt_boxes']):
#                         y_idx, x_idx = np.where(markers == (idx + 1))
#                         if len(x_idx) < 3: continue 
#                         pred_box = mask2rbox(np.stack([x_idx, y_idx], axis=1))
#                         if pred_box is not None:
#                             total_iou += compute_iou(pred_box, gt_box)
#                             total_count += 1
                
#                 avg_iou = total_iou / max(1, total_count)
#                 f_out.write(f"{sig},{ds},{pos},{neg},{avg_iou:.5f}\n")
#                 f_out.flush()
                
#                 if avg_iou > best_iou:
#                     best_iou, best_params = avg_iou, {'sigma': sig, 'ds': ds, 'pos': pos, 'neg': neg}
#                     print(f" [New Best] IoU: {best_iou:.4f} | sig={sig}, ds={ds}, pos={pos}, neg={neg}")

#         print(f"\nFinal Best: {best_iou:.5f} with {best_params}")

# if __name__ == '__main__':
#     run_grid_search()

# import os
# import cv2
# import torch
# import numpy as np
# import itertools
# import random
# from tqdm import tqdm
# from shapely.geometry import Polygon
# import torch.nn.functional as F
# from concurrent.futures import ProcessPoolExecutor

# # ==============================================================================
# # 1. 核心算法逻辑 (保持不变)
# # ==============================================================================

# def get_gaussian_base(points, image_shape, default_sigma, down_sample, device='cuda'):
#     H, W = image_shape
#     D = down_sample
#     h, w = H // D, W // D
#     J = len(points)
    
#     if J == 0: return None

#     mu = torch.tensor(points, device=device, dtype=torch.float32)
#     # Standard Mode: Isotropic Sigma
#     sigma = torch.eye(2, device=device).unsqueeze(0).repeat(J, 1, 1) * default_sigma
    
#     x = torch.linspace(0, h-1, h, device=device)
#     y = torch.linspace(0, w-1, w, device=device)
#     xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    
#     # 显存优化：逐个实例计算，避免生成巨大的 (J, h, w) 张量
#     # 如果 J 很大 (如 > 500)，一次性分配 vor_maps 会直接 OOM
#     # 我们改为直接维护 max_val 和 max_idx
    
#     mm = (mu / D).round()
#     sg = sigma / (D ** 2)
    
#     # 初始化输出 map
#     global_val = torch.zeros((h, w), device=device, dtype=torch.float32)
#     global_idx = torch.zeros((h, w), device=device, dtype=torch.long)
    
#     # 批处理实例，防止 J 过大
#     batch_size = 100 
#     for start in range(0, J, batch_size):
#         end = min(start + batch_size, J)
#         sub_J = end - start
        
#         # 计算当前 batch 的热图
#         batch_maps = torch.zeros((sub_J, h, w), device=device)
#         for k in range(sub_J):
#             j = start + k
#             dxy = (xy.view(-1, 2) - mm[j][None]).unsqueeze(-1)
#             batch_maps[k] = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sg[j][None], dxy))).view(h, w)
        
#         # 局部 max
#         sub_val, sub_idx_rel = torch.max(batch_maps, 0)
#         sub_idx = sub_idx_rel + start # 修正索引偏移
        
#         # 更新全局 max
#         update_mask = sub_val > global_val
#         global_val[update_mask] = sub_val[update_mask]
#         global_idx[update_mask] = sub_idx[update_mask]
        
#         del batch_maps, sub_val, sub_idx_rel # 及时释放
        
#     if D > 1:
#         global_idx = F.interpolate(global_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
#         global_val = F.interpolate(global_val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
#     return global_val, global_idx

# def fast_watershed_fit(val, vor_idx, image, pos_thres, neg_thres, num_instances):
#     device = val.device
#     J = num_instances
    
#     kernel = torch.ones((1, 1, 3, 3), device=device)
#     kernel[0, 0, 1, 1] = -8
#     ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
    
#     markers = vor_idx.clone() + 1
#     markers[val < pos_thres] = 0 
#     markers[val < neg_thres] = J + 1 
#     markers[ridges] = J + 1
    
#     markers_np = markers.cpu().numpy().astype(np.int32)
#     cv2.watershed(image, markers_np)
    
#     return markers_np

# # ==============================================================================
# # 2. 几何工具 (保持不变)
# # ==============================================================================
# def mask2rbox(mask_pts):
#     if len(mask_pts) < 3: return None
#     rect = cv2.minAreaRect(mask_pts.astype(np.float32))
#     return cv2.boxPoints(rect) 

# def poly2rbox(poly):
#     poly = np.array(poly).reshape(-1, 2).astype(np.float32)
#     return cv2.boxPoints(cv2.minAreaRect(poly))

# def compute_iou(box1, box2):
#     try:
#         p1, p2 = Polygon(box1), Polygon(box2)
#         if not p1.is_valid or not p2.is_valid: return 0.0
#         inter = p1.intersection(p2).area
#         return inter / (p1.area + p2.area - inter + 1e-6)
#     except: return 0.0

# # ==============================================================================
# # 3. 数据加载 (保持不变)
# # ==============================================================================
# def load_single_item(args):
#     img_dir, label_dir, file_id = args
#     img_path = os.path.join(img_dir, f"{file_id}.png")
#     if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
#     img = cv2.imread(img_path)
#     if img is None: return None
    
#     label_path = os.path.join(label_dir, f"{file_id}.txt")
#     points, gt_boxes = [], []
#     try:
#         with open(label_path, 'r') as f:
#             for line in f.readlines():
#                 parts = line.strip().split()
#                 if len(parts) < 9: continue 
#                 poly = list(map(float, parts[:8]))
#                 poly_np = np.array(poly).reshape(4, 2)
#                 points.append(np.mean(poly_np, axis=0))
#                 gt_boxes.append(poly2rbox(poly))
#         return {'image': img, 'points': np.array(points), 'gt_boxes': gt_boxes}
#     except: return None

# class DOTALoader:
#     def __init__(self, root_path, sample_ratio=0.1):
#         self.img_dir = os.path.join(root_path, 'images')
#         self.label_dir = os.path.join(root_path, 'labelTxt')
#         all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
#         # sample_num = max(1, int(len(all_files) * sample_ratio))
#         # self.files = random.sample(all_files, sample_num)
#         self.files = all_files # 跑全量
#         print(f"[Init] Sampling {len(self.files)} images (Ratio: {sample_ratio})")

#     def load_all(self):
#         tasks = [(self.img_dir, self.label_dir, f_id) for f_id in self.files]
#         with ProcessPoolExecutor() as executor:
#             results = list(tqdm(executor.map(load_single_item, tasks, chunksize=10), total=len(tasks), desc="Loading Images"))
#         return [r for r in results if r is not None]

# # ==============================================================================
# # 4. 优化后的主搜索逻辑 (Inverted Loop)
# # ==============================================================================

# def run_grid_search():
#     data_root = '/mnt/data/xiekaikai/split_ss_dota/trainval' 
#     output_file = 'tuning_results_v2.txt'
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
#     search_sigma = [2048, 4096, 512, 1024]
#     search_downsample = [2]
#     # 缩小搜索范围以加快速度
#     search_pos = [0.95, 0.98, 0.99, 0.994] 
#     search_neg = [0.005, 0.01, 0.02, 0.05]
    
#     # 预先生成所有参数组合
#     threshold_combos = list(itertools.product(search_pos, search_neg))
    
#     # 用于累积每个组合的统计数据
#     # Key: (sigma, ds, pos, neg) -> Value: {'iou': 0, 'count': 0}
#     stats = {}
#     for sig in search_sigma:
#         for ds in search_downsample:
#             for p, n in threshold_combos:
#                 if p > n:
#                     stats[(sig, ds, p, n)] = {'iou': 0.0, 'count': 0}

#     print("Loading dataset into RAM...")
#     dataset = DOTALoader(data_root, sample_ratio=1.0).load_all()
#     print(f"Loaded {len(dataset)} images.")
    
#     # --- 核心修改：以图片为主循环 ---
#     # 这样一张图片在显存里只需要存在一次，跑完所有参数就释放
    
#     # 外层循环 Sigma (因为 Sigma 决定了 Base Map 的形状，计算最重)
#     for sig, ds in itertools.product(search_sigma, search_downsample):
#         print(f"\n>>> Processing Dataset for Sigma={sig}, DS={ds}...")
        
#         for data in tqdm(dataset):
#             # 1. 计算当前图片的 Gaussian Base (GPU)
#             try:
#                 base = get_gaussian_base(data['points'], data['image'].shape[:2], sig, ds, device)
#             except Exception as e:
#                 print(f"Error in Gaussian Base: {e}")
#                 continue
                
#             if base is None: continue
#             val, vor_idx = base
            
#             # 2. 遍历所有阈值组合 (复用同一个 Base)
#             for pos, neg in threshold_combos:
#                 if pos <= neg: continue
#                 key = (sig, ds, pos, neg)
                
#                 # 快速分水岭
#                 try:
#                     markers = fast_watershed_fit(val, vor_idx, data['image'], pos, neg, len(data['points']))
                    
#                     # 统计 IoU
#                     img_iou = 0
#                     img_count = 0
#                     for idx, gt_box in enumerate(data['gt_boxes']):
#                         y_idx, x_idx = np.where(markers == (idx + 1))
#                         if len(x_idx) < 3: continue 
#                         pred_box = mask2rbox(np.stack([x_idx, y_idx], axis=1))
#                         if pred_box is not None:
#                             img_iou += compute_iou(pred_box, gt_box)
#                             img_count += 1
                    
#                     if img_count > 0:
#                         stats[key]['iou'] += img_iou
#                         stats[key]['count'] += img_count
                        
#                 except Exception as e:
#                     pass

#             # 3. 释放显存！关键！
#             del val, vor_idx
#             # torch.cuda.empty_cache() # 可选，如果还不够用就打开这行

#     # --- 汇总结果 ---
#     print("\nWriting results...")
#     best_iou = 0
#     best_cfg = None
    
#     with open(output_file, 'w') as f:
#         f.write("sigma,DS,pos,neg,MeanIoU\n")
#         for key, val in stats.items():
#             sig, ds, p, n = key
#             total_iou = val['iou']
#             total_count = val['count']
            
#             mean_iou = total_iou / max(1, total_count)
#             f.write(f"{sig},{ds},{p},{n},{mean_iou:.5f}\n")
            
#             if mean_iou > best_iou:
#                 best_iou = mean_iou
#                 best_cfg = key

#     print(f"Best Result: IoU={best_iou:.4f} @ {best_cfg}")

# if __name__ == '__main__':
#     run_grid_search()

import os
import cv2
import torch
import numpy as np
import itertools
import random
from tqdm import tqdm
from shapely.geometry import Polygon
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 1. 核心算法 (保持不变，无需修改)
# ==============================================================================

def get_gaussian_base(points, image_shape, default_sigma, down_sample, device='cuda'):
    H, W = image_shape
    D = down_sample
    h, w = H // D, W // D
    J = len(points)
    if J == 0: return None

    mu = torch.tensor(points, device=device, dtype=torch.float32)
    sigma = torch.eye(2, device=device).unsqueeze(0).repeat(J, 1, 1) * default_sigma
    
    x = torch.linspace(0, h-1, h, device=device)
    y = torch.linspace(0, w-1, w, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    
    mm = (mu / D).round()
    sg = sigma / (D ** 2)
    
    global_val = torch.zeros((h, w), device=device, dtype=torch.float32)
    global_idx = torch.zeros((h, w), device=device, dtype=torch.long)
    
    batch_size = 100 
    for start in range(0, J, batch_size):
        end = min(start + batch_size, J)
        sub_J = end - start
        batch_maps = torch.zeros((sub_J, h, w), device=device)
        for k in range(sub_J):
            j = start + k
            dxy = (xy.view(-1, 2) - mm[j][None]).unsqueeze(-1)
            batch_maps[k] = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sg[j][None], dxy))).view(h, w)
        sub_val, sub_idx_rel = torch.max(batch_maps, 0)
        sub_idx = sub_idx_rel + start
        update_mask = sub_val > global_val
        global_val[update_mask] = sub_val[update_mask]
        global_idx[update_mask] = sub_idx[update_mask]
        del batch_maps, sub_val, sub_idx_rel
        
    if D > 1:
        global_idx = F.interpolate(global_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
        global_val = F.interpolate(global_val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
    return global_val, global_idx

def fast_watershed_fit(val, vor_idx, image, pos_thres, neg_thres, num_instances):
    device = val.device
    J = num_instances
    kernel = torch.ones((1, 1, 3, 3), device=device)
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
    markers = vor_idx.clone() + 1
    markers[val < pos_thres] = 0 
    markers[val < neg_thres] = J + 1 
    markers[ridges] = J + 1
    markers_np = markers.cpu().numpy().astype(np.int32)
    cv2.watershed(image, markers_np)
    return markers_np

def mask2rbox(mask_pts):
    if len(mask_pts) < 3: return None
    rect = cv2.minAreaRect(mask_pts.astype(np.float32))
    return cv2.boxPoints(rect) 

def poly2rbox(poly):
    poly = np.array(poly).reshape(-1, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(poly))

def compute_iou(box1, box2):
    try:
        p1, p2 = Polygon(box1), Polygon(box2)
        if not p1.is_valid or not p2.is_valid: return 0.0
        inter = p1.intersection(p2).area
        return inter / (p1.area + p2.area - inter + 1e-6)
    except: return 0.0

# ==============================================================================
# 2. 数据加载
# ==============================================================================

def load_single_item(args):
    img_dir, label_dir, file_id = args
    img_path = os.path.join(img_dir, f"{file_id}.png")
    if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
    img = cv2.imread(img_path)
    if img is None: return None
    
    label_path = os.path.join(label_dir, f"{file_id}.txt")
    points, gt_boxes = [], []
    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 9: continue 
                poly = list(map(float, parts[:8]))
                poly_np = np.array(poly).reshape(4, 2)
                points.append(np.mean(poly_np, axis=0))
                gt_boxes.append(poly2rbox(poly))
        return {'image': img, 'points': np.array(points), 'gt_boxes': gt_boxes}
    except: return None

class DOTALoader:
    def __init__(self, root_path):
        self.img_dir = os.path.join(root_path, 'images')
        self.label_dir = os.path.join(root_path, 'labelTxt')
        self.all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
        print(f"[Init] Found {len(self.all_files)} total images.")

    def load_sampled(self, num_samples=2000):
        # [核心修改] 强制只取 N 张图
        sample_num = min(len(self.all_files), num_samples)
        files_to_load = random.sample(self.all_files, sample_num)
        
        print(f"[Init] Randomly sampling {sample_num} images for Fast Grid Search...")
        
        tasks = [(self.img_dir, self.label_dir, f_id) for f_id in files_to_load]
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(load_single_item, tasks, chunksize=10), total=len(tasks), desc="Loading Images"))
        return [r for r in results if r is not None]

# ==============================================================================
# 3. 极速 Grid Search (带实时反馈)
# ==============================================================================

def run_fast_search():
    data_root = '/mnt/data/xiekaikai/split_ss_dota/trainval' 
    output_file = 'tuning_results_fast.txt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # [关键] 限制图片数量，2000张足够了
    NUM_SAMPLES = 2000
    
    # 搜索空间
    search_sigma = [2048, 4096, 512, 1024]
    search_downsample = [2]
    # 稍微放宽一点范围，步长 0.01 足够粗搜
    search_pos = [0.95, 0.96, 0.97, 0.98, 0.99, 0.994] 
    search_neg = [0.005, 0.01, 0.02, 0.05]
    
    threshold_combos = list(itertools.product(search_pos, search_neg))
    threshold_combos = [(p, n) for p, n in threshold_combos if p > n]
    
    print("Loading dataset...")
    loader = DOTALoader(data_root)
    dataset = loader.load_sampled(NUM_SAMPLES)
    
    # 初始化统计字典
    # Key: (sigma, ds, pos, neg) -> Value: {'iou': 0, 'count': 0}
    stats = {}
    
    with open(output_file, 'w') as f:
        f.write("sigma,DS,pos,neg,MeanIoU\n")

    print("\n>>> Starting Fast Grid Search...")
    print(f"Total Params to test: {len(search_sigma) * len(threshold_combos)}")
    
    # 记录全局最优
    global_best_iou = 0.0
    global_best_cfg = None

    for sig, ds in itertools.product(search_sigma, search_downsample):
        print(f"\n[Testing Sigma={sig}, DS={ds}]")
        
        # 清空当前 Sigma 的统计缓存
        current_sigma_stats = {(p,n): {'iou':0, 'cnt':0} for p,n in threshold_combos}
        
        # 遍历图片
        for img_idx, data in enumerate(tqdm(dataset, desc=f"Sigma {sig}")):
            try:
                base = get_gaussian_base(data['points'], data['image'].shape[:2], sig, ds, device)
            except Exception as e:
                continue
            if base is None: continue
            val, vor_idx = base
            
            # 对这张图，跑所有阈值
            for pos, neg in threshold_combos:
                try:
                    markers = fast_watershed_fit(val, vor_idx, data['image'], pos, neg, len(data['points']))
                    
                    img_iou_sum = 0
                    img_match_count = 0
                    for idx, gt_box in enumerate(data['gt_boxes']):
                        y_idx, x_idx = np.where(markers == (idx + 1))
                        if len(x_idx) < 3: continue 
                        pred_box = mask2rbox(np.stack([x_idx, y_idx], axis=1))
                        if pred_box is not None:
                            img_iou_sum += compute_iou(pred_box, gt_box)
                            img_match_count += 1
                    
                    if img_match_count > 0:
                        current_sigma_stats[(pos,neg)]['iou'] += img_iou_sum
                        current_sigma_stats[(pos,neg)]['cnt'] += img_match_count
                        
                except: pass
            
            del val, vor_idx
            
            # [实时反馈] 每 100 张图打印一次当前 Sigma 下的最好参数
            if (img_idx + 1) % 100 == 0:
                temp_best_iou = 0
                temp_best_p = None
                for (p, n), s in current_sigma_stats.items():
                    if s['cnt'] > 0:
                        iou = s['iou'] / s['cnt']
                        if iou > temp_best_iou:
                            temp_best_iou = iou
                            temp_best_p = (p, n)
                tqdm.write(f"   > Progress {img_idx+1}/{len(dataset)} | Current Best: IoU={temp_best_iou:.4f} @ (pos={temp_best_p[0]}, neg={temp_best_p[1]})")

        # --- 一个 Sigma 跑完，立刻汇总写入文件 ---
        with open(output_file, 'a') as f:
            for (p, n), s in current_sigma_stats.items():
                if s['cnt'] > 0:
                    mean_iou = s['iou'] / s['cnt']
                    # 写入文件
                    f.write(f"{sig},{ds},{p},{n},{mean_iou:.5f}\n")
                    
                    # 更新全局最优
                    if mean_iou > global_best_iou:
                        global_best_iou = mean_iou
                        global_best_cfg = (sig, ds, p, n)
        
        print(f"--- Finished Sigma={sig} ---")
        print(f"--- Current Global Best: {global_best_iou:.4f} @ {global_best_cfg} ---")

    print(f"\nAll Done! Final Best: {global_best_iou:.4f} @ {global_best_cfg}")

if __name__ == '__main__':
    run_fast_search()