# # 注水盆地中心
# import torch
# import numpy as np
# import cv2
# import os
# import tqdm
# import random
# import sys
# import pickle  
# import math
# import torch.nn.functional as F

# # ==============================================================================
# # 1. 配置区域 
# # ==============================================================================
# DATA_ROOT = '/mnt/data/xiekaikai/dota/val/'
# SAVE_PATH = '/mnt/data/xiekaikai/dota/val/val.pkl'
# DEVICE = 'cuda:6'

# # 🌟 新增功能：在这里手动选择生成方法！
# # 可选值：'Row5_PCA2' 或 'Row6_Rect'
# METHOD = 'Row5_PCA2'  

# # 当前方法的最优参数
# CFG = dict(sigma=4096, ds=2, pos=0.996, neg=0.1) #PCA
# # CFG = dict(sigma=4096, ds=2, pos=0.996, neg=0.009) #Rect

# # 后处理配置 (直接应用到伪标签中)
# POST_PROCESS = {
#     11: 1.2  # Roundabout 放大 1.2 倍
# }

# CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#            'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#            'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
#            'harbor', 'swimming-pool', 'helicopter')

# # ==============================================================================
# # 2. 核心算法函数
# # ==============================================================================

# def rbox2sigma(w, h, angle, device):
#     w = torch.tensor(w, device=device, dtype=torch.float32)
#     h = torch.tensor(h, device=device, dtype=torch.float32)
#     angle = torch.tensor(angle, device=device, dtype=torch.float32)
#     cos_a = torch.cos(angle); sin_a = torch.sin(angle)
#     R = torch.stack([torch.stack([cos_a, -sin_a], -1), torch.stack([sin_a, cos_a], -1)], -2)
#     Lambda = torch.stack([torch.stack([(w/2)**2, torch.zeros_like(w)], -1), 
#                           torch.stack([torch.zeros_like(w), (h/2)**2], -1)], -2)
#     return R @ Lambda @ R.transpose(-1, -2)

# def get_watershed_wh_original_loss(mask_pts, center_pt, angle_rad):
#     """投影测量法"""
#     if len(mask_pts) < 2: return 16.0, 16.0 # Fallback for tiny masks
#     pts_centered = mask_pts - center_pt
#     cos_inv = np.cos(-angle_rad); sin_inv = np.sin(-angle_rad)
#     R_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])
#     pts_rot = np.dot(pts_centered, R_inv.T)
#     w = max(2 * np.max(np.abs(pts_rot[:, 0])), 4.0) # Min size constraint
#     h = max(2 * np.max(np.abs(pts_rot[:, 1])), 4.0)
#     return w, h

# def gaussian_2d(xy, mu, sigma):
#     dxy = (xy - mu).unsqueeze(-1)
#     inv_sigma = torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)
#     return torch.exp(-0.5 * dxy.permute(0, 1, 3, 2).matmul(inv_sigma)).squeeze()

# def run_watershed_generation(sigmas, gt_points, img_shape, pos_thres, neg_thres, down_sample, device):
#     H, W = img_shape[:2]; D = down_sample; J = len(gt_points)
#     h_map, w_map = H // D, W // D
#     if J == 0: return np.zeros((H, W), dtype=np.int32)
    
#     ax_x = torch.linspace(0, w_map-1, w_map, device=device)
#     ax_y = torch.linspace(0, h_map-1, h_map, device=device)
#     grid_x, grid_y = torch.meshgrid(ax_x, ax_y, indexing='xy')
#     xy = torch.stack([grid_x, grid_y], -1)
    
#     mm = (torch.tensor(gt_points, device=device).float() / D).round()
#     sg = sigmas / (D**2)
    
#     vor = torch.zeros((J, h_map, w_map), device=device)
#     for j in range(J): vor[j] = gaussian_2d(xy, mm[j], sg[j])
        
#     val, vor_idx = torch.max(vor, 0)
#     if D > 1:
#         vor_idx = F.interpolate(vor_idx[None, None].float(), size=(H, W), mode='nearest')[0, 0].long()
#         val = F.interpolate(val[None, None], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
#     # 注：原版代码中包含这段山脊隔离逻辑，我们予以保留以防止目标粘连
#     kernel = torch.ones((1, 1, 3, 3), device=device); kernel[0,0,1,1] = -8
#     ridges = torch.conv2d(vor_idx.float()[None,None], kernel, padding=1)[0,0] != 0
    
#     vor_idx += 1
#     vor_idx[val < pos_thres] = 0
#     vor_idx[val < neg_thres] = J + 1
#     vor_idx[ridges] = J + 1
    
#     return vor_idx.cpu().numpy().astype(np.int32)

# def analyze_geometry_specific(mask_stage1, gt_points, default_sigma, device, method):
#     """
#     通用几何分析：支持 PCA2 和 Rect 自由切换
#     """
#     J = len(gt_points)
#     sigmas_list = []
#     angles_list = []
    
#     for j in range(J):
#         pts_y, pts_x = np.where(mask_stage1 == j + 1)
        
#         # --- 异常处理：Fallback ---
#         if len(pts_x) < 5:
#             fb_s = torch.eye(2, device=device) * default_sigma
#             sigmas_list.append(fb_s)
#             angles_list.append(0.0)
#             continue
            
#         mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
        
#         # 在原版逻辑中，构建指导 Stage2 注水的 Sigma 时，依然使用的是人工标注点
#         gt_center = gt_points[j]
        
#         # --- 根据选择的方法提取角度 ---
#         if method == 'Row6_Rect':
#             rect = cv2.minAreaRect(mask_pts)
#             _, _, angle_deg_rect = rect
#             # 【物理长边对齐校正】
#             if angle_deg_rect > 45: 
#                 angle_deg_rect -= 90
#             theta = np.radians(angle_deg_rect)
            
#         elif method == 'Row5_PCA2':
#             pts_c = mask_pts - np.mean(mask_pts, axis=0)
#             cov = np.dot(pts_c.T, pts_c) / (mask_pts.shape[0] - 1)
#             eig_vals, eig_vecs = np.linalg.eigh(cov)
#             v_sub = eig_vecs[:, 0] # PCA 次轴
#             theta = np.arctan2(v_sub[1], v_sub[0])
            
#         else:
#             theta = 0.0

#         w_raw, h_raw = get_watershed_wh_original_loss(mask_pts, gt_center, theta)
#         area_raw = (w_raw / 2) * (h_raw / 2)
#         scale = np.sqrt(default_sigma / (area_raw + 1e-6))
        
#         sigmas_list.append(rbox2sigma(w_raw*scale, h_raw*scale, theta, device))
#         angles_list.append(theta)
        
#     return torch.stack(sigmas_list), np.array(angles_list)

# # ==============================================================================
# # 3. 数据生成主流程
# # ==============================================================================

# def generate_pseudo_labels():
#     print(f"Start Generating Pseudo Labels for DOTA...")
#     print(f"Selected Method: {METHOD} (Using CENTROID for final measurement)")
#     print(f"Config: {CFG}")
#     print(f"Output: {SAVE_PATH}")
    
#     img_dir = os.path.join(DATA_ROOT, 'images')
#     lbl_dir = os.path.join(DATA_ROOT, 'labelTxt')
#     files = sorted([os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    
#     pseudo_data_dict = {}
#     valid_count = 0
#     total_objs = 0
    
#     for i, file_id in enumerate(tqdm.tqdm(files)):
#         img_path = os.path.join(img_dir, f"{file_id}.png")
#         if not os.path.exists(img_path): img_path = os.path.join(img_dir, f"{file_id}.jpg")
#         if not os.path.exists(img_path): continue
        
#         gt_points, gt_labels = [], []
#         try:
#             with open(os.path.join(lbl_dir, f"{file_id}.txt"), 'r') as f:
#                 for ln in f:
#                     p = ln.strip().split()
#                     if len(p) < 9 or p[8] not in CLASSES: continue
#                     poly = np.array(list(map(float, p[:8]))).reshape(4, 2)
#                     gt_points.append(poly.mean(0))
#                     gt_labels.append(CLASSES.index(p[8]))
#         except: continue
        
#         if len(gt_points) == 0:
#             pseudo_data_dict[file_id] = []
#             continue
            
#         gt_points = np.array(gt_points)
#         J = len(gt_points)
#         img = cv2.medianBlur(cv2.imread(img_path), 3)
#         H, W = img.shape[:2]
        
#         # 2. Stage 1 (Standard)
#         s_std = torch.eye(2, device=DEVICE).unsqueeze(0).repeat(J, 1, 1) * CFG['sigma']
#         mask_s1 = run_watershed_generation(s_std, gt_points, img.shape, CFG['pos'], CFG['neg'], CFG['ds'], DEVICE)
#         cv2.watershed(img, mask_s1)
        
#         # 3. Geometry Analysis
#         sigmas_s2, angles_s2 = analyze_geometry_specific(mask_s1, gt_points, CFG['sigma'], DEVICE, method=METHOD)
        
#         # 4. Stage 2 (Refined)
#         mask_s2 = run_watershed_generation(sigmas_s2, gt_points, img.shape, CFG['pos'], CFG['neg'], CFG['ds'], DEVICE)
#         cv2.watershed(img, mask_s2)
        
#         # 5. Final Measurement & Storage
#         img_pseudo_boxes = []
#         for idx in range(J):
#             py, px = np.where(mask_s2 == idx + 1)
#             label_id = gt_labels[idx]
#             cx, cy = gt_points[idx]
            
#             # --- Fallback for bad mask ---
#             if len(px) < 3:
#                 # 默认小框
#                 img_pseudo_boxes.append([cx, cy, 16.0, 16.0, 0.0, label_id])
#                 total_objs += 1
#                 continue
            
#             m_pts = np.stack([px, py], axis=1).astype(np.float32)
#             angle = angles_s2[idx]
            
#             # 【🔥 核心保留】：依然使用水团质心来计算宽高！防止小目标虚胖！
#             center = m_pts.mean(0) 
#             w_raw, h_raw = get_watershed_wh_original_loss(m_pts, center, angle)
            
#             # --- 后处理 ---
#             if label_id in POST_PROCESS:
#                 scale = POST_PROCESS[label_id]
#                 w_raw *= scale
#                 h_raw *= scale
                
#             # --- LED 强制对齐 ---
#             if h_raw > w_raw:
#                 w_raw, h_raw = h_raw, w_raw
#                 angle += (np.pi / 2) 
            
#             angle = (angle + np.pi) % (2 * np.pi) - np.pi
            
#             # 最终保存时，定位信息依然使用人工标注的真实点 cx, cy
#             img_pseudo_boxes.append([cx, cy, w_raw, h_raw, angle, label_id])
#             total_objs += 1
            
#         pseudo_data_dict[file_id] = img_pseudo_boxes
#         valid_count += 1

#     # 6. Save to PKL
#     print(f"\nProcessing Complete.")
#     print(f"Total Images: {valid_count}")
#     print(f"Total Pseudo RBoxes: {total_objs}")
#     print(f"Saving to {SAVE_PATH} ...")
#     with open(SAVE_PATH, 'wb') as f:
#         pickle.dump(pseudo_data_dict, f)
#     print("Done.")

# if __name__ == '__main__':
#     generate_pseudo_labels()

# GT中心
import torch
import numpy as np
import cv2
import os
import tqdm
import random
import sys
import pickle  
import math
import torch.nn.functional as F

# ==============================================================================
# 1. 配置区域 
# ==============================================================================
DATA_ROOT = '/mnt/data/liurunxiang/dataset/split_ss_dota/trainval/'
SAVE_PATH = '/mnt/data/liurunxiang/dataset/split_ss_dota/trainval/dota1-Rect.pkl'
DEVICE = 'cuda:1'

# 🌟 新增功能：在这里手动选择生成方法！
# 可选值：'Row5_PCA2' 或 'Row6_Rect'
METHOD = 'Row6_Rect'  

# 当前方法的最优参数
# CFG = dict(sigma=4096, ds=2, pos=0.996, neg=0.1) #PCA
CFG = dict(sigma=4096, ds=2, pos=0.996, neg=0.009) #Rect

# 后处理配置 (直接应用到伪标签中)
POST_PROCESS = {
    11: 1.2  # Roundabout 放大 1.2 倍
}

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
           'harbor', 'swimming-pool', 'helicopter')

# ==============================================================================
# 2. 核心算法函数
# ==============================================================================

def rbox2sigma(w, h, angle, device):
    w = torch.tensor(w, device=device, dtype=torch.float32)
    h = torch.tensor(h, device=device, dtype=torch.float32)
    angle = torch.tensor(angle, device=device, dtype=torch.float32)
    cos_a = torch.cos(angle); sin_a = torch.sin(angle)
    R = torch.stack([torch.stack([cos_a, -sin_a], -1), torch.stack([sin_a, cos_a], -1)], -2)
    Lambda = torch.stack([torch.stack([(w/2)**2, torch.zeros_like(w)], -1), 
                          torch.stack([torch.zeros_like(w), (h/2)**2], -1)], -2)
    return R @ Lambda @ R.transpose(-1, -2)

def get_watershed_wh_original_loss(mask_pts, center_pt, angle_rad):
    """投影测量法"""
    if len(mask_pts) < 2: return 16.0, 16.0 # Fallback for tiny masks
    pts_centered = mask_pts - center_pt
    cos_inv = np.cos(-angle_rad); sin_inv = np.sin(-angle_rad)
    R_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])
    pts_rot = np.dot(pts_centered, R_inv.T)
    w = max(2 * np.max(np.abs(pts_rot[:, 0])), 4.0) # Min size constraint
    h = max(2 * np.max(np.abs(pts_rot[:, 1])), 4.0)
    return w, h

def gaussian_2d(xy, mu, sigma):
    dxy = (xy - mu).unsqueeze(-1)
    inv_sigma = torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)
    return torch.exp(-0.5 * dxy.permute(0, 1, 3, 2).matmul(inv_sigma)).squeeze()

def run_watershed_generation(sigmas, gt_points, img_shape, pos_thres, neg_thres, down_sample, device):
    H, W = img_shape[:2]; D = down_sample; J = len(gt_points)
    h_map, w_map = H // D, W // D
    if J == 0: return np.zeros((H, W), dtype=np.int32)
    
    ax_x = torch.linspace(0, w_map-1, w_map, device=device)
    ax_y = torch.linspace(0, h_map-1, h_map, device=device)
    grid_x, grid_y = torch.meshgrid(ax_x, ax_y, indexing='xy')
    xy = torch.stack([grid_x, grid_y], -1)
    
    mm = (torch.tensor(gt_points, device=device).float() / D).round()
    sg = sigmas / (D**2)
    
    vor = torch.zeros((J, h_map, w_map), device=device)
    for j in range(J): vor[j] = gaussian_2d(xy, mm[j], sg[j])
        
    val, vor_idx = torch.max(vor, 0)
    if D > 1:
        vor_idx = F.interpolate(vor_idx[None, None].float(), size=(H, W), mode='nearest')[0, 0].long()
        val = F.interpolate(val[None, None], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
    kernel = torch.ones((1, 1, 3, 3), device=device); kernel[0,0,1,1] = -8
    ridges = torch.conv2d(vor_idx.float()[None,None], kernel, padding=1)[0,0] != 0
    
    vor_idx += 1
    vor_idx[val < pos_thres] = 0
    vor_idx[val < neg_thres] = J + 1
    vor_idx[ridges] = J + 1
    
    return vor_idx.cpu().numpy().astype(np.int32)

def analyze_geometry_specific(mask_stage1, gt_points, default_sigma, device, method):
    """
    通用几何分析：支持 PCA2 和 Rect 自由切换，恢复最稳妥的 np.where 提取
    """
    J = len(gt_points)
    sigmas_list = []
    angles_list = []
    
    for j in range(J):
        pts_y, pts_x = np.where(mask_stage1 == j + 1)
        
        # --- 异常处理：Fallback ---
        if len(pts_x) < 5:
            fb_s = torch.eye(2, device=device) * default_sigma
            sigmas_list.append(fb_s)
            angles_list.append(0.0)
            continue
            
        mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
        gt_center = gt_points[j]
        
        # --- 根据选择的方法提取角度 ---
        if method == 'Row6_Rect':
            rect = cv2.minAreaRect(mask_pts)
            _, _, angle_deg_rect = rect
            # 【核心修复】：物理长边对齐
            if angle_deg_rect > 45: 
                angle_deg_rect -= 90
            theta = np.radians(angle_deg_rect)
            
        elif method == 'Row5_PCA2':
            pts_c = mask_pts - np.mean(mask_pts, axis=0)
            cov = np.dot(pts_c.T, pts_c) / (mask_pts.shape[0] - 1)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            v_sub = eig_vecs[:, 0] # PCA 次轴
            theta = np.arctan2(v_sub[1], v_sub[0])
            
        else:
            theta = 0.0

        # 构建 Stage 2 需要的 Sigma (严格锚定 GT Center)
        w_raw, h_raw = get_watershed_wh_original_loss(mask_pts, gt_center, theta)
        area_raw = (w_raw / 2) * (h_raw / 2)
        scale = np.sqrt(default_sigma / (area_raw + 1e-6))
        
        sigmas_list.append(rbox2sigma(w_raw*scale, h_raw*scale, theta, device))
        angles_list.append(theta)
        
    return torch.stack(sigmas_list), np.array(angles_list)

# ==============================================================================
# 3. 数据生成主流程
# ==============================================================================

def generate_pseudo_labels():
    print(f"Start Generating Pseudo Labels for DOTA...")
    print(f"Selected Method: {METHOD}")
    print(f"Config: {CFG}")
    print(f"Output: {SAVE_PATH}")
    
    img_dir = os.path.join(DATA_ROOT, 'images')
    lbl_dir = os.path.join(DATA_ROOT, 'labelTxt')
    files = sorted([os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    
    pseudo_data_dict = {}
    valid_count = 0
    total_objs = 0
    
    for i, file_id in enumerate(tqdm.tqdm(files)):
        img_path = os.path.join(img_dir, f"{file_id}.png")
        if not os.path.exists(img_path): img_path = os.path.join(img_dir, f"{file_id}.jpg")
        if not os.path.exists(img_path): continue
        
        gt_points, gt_labels = [], []
        try:
            with open(os.path.join(lbl_dir, f"{file_id}.txt"), 'r') as f:
                for ln in f:
                    p = ln.strip().split()
                    if len(p) < 9 or p[8] not in CLASSES: continue
                    poly = np.array(list(map(float, p[:8]))).reshape(4, 2)
                    gt_points.append(poly.mean(0))
                    gt_labels.append(CLASSES.index(p[8]))
        except: continue
        
        if len(gt_points) == 0:
            pseudo_data_dict[file_id] = []
            continue
            
        gt_points = np.array(gt_points)
        J = len(gt_points)
        img = cv2.medianBlur(cv2.imread(img_path), 3)
        H, W = img.shape[:2]
        
        # 2. Stage 1 (Standard)
        s_std = torch.eye(2, device=DEVICE).unsqueeze(0).repeat(J, 1, 1) * CFG['sigma']
        mask_s1 = run_watershed_generation(s_std, gt_points, img.shape, CFG['pos'], CFG['neg'], CFG['ds'], DEVICE)
        cv2.watershed(img, mask_s1)
        
        # 3. Geometry Analysis (动态使用选择的方法)
        sigmas_s2, angles_s2 = analyze_geometry_specific(mask_s1, gt_points, CFG['sigma'], DEVICE, method=METHOD)
        
        # 4. Stage 2 (Refined)
        mask_s2 = run_watershed_generation(sigmas_s2, gt_points, img.shape, CFG['pos'], CFG['neg'], CFG['ds'], DEVICE)
        cv2.watershed(img, mask_s2)
        
        # 5. Final Measurement & Storage
        img_pseudo_boxes = []
        
        # 恢复使用最稳健的 np.where 提取
        for idx in range(J):
            py, px = np.where(mask_s2 == idx + 1)
            label_id = gt_labels[idx]
            cx, cy = gt_points[idx]
            
            # --- Fallback for bad mask ---
            # 如果没长出来，依然要给个默认小框，防止检测器把这个点漏掉
            if len(px) < 3:
                img_pseudo_boxes.append([cx, cy, 16.0, 16.0, 0.0, label_id])
                total_objs += 1
                continue
            
            m_pts = np.stack([px, py], axis=1).astype(np.float32)
            angle = angles_s2[idx]
            
            # 【🔥 核心修复】：严格锚定人工标注真实点，拒绝水团重心的漂移！
            center = np.array([cx, cy]) 
            w_raw, h_raw = get_watershed_wh_original_loss(m_pts, center, angle)
            
            # --- 后处理 (直接应用) ---
            if label_id in POST_PROCESS:
                scale = POST_PROCESS[label_id]
                w_raw *= scale
                h_raw *= scale
                
            # --- LED 强制对齐 ---
            if h_raw > w_raw:
                w_raw, h_raw = h_raw, w_raw
                angle += (np.pi / 2) 
            
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            
            # 保存格式: [x, y, w, h, angle, label]
            img_pseudo_boxes.append([cx, cy, w_raw, h_raw, angle, label_id])
            total_objs += 1
            
        pseudo_data_dict[file_id] = img_pseudo_boxes
        valid_count += 1

    # 6. Save to PKL
    print(f"\nProcessing Complete.")
    print(f"Total Images: {valid_count}")
    print(f"Total Pseudo RBoxes: {total_objs}")
    print(f"Saving to {SAVE_PATH} ...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(pseudo_data_dict, f)
    print("Done.")

if __name__ == '__main__':
    generate_pseudo_labels()