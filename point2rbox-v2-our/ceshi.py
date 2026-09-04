import torch
import numpy as np
import cv2
import os
import tqdm
import random
import math
from shapely.geometry import Polygon
import torch.nn.functional as F

# ==============================================================================
# 1. 全局配置
# ==============================================================================
NUM_TEST_IMAGES = -1  # 测试图片数量
DATA_ROOT = '/mnt/data/xiekaikai/split_ss_dota/trainval'
DEVICE = 'cuda:1'

CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
           'harbor', 'swimming-pool', 'helicopter')

# --- 严格复刻原版 Config ---
BASELINE_CFG = dict(
    # [核心差异] Standard Mode 使用各向同性大 Sigma
    voronoi_type='standard', 
    default_sigma=2048, 
    down_sample=2,
    
    # 阈值配置
    thres_default=[0.98, 0.05],
    thres_override=[
        ([2, 11], [0.999, 0.6]),        # bridge, roundabout
        ([7, 8, 10, 14], [0.95, 0.005]) # tennis, basketball, soccer, helicopter
    ],
    
    # 特殊类别处理
    square_cls=[1, 9, 11], # baseball, storage-tank, roundabout
    post_process={11: 1.2} # roundabout scaling
)

# ==============================================================================
# 2. 核心工具函数
# ==============================================================================

def get_thres_for_label(label_id):
    """获取分类别阈值"""
    for cls_list, thres in BASELINE_CFG['thres_override']:
        if label_id in cls_list:
            return thres[0], thres[1]
    return BASELINE_CFG['thres_default'][0], BASELINE_CFG['thres_default'][1]

def poly2rbox_angle_deg(poly):
    """将 GT Poly 转为 (cx, cy, w, h, angle_deg)"""
    poly = np.array(poly).reshape(-1, 2).astype(np.float32)
    return cv2.minAreaRect(poly)

def rbox2poly_custom(cx, cy, w, h, angle_deg):
    """将 (cx, cy, w, h, angle_deg) 转为 4点坐标"""
    rect = ((cx, cy), (w, h), angle_deg)
    return cv2.boxPoints(rect)

def compute_iou_shapely(box1, box2):
    try:
        p1 = Polygon(box1)
        p2 = Polygon(box2)
        if not p1.is_valid or not p2.is_valid: return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / (union + 1e-6)
    except:
        return 0.0

def calc_angle_diff(angle1_deg, angle2_deg):
    """计算角度偏差 (考虑 180 度周期性)"""
    diff = abs(angle1_deg - angle2_deg) % 180
    diff = min(diff, 180 - diff)
    return diff

def gaussian_2d(xy, mu, sigma):
    dxy = (xy - mu).unsqueeze(-1)
    # 简单的对角阵求逆可优化，但为了通用性保持矩阵运算
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)))
    return t0

def generate_standard_mask(img_shape, gt_points, gt_labels, device='cuda'):
    """
    完全复刻原版 Standard Mode 生成逻辑
    特点：不依赖预测框，Sigma 固定且为正圆
    """
    H, W = img_shape[:2]
    D = BASELINE_CFG['down_sample']
    DEFAULT_SIGMA = BASELINE_CFG['default_sigma'] # 4096
    
    if len(gt_points) == 0:
        return np.zeros((H, W), dtype=np.int32)

    J = len(gt_points)
    gt_tensor = torch.tensor(gt_points, device=device, dtype=torch.float32)
    
    # 1. 构建 Standard Sigma (各向同性/圆形)
    # 对应原版代码：sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma))
    sigma = torch.eye(2, device=device).unsqueeze(0).repeat(J, 1, 1) * DEFAULT_SIGMA
    
    # 2. 缩放到特征图尺度
    mm = (gt_tensor / D).round()
    sg = sigma / (D ** 2)
    
    # 3. 生成 Voronoi 热图
    h_map, w_map = H // D, W // D
    x = torch.linspace(0, h_map-1, h_map, device=device)
    y = torch.linspace(0, w_map-1, w_map, device=device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    
    vor = torch.zeros((J, h_map, w_map), device=device)
    
    for j in range(J):
        vor[j] = gaussian_2d(xy.view(-1, 2), mm[j][None], sg[j][None]).view(h_map, w_map)
        
    # 4. 应用分类别阈值
    val, vor_idx = torch.max(vor, 0)
    
    if D > 1:
        vor_idx = F.interpolate(vor_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
        val = F.interpolate(val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
    kernel = torch.ones((1, 1, 3, 3), device=device)
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor_idx.float()[None, None], kernel, padding=1)[0, 0] != 0
    
    vor_idx += 1
    
    # 构建全图阈值 Map
    pos_map = torch.full_like(val, BASELINE_CFG['thres_default'][0])
    neg_map = torch.full_like(val, BASELINE_CFG['thres_default'][1])
    
    inst_pos = []
    inst_neg = []
    for lbl in gt_labels:
        p, n = get_thres_for_label(lbl)
        inst_pos.append(p)
        inst_neg.append(n)
        
    pos_map[vor_idx>0] = torch.tensor(inst_pos, device=device)[vor_idx[vor_idx>0]-1]
    neg_map[vor_idx>0] = torch.tensor(inst_neg, device=device)[vor_idx[vor_idx>0]-1]
    
    vor_idx[val < pos_map] = 0
    vor_idx[val < neg_map] = J + 1
    vor_idx[ridges] = J + 1
    
    # 5. Watershed
    markers_np = vor_idx.cpu().numpy().astype(np.int32)
    dummy_img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.watershed(dummy_img, markers_np)
    
    return markers_np

# ==============================================================================
# 3. 数据加载
# ==============================================================================

class DOTALoader:
    def __init__(self, root_path, sample_num=-1):
        self.img_dir = os.path.join(root_path, 'images')
        self.label_dir = os.path.join(root_path, 'labelTxt')
        all_files = [os.path.splitext(f)[0] for f in os.listdir(self.label_dir) if f.endswith('.txt')]
        if sample_num > 0 and sample_num < len(all_files):
            print(f"Randomly sampling {sample_num} images...")
            self.files = random.sample(all_files, sample_num)
        else:
            print(f"Loading ALL {len(all_files)} images...")
            self.files = all_files

    def load_item(self, idx):
        file_id = self.files[idx]
        img_path = os.path.join(self.img_dir, f"{file_id}.png")
        if not os.path.exists(img_path): return None
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        points, gt_boxes, gt_labels = [], [], []
        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 9: continue
                    poly = list(map(float, parts[:8]))
                    cls_name = parts[8]
                    if cls_name not in CLASSES: continue
                    cls_id = CLASSES.index(cls_name)
                    
                    poly_np = np.array(poly, dtype=np.float32).reshape(4, 2)
                    cx, cy = np.mean(poly_np, axis=0)
                    
                    points.append([cx, cy])
                    gt_boxes.append(poly_np) 
                    gt_labels.append(cls_id)
        except: return None
        return {'img_path': img_path, 'points': np.array(points), 'gt_boxes': gt_boxes, 'gt_labels': gt_labels}

# ==============================================================================
# 4. 主统计循环
# ==============================================================================

def run_standard_baseline_analysis():
    loader = DOTALoader(DATA_ROOT, sample_num=NUM_TEST_IMAGES)
    
    # 统计容器
    cls_stats = {k: {'iou': 0.0, 'angle_diff': 0.0, 'count': 0} for k in CLASSES}
    
    print(f"Starting STANDARD Baseline Analysis (Circular Sigma, minAreaRect)...")
    
    for i in tqdm.tqdm(range(len(loader.files))):
        data = loader.load_item(i)
        if data is None: continue
        
        img = cv2.imread(data['img_path'])
        if img is None: continue
        
        # 1. 生成 Mask (Standard Mode)
        mask = generate_standard_mask(
            img.shape, data['points'], data['gt_labels'], device=DEVICE)
        
        for idx, (gt_poly, label_id) in enumerate(zip(data['gt_boxes'], data['gt_labels'])):
            label_name = CLASSES[label_id]
            inst_id = idx + 1
            
            # 2. 提取 Mask
            pts_y, pts_x = np.where(mask == inst_id)
            if len(pts_x) < 3: 
                # 没生成出来，算 0
                cls_stats[label_name]['count'] += 1
                continue
                
            mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
            
            # 3. 拟合最小外接矩形 (minAreaRect)
            # 因为是冷启动，没有预测角度，只能靠 mask 自己
            rect = cv2.minAreaRect(mask_pts)
            ((cx, cy), (w, h), angle_deg) = rect
            
            # 4. 应用特殊配置
            # 4.1 强制角度为 0
            if label_id in BASELINE_CFG['square_cls']:
                angle_deg = 0.0
                
            # 4.2 尺寸缩放
            if label_id in BASELINE_CFG['post_process']:
                scale = BASELINE_CFG['post_process'][label_id]
                w *= scale
                h *= scale
                
            # 5. 生成预测框 & 计算 IoU
            pred_box = rbox2poly_custom(cx, cy, w, h, angle_deg)
            iou = compute_iou_shapely(pred_box, gt_poly)
            
            # 6. 计算角度偏差
            if label_id in BASELINE_CFG['square_cls']:
                ang_diff = 0.0
            else:
                # 获取 GT 的 minAreaRect 角度作为基准
                gt_rect = cv2.minAreaRect(gt_poly)
                gt_angle = gt_rect[2]
                ang_diff = calc_angle_diff(angle_deg, gt_angle)
            
            # 7. 记录
            cls_stats[label_name]['iou'] += iou
            cls_stats[label_name]['angle_diff'] += ang_diff
            cls_stats[label_name]['count'] += 1
            
    # --- 输出报表 ---
    print("\n" + "="*70)
    print(f"{'Class':<20} | {'Mean IoU':<10} | {'Ang Diff(°)':<12} | {'Count':<6}")
    print("-" * 70)
    
    total_iou = 0
    total_ang = 0
    total_n = 0
    
    for k in CLASSES:
        s = cls_stats[k]
        n = s['count']
        if n > 0:
            avg_iou = s['iou'] / n
            avg_ang = s['angle_diff'] / n
            total_iou += s['iou']
            total_ang += s['angle_diff']
            total_n += n
        else:
            avg_iou = 0.0; avg_ang = 0.0
            
        print(f"{k:<20} | {avg_iou:.4f}     | {avg_ang:.2f}         | {n:<6}")
        
    print("-" * 70)
    print(f"OVERALL | IoU: {total_iou/max(1,total_n):.4f} | Ang Diff: {total_ang/max(1,total_n):.2f}°")
    print("="*70)

if __name__ == '__main__':
    run_standard_baseline_analysis()
