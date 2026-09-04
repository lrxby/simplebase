import os
import cv2
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

# ==============================================================================
# 1. 核心算法逻辑 (保持不变)
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
    
    vor_maps = torch.zeros((J, h, w), device=device)
    mm = (mu / D).round()
    sg = sigma / (D ** 2)
    
    for j in range(J):
        dxy = (xy.view(-1, 2) - mm[j][None]).unsqueeze(-1)
        vor_maps[j] = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sg[j][None], dxy))).view(h, w)
        
    val, vor_idx = torch.max(vor_maps, 0)
    if D > 1:
        vor_idx = F.interpolate(vor_idx[None, None, ...].float(), size=(H, W), mode='nearest')[0, 0].long()
        val = F.interpolate(val[None, None, ...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
    return val, vor_idx

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

# ==============================================================================
# 2. 绘图工具 (功能拆分)
# ==============================================================================

def draw_masks(img, markers, num_instances):
    """左图：只画注水盆地 (Masks)"""
    vis_img = img.copy()
    # 为每个实例生成随机颜色
    colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_instances + 1)]

    for idx in range(num_instances):
        inst_id = idx + 1
        mask = (markers == inst_id)
        
        if not np.any(mask): continue

        # 画半透明 Mask
        color = colors[idx]
        # 0.6 原图 + 0.4 颜色
        vis_img[mask] = vis_img[mask] * 0.6 + np.array(color) * 0.4
        
    return vis_img

def draw_boxes(img, markers, gt_polys):
    """右图：只画框 (Pred vs GT)"""
    vis_img = img.copy()
    num_instances = len(gt_polys)

    for idx in range(num_instances):
        inst_id = idx + 1
        
        # 1. 绘制 GT 框 (红色, 线宽 2)
        gt_poly = np.array(gt_polys[idx]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(vis_img, [gt_poly], True, (0, 0, 255), 2)

        # 2. 绘制 预测框 (绿色, 线宽 2)
        # 从 markers 中提取点集
        y_idx, x_idx = np.where(markers == inst_id)
        pts = np.stack([x_idx, y_idx], axis=1)
        
        if len(pts) >= 3:
            # 计算最小外接矩形
            rect = cv2.minAreaRect(pts.astype(np.float32))
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(vis_img, [box], True, (0, 255, 0), 2)
            
            # 可选：画中心点方便对齐
            center = np.mean(pts, axis=0).astype(int)
            cv2.circle(vis_img, tuple(center), 3, (0, 255, 0), -1)

    return vis_img

# ==============================================================================
# 3. 主程序
# ==============================================================================

def main():
    # 配置
    data_root = '/mnt/data/xiekaikai/split_ss_dota/trainval'
    save_dir = './1024'
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设定的超参数
    params = {
        'pos_thres': 0.95,
        'neg_thres': 0.01,
        'down_sample': 2,
        'default_sigma': 1024
    }

    img_dir = os.path.join(data_root, 'images')
    lbl_dir = os.path.join(data_root, 'labelTxt')
    all_files = [f[:-4] for f in os.listdir(lbl_dir) if f.endswith('.txt')]
    
    # 随机挑选 50 张
    if len(all_files) > 30:
        selected_files = random.sample(all_files, 30)
    else:
        selected_files = all_files
        
    print(f"Visualizing {len(selected_files)} images to {save_dir}...")

    for file_id in tqdm(selected_files):
        # 加载图片
        img_path = os.path.join(img_dir, f"{file_id}.png")
        if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
        if not os.path.exists(img_path): continue
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 加载标注
        points, gt_polys = [], []
        lbl_path = os.path.join(lbl_dir, f"{file_id}.txt")
        try:
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 9: continue
                    poly = list(map(float, parts[:8]))
                    poly_np = np.array(poly).reshape(4, 2)
                    points.append(np.mean(poly_np, axis=0))
                    gt_polys.append(poly)
        except: continue

        if len(points) == 0: continue

        # 生成注水盆地
        with torch.no_grad():
            res = get_gaussian_base(points, img.shape[:2], params['default_sigma'], params['down_sample'], device)
            if res is None: continue
            val, vor_idx = res
            markers = fast_watershed_fit(val, vor_idx, img, params['pos_thres'], params['neg_thres'], len(points))

        # --- 生成两张图 ---
        # 左图：Masks
        img_masks = draw_masks(img, markers, len(points))
        # 右图：Boxes
        img_boxes = draw_boxes(img, markers, gt_polys)

        # --- 拼接并保存 ---
        plt.figure(figsize=(24, 12)) # 调大画布
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_masks, cv2.COLOR_BGR2RGB))
        plt.title(f"Watershed Basins (Masks)\npos={params['pos_thres']} | neg={params['neg_thres']} | sig={params['default_sigma']}", fontsize=14)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Box Comparison\nGreen: Predicted (from Mask) | Red: Ground Truth", fontsize=14)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{file_id}_vis.png"), dpi=100)
        plt.close()

if __name__ == '__main__':
    main()