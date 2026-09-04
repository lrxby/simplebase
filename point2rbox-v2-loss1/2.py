import cv2
import numpy as np
import os
import torch

# 配置您的路径
IMG_DIR = '/mnt/data/xiekaikai/DroneVehicle/train/images'
LBL_DIR = '/mnt/data/xiekaikai/DroneVehicle/train/annfiles/'
SAVE_DIR = 'debug_vis_drone'
os.makedirs(SAVE_DIR, exist_ok=True)

# 随便找一个 txt 文件名
SAMPLE_ID = '00005' # 请替换为您文件夹里真实存在的一个文件名（不带后缀）

def poly2rbox(poly):
    poly = np.array(poly).reshape(-1, 2).astype(np.float32)
    return cv2.minAreaRect(poly)

def visualize():
    img_path = os.path.join(IMG_DIR, f"{SAMPLE_ID}.jpg")
    lbl_path = os.path.join(LBL_DIR, f"{SAMPLE_ID}.txt")
    
    if not os.path.exists(img_path): img_path = img_path.replace('.jpg', '.png')
    
    print(f"Reading {img_path}...")
    img = cv2.imread(img_path)
    if img is None: 
        print("Image not found!")
        return

    print(f"Reading {lbl_path}...")
    points = []
    polys = []
    
    with open(lbl_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 8: continue
            p = list(map(float, parts[:8]))
            poly = np.array(p).reshape(4, 2).astype(np.int32)
            polys.append(poly)
            # 计算中心点
            center = np.mean(poly, axis=0).astype(np.int32)
            points.append(center)

    # 1. 画 GT 框 (绿色)
    cv2.polylines(img, polys, True, (0, 255, 0), 2)
    
    # 2. 画中心点 (红色)
    for pt in points:
        cv2.circle(img, (pt[0], pt[1]), 3, (0, 0, 255), -1)

    # 3. 模拟注水 (简单高斯热图)
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    for pt in points:
        # 画一个简单的白色圆代表高斯核
        cv2.circle(heatmap, (pt[0], pt[1]), 20, 1, -1)
    
    # 叠加热图看看位置对不对
    heatmap_colored = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)

    save_path = os.path.join(SAVE_DIR, f"vis_{SAMPLE_ID}.jpg")
    cv2.imwrite(save_path, overlay)
    print(f"Saved visualization to {save_path}")
    print("Please check this image. If the red dots are NOT inside the green boxes, or if they are totally misaligned with the cars, then the coordinates are WRONG.")

if __name__ == '__main__':
    visualize()
