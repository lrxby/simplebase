# check_diff_roi.py
import torch
from mmrotate.models.utils.differentiable_roi_align import DifferentiableRoIAlignRotated

def check():
    device = torch.device("cuda:0")
    # 1. 特征图 (需梯度)
    feat = torch.randn(1, 1, 10, 10, device=device, requires_grad=True)
    # 2. ROI (需梯度)
    rois = torch.tensor([[0.0, 5.0, 5.0, 4.0, 2.0, 0.5]], dtype=torch.float32, device=device)
    rois.requires_grad = True
    
    # 3. 初始化您的模块
    diff_roi_align = DifferentiableRoIAlignRotated(
        output_size=(3, 3), 
        spatial_scale=1.0, 
        sampling_ratio=1 # 单点采样
    )
    
    # 4. 前向 & 反向
    out = diff_roi_align(feat, rois)
    loss = out.sum()
    loss.backward()
    
    print(f"Output shape: {out.shape}")
    
    # 5. 验证
    if rois.grad is not None and rois.grad.abs().sum() > 0:
        print("✅ [SUCCESS] DifferentiableRoIAlignRotated 对 ROI 坐标【可导】！")
        print(f"   ROIs Grad: {rois.grad}")
    else:
        print("❌ [FAIL] 依然不可导，请检查代码。")

if __name__ == '__main__':
    check()