# align-sizel2.py: 对齐实验④ Eq.7 SizeLoss 损失形式
# 论文: ||X W_hat - Y||^2 (L2/MSE)
# 代码: nn.SmoothL1Loss(beta=1.0)
# 仅改 SizeLoss.loss_type='l2', 其余与 audit-fixed-ow3.py 完全一致
_base_ = ['./audit-fixed-ow3.py']
model = dict(
    bbox_head=dict(
        loss_size=dict(type='SizeLoss', loss_type='l2')))
