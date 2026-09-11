# align-eps.py: 对齐实验⑤ Eq.3 AngleLoss 归一化 eps
# 论文: eps=0.01
# 代码: 1e-6
# 仅改 AngleLoss.eps=0.01, 其余与 audit-fixed-ow3.py 完全一致
_base_ = ['./audit-fixed-ow3.py']
model = dict(
    bbox_head=dict(
        loss_angle=dict(type='AngleLoss', eps=0.01)))
