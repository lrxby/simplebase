# align-neighbor.py: 对齐实验② Eq.2 邻域权重
# 论文: W(i,j)=exp(-(|dx|+|dy|)/(2*w_i*h_i)) 曼哈顿
# 代码: exp(-d^2/(8wh)) 欧氏平方
# 仅改 AngleLoss.neighbor_metric='manhattan', 其余与 audit-fixed-ow3.py 完全一致
_base_ = ['./audit-fixed-ow3.py']
model = dict(
    bbox_head=dict(
        loss_angle=dict(type='AngleLoss', neighbor_metric='manhattan')))
