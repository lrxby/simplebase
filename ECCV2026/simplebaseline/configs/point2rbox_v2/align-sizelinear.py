# align-sizelinear.py: 对齐实验③ Eq.4-6 SizeLoss 回归空间
# 论文: 线性直接回归 w (k_x*x + k_y*y + b_c = w)
# 代码: y_w=log(w), y_h=log(h) 对数空间
# 仅改 SizeLoss.reg_space='linear', 其余与 audit-fixed-ow3.py 完全一致
_base_ = ['./audit-fixed-ow3.py']
model = dict(
    bbox_head=dict(
        loss_size=dict(type='SizeLoss', reg_space='linear')))
