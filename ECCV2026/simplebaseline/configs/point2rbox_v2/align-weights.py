# align-weights.py: 对齐实验① Eq.12 总损失权重
# 论文: L_cls + L_center + 0.2*L_angle + 0.1*L_size + 0.5*L_RBox
# 方案: 配置写论文权重 0.2/0.1/0.5, loss 类内部 scale_factor 缩放, 实际等效 1.0/1.0/3.0
#   angle:  0.2 * 5.0 = 1.0
#   size:   0.1 * 10.0 = 1.0
#   ourwater: 0.5 * 6.0 = 3.0  (与 ow3 完全一致)
# 仅改权重表示方式这一处, 其余与 audit-fixed-ow3.py 完全一致
_base_ = ['./audit-fixed-ow3.py']
model = dict(
    bbox_head=dict(
        loss_angle=dict(type='AngleLoss', loss_weight=0.2, scale_factor=5.0),
        loss_size=dict(type='SizeLoss', loss_weight=0.1, scale_factor=10.0),
        loss_ourwater=dict(type='OurWaterLoss', loss_weight=0.5, scale_factor=6.0)))
