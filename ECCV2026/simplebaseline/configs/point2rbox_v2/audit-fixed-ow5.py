# audit-fixed-ow5.py: 继续升 ourwater 权重到 5（动态决策备用）
# 单变量: loss_ourwater.loss_weight 1.0 -> 5.0
_base_ = ['./audit-fixed-baseline-s0.py']
model = dict(
    bbox_head=dict(
        loss_ourwater=dict(type='OurWaterLoss', loss_weight=5.0)))
