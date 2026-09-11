# audit-fixed-ow7.py: ourwater 权重升到 7（G3 动态决策备用）
_base_ = ['./audit-fixed-baseline-s0.py']
model = dict(
    bbox_head=dict(
        loss_ourwater=dict(type='OurWaterLoss', loss_weight=7.0)))
