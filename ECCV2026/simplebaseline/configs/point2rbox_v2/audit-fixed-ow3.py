# audit-fixed-ow3.py: 修复后代码上复验 ourwater 权重 3（方案 C）
# 单变量: loss_ourwater.loss_weight 1.0 -> 3.0，其余与 audit-fixed-baseline-s0 完全一致
_base_ = ['./audit-fixed-baseline-s0.py']
model = dict(
    bbox_head=dict(
        loss_ourwater=dict(type='OurWaterLoss', loss_weight=3.0)))
