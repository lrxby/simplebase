# ---------------------------------------------------------------
# audit-fixed-noangle-s0.py
# 诊断实验: 继承修复后基线, 仅关闭 AngleLoss (开关, 非权重乘 0)
# 用于检验: 在监督正确的条件下, OurWaterLoss 能否独立学出方向
# ---------------------------------------------------------------
_base_ = './audit-fixed-baseline-s0.py'

# 仅关闭 AngleLoss, 其余 (seed/pipeline/损失/优化器/schedule) 与基线完全一致
model = dict(
    bbox_head=dict(
        use_angle_loss=False
    )
)

# 独立输出目录提示 (训练时仍以 --work-dir 为准)
